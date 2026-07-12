"""Safe, unattended, read-only workspace inspection tools.

These are the project's first model-facing source tools. They run without an
approval interrupt but only under a ``source_read`` attestation with a loopback
model endpoint. Every path is confined to the opened workspace, sensitive files
are denied, links and reparse points are refused, and content is strict-decoded
and redacted before it can reach a model.

The observation is the durable unit of evidence: each returned ``ToolObservation``
carries an opaque id, the workspace-relative path, a content hash, numbered
lines, and truncation/redaction state. Final answers may only cite content that
an earlier observation actually returned.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from inverse_agent.redaction import SECRET_PATTERNS

# Token-denominated bounds (a chars/token upper-bound heuristic keeps reads inside
# a small local context window). Byte limits are defensive backstops.
CHARS_PER_TOKEN = 4
READ_MAX_LINES = 200
READ_MAX_TOKENS = 3_000
READ_MAX_BYTES = READ_MAX_TOKENS * CHARS_PER_TOKEN
FILE_MAX_BYTES = 1024 * 1024
LIST_MAX_ENTRIES = 500
SEARCH_MAX_MATCHES = 100
SEARCH_MAX_FILES = 2_000
SEARCH_MAX_SCAN_BYTES = 8 * 1024 * 1024
# Upper bound on directory entries visited during a single recursive walk, so a
# huge tree cannot cause unbounded latency/memory before other caps apply.
WALK_VISIT_LIMIT = 50_000
SNIPPET_MAX_CHARS = 200
PATH_MAX_CHARS = 512
QUERY_MAX_CHARS = 256
GLOB_MAX_CHARS = 128
# Post-serialization ceiling for list/search results (~2,000 tokens backstop).
RESPONSE_MAX_BYTES = 16 * 1024

# Windows reserved device names (checked per path component, case-insensitively,
# ignoring any extension).
_WINDOWS_DEVICE_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)

# Directory names that are never traversed into (compared case-insensitively).
_DENIED_DIR_NAMES = frozenset(
    {".git", ".hg", ".svn", "__pycache__", ".venv", "node_modules", ".docker", ".aws", ".ssh"}
)

# Sensitive-file deny policy. Each future domain pack contributes patterns here.
_DENIED_NAME_PATTERNS = (
    re.compile(r"(?i)^\.env(\..+)?$"),
    re.compile(r"(?i)^\.envrc$"),
    re.compile(r"(?i)\.pem$"),
    re.compile(r"(?i)\.key$"),
    re.compile(r"(?i)\.p8$"),
    re.compile(r"(?i)\.keystore$"),
    re.compile(r"(?i)\.jks$"),
    re.compile(r"(?i)\.p12$"),
    re.compile(r"(?i)\.pfx$"),
    re.compile(r"(?i)\.mobileprovision$"),
    re.compile(r"(?i)^id_(rsa|dsa|ecdsa|ed25519)$"),
    re.compile(r"(?i)^google-services\.json$"),
    re.compile(r"(?i)^local\.properties$"),
    re.compile(r"(?i)^\.npmrc$"),
    re.compile(r"(?i)^\.netrc$"),
    re.compile(r"(?i)^credentials(\.json)?$"),
)

_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class FsToolError(ValueError):
    """A read-tool request violated a path, size, or content policy."""


class PolicyViolationError(FsToolError):
    """A security-relevant refusal (confinement, links, sensitive files).

    Distinct from benign, retryable errors (bad line range, missing file): a
    policy violation terminates the investigation as an immediate refusal.
    """


class StrictDecodeError(FsToolError):
    """A file could not be strict-decoded as UTF-8 and was refused."""


@dataclass(frozen=True)
class ToolObservation:
    """A durable, model-visible result of one read-tool call.

    ``incomplete`` is set when content was redacted or otherwise could not be
    delivered faithfully; ``truncated`` is a normal property of large results and
    does not by itself make an observation incomplete.
    """

    observation_id: str
    tool: str
    path: str
    content_hash: str
    text: str
    lines: tuple[str, ...] = ()
    start_line: int = 1
    truncated: bool = False
    incomplete: bool = False
    redacted: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


def _observation_id(tool: str, path: str, salt: str) -> str:
    digest = hashlib.sha256(f"{tool}\0{path}\0{salt}".encode()).hexdigest()
    return f"obs_{digest[:16]}"


def _estimate_tokens(text: str) -> int:
    # Conservative upper bound: never undercount.
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _reject_component(name: str) -> None:
    if not name or name in {".", ".."}:
        raise PolicyViolationError("path traversal is not permitted")
    if "\x00" in name:
        raise PolicyViolationError("path contains a null byte")
    if ":" in name:
        # NTFS alternate-data-stream syntax (name:stream) and drive-letter syntax.
        raise PolicyViolationError("path contains an alternate-data-stream or drive separator")
    if name != name.rstrip(" ."):
        raise PolicyViolationError("path component has a trailing dot or space alias")
    if name.lower() in _DENIED_DIR_NAMES:
        raise PolicyViolationError(f"path traverses a denied directory: {name}")
    stem = name.split(".", 1)[0].lower()
    if stem in _WINDOWS_DEVICE_NAMES:
        raise PolicyViolationError(f"path uses a reserved device name: {name}")


def _reject_sensitive_name(name: str) -> None:
    for pattern in _DENIED_NAME_PATTERNS:
        if pattern.search(name):
            raise PolicyViolationError(f"file is denied by the sensitive-file policy: {name}")


def _is_reparse_point(entry: os.stat_result) -> bool:
    attributes = getattr(entry, "st_file_attributes", 0)
    if attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT:
        return True
    return stat.S_ISLNK(entry.st_mode)


def _relative_parts(workspace: Path, raw_path: str) -> tuple[str, ...]:
    if not raw_path or len(raw_path) > PATH_MAX_CHARS:
        raise PolicyViolationError("path is empty or exceeds the length limit")
    if "\x00" in raw_path:
        raise PolicyViolationError("path contains a null byte")
    normalized = raw_path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or (len(normalized) >= 2 and normalized[1] == ":"):
        raise PolicyViolationError("absolute paths are not permitted")
    parts = tuple(part for part in pure.parts if part not in ("", "."))
    for part in parts:
        _reject_component(part)
    return parts


def _resolve_within(workspace: Path, raw_path: str) -> tuple[Path, str]:
    """Resolve a workspace-relative path component by component, refusing links.

    Returns the resolved filesystem path and its canonical workspace-relative
    form. Traversal is anchored at the workspace root and every intermediate
    component is checked (non-reparse, directory) before descending, then the
    fully realpath-resolved target is confirmed to still lie inside the root and
    to use only canonical (non-alias) names. This is a check-then-open design
    with a realpath containment backstop; the spec's fully handle-relative
    (``FILE_FLAG_OPEN_REPARSE_POINT`` / ``OBJ_DONT_REPARSE``) open that closes the
    residual mid-traversal swap race is deferred to the hardened v0.2b build.
    """

    parts = _relative_parts(workspace, raw_path)
    if not parts:
        raise PolicyViolationError("path must reference a file or directory inside the workspace")
    current = workspace
    for index, part in enumerate(parts):
        current = current / part
        try:
            entry = current.lstat()
        except OSError as exc:
            raise FsToolError("path does not exist inside the workspace") from exc
        if _is_reparse_point(entry):
            raise PolicyViolationError("path component is a symlink, junction, or reparse point")
        is_last = index == len(parts) - 1
        if not is_last and not stat.S_ISDIR(entry.st_mode):
            raise PolicyViolationError("path traverses a non-directory")
    # Canonicalize: expand 8.3 aliases and confirm the resolved path is still
    # inside the root, then run every policy check against the canonical name.
    canonical = _canonical_relative(workspace, current)
    if tuple(name.lower() for name in canonical) != tuple(name.lower() for name in parts):
        raise PolicyViolationError("path uses a short-name alias or non-canonical form")
    for name in canonical:
        _reject_component(name)
    return current, "/".join(canonical)


def _looks_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    sample = data[:4096]
    if not sample:
        return False
    text_bytes = bytes({7, 8, 9, 10, 12, 13, 27, *range(0x20, 0x100)})
    nontext = sample.translate(None, text_bytes)
    return len(nontext) / len(sample) > 0.30


def _decode_strict(data: bytes) -> str:
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictDecodeError(
            "file is not valid UTF-8; refusing to deliver replacement bytes"
        ) from exc
    # Normalize line endings so numbered lines are stable across CRLF/CR files.
    return decoded.replace("\r\n", "\n").replace("\r", "\n")


def _sanitize_line_preserving(text: str) -> tuple[str, bool]:
    """Redact secrets over the FULL text while preserving the line count.

    Redacting line by line would defeat the multi-line private-key patterns
    (only the ``BEGIN`` header would match), leaking the key body. Instead the
    secret patterns run over the whole text and each replacement re-inserts the
    same number of newlines it covered, so downstream line numbers stay valid.
    """

    redacted = {"any": False}

    def _replace(match: re.Match[str]) -> str:
        redacted["any"] = True
        return "[REDACTED_SECRET]" + "\n" * match.group(0).count("\n")

    result = text
    for _name, pattern in SECRET_PATTERNS:
        result = pattern.sub(_replace, result)
    return result, redacted["any"]


def _canonical_relative(workspace: Path, resolved: Path) -> tuple[str, ...]:
    """Return the canonical (long-name, link-resolved) workspace-relative parts.

    On Windows this expands 8.3 short-name aliases and resolves any residual
    reparse indirection, so name-based deny checks cannot be evaded with an
    alias, and confirms the final path is still inside the workspace root.
    """

    final = Path(os.path.realpath(resolved))
    root = Path(os.path.realpath(workspace))
    try:
        relative = final.relative_to(root)
    except ValueError as exc:
        raise PolicyViolationError("resolved path escapes the workspace root") from exc
    return relative.parts


@dataclass(frozen=True)
class WorkspaceReader:
    """A source_read-scoped reader bound to one resolved workspace root."""

    workspace: Path
    root_id: tuple[int, int]

    @classmethod
    def open(cls, workspace: Path) -> WorkspaceReader:
        resolved = workspace.resolve()
        if not resolved.is_dir():
            raise FsToolError("workspace is not an existing directory")
        entry = resolved.stat()
        return cls(resolved, (entry.st_dev, entry.st_ino))

    def _verify_root(self) -> None:
        """Detect a root rename/swap between open() and this call."""

        try:
            entry = self.workspace.stat()
        except OSError as exc:
            raise FsToolError("workspace root is no longer accessible") from exc
        if (entry.st_dev, entry.st_ino) != self.root_id:
            raise PolicyViolationError("workspace root was replaced since it was opened")

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        max_lines: int = READ_MAX_LINES,
    ) -> ToolObservation:
        if start_line < 1:
            raise FsToolError("start_line must be >= 1")
        if not 1 <= max_lines <= READ_MAX_LINES:
            raise FsToolError(f"max_lines must be between 1 and {READ_MAX_LINES}")
        self._verify_root()
        resolved, relative = _resolve_within(self.workspace, path)
        _reject_sensitive_name(relative.rsplit("/", 1)[-1])
        entry = resolved.lstat()
        if not stat.S_ISREG(entry.st_mode):
            raise FsToolError("path is not a regular file")
        if entry.st_nlink > 1:
            raise PolicyViolationError("file has multiple hard links; refusing to read")
        if entry.st_size > FILE_MAX_BYTES:
            raise FsToolError("file exceeds the maximum readable size")
        data = resolved.read_bytes()
        # Fail closed on mid-read mutation.
        after = resolved.lstat()
        if (after.st_size, after.st_mtime_ns) != (entry.st_size, entry.st_mtime_ns):
            raise FsToolError("file changed while being read")
        raw_hash = hashlib.sha256(data).hexdigest()
        window_salt = f"{raw_hash}:{start_line}:{max_lines}"
        if _looks_binary(data):
            return ToolObservation(
                observation_id=_observation_id("read_file", relative, window_salt),
                tool="read_file",
                path=relative,
                content_hash=raw_hash,
                text="",
                incomplete=False,
                metadata={"binary": True, "size_bytes": entry.st_size},
            )
        decoded = _decode_strict(data)
        # Sanitize the WHOLE file before slicing so a window starting inside a
        # multi-line secret cannot leak the body, and redaction is line-preserving.
        sanitized_full, redacted = _sanitize_line_preserving(decoded)
        all_lines = sanitized_full.split("\n")
        total_lines = len(all_lines)
        # Clamp to end-of-file: a start past EOF yields an empty, non-citable window
        # rather than a manufactured phantom line.
        if start_line > total_lines:
            window: list[str] = []
        else:
            window = all_lines[start_line - 1 : start_line - 1 + max_lines]
        truncated = start_line > 1 or (start_line - 1 + max_lines) < total_lines
        text = "\n".join(window)
        if len(text.encode("utf-8")) > READ_MAX_BYTES or _estimate_tokens(text) > READ_MAX_TOKENS:
            text = _clip_to_byte_budget(text, READ_MAX_BYTES)
            window = text.split("\n")
            truncated = True
        numbered = tuple(
            f"{start_line + offset}: {line}" for offset, line in enumerate(window)
        )
        return ToolObservation(
            observation_id=_observation_id("read_file", relative, window_salt),
            tool="read_file",
            path=relative,
            content_hash=raw_hash,
            text=text,
            lines=numbered,
            start_line=start_line,
            truncated=truncated,
            incomplete=redacted,
            redacted=redacted,
            metadata={"total_lines": total_lines},
        )

    def list_files(self, path: str = ".", *, glob: str | None = None) -> ToolObservation:
        if glob is not None and len(glob) > GLOB_MAX_CHARS:
            raise FsToolError("glob pattern exceeds the length limit")
        self._verify_root()
        if path in ("", "."):
            base, relative = self.workspace, "."
        else:
            base, relative = _resolve_within(self.workspace, path)
            if not base.is_dir():
                raise FsToolError("list target is not a directory")
        # A recursive glob (containing ``**``) returns a bounded file tree of
        # matching workspace-relative paths, so a model can discover nested files
        # in one call instead of drilling directory by directory.
        if glob is not None and "**" in glob:
            return self._list_recursive(base, relative, glob)
        # Scan lazily and bound the number of entries examined so a directory
        # with millions of children cannot be fully materialized before the
        # LIST_MAX_ENTRIES cap applies.
        rows: list[tuple[str, bool]] = []
        visited = 0
        truncated_scan = False
        try:
            with os.scandir(base) as scan:
                for entry in scan:
                    visited += 1
                    if visited > WALK_VISIT_LIMIT or len(rows) >= LIST_MAX_ENTRIES * 4:
                        truncated_scan = True
                        break
                    name = entry.name
                    try:
                        _reject_component(name)
                    except FsToolError:
                        continue
                    try:
                        child_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if _is_reparse_point(child_stat):
                        continue
                    is_dir = stat.S_ISDIR(child_stat.st_mode)
                    if not is_dir:
                        # Sensitive files are denied by name in listings too, so a
                        # directory listing cannot enumerate secret-bearing paths.
                        try:
                            _reject_sensitive_name(name)
                        except FsToolError:
                            continue
                    if glob is not None and not is_dir and not _glob_match(glob, name):
                        continue
                    rows.append((name, is_dir))
        except OSError as exc:
            raise FsToolError("directory could not be listed") from exc
        rows.sort()
        entries = [
            f"{name}{'/' if is_dir else ''}"[:PATH_MAX_CHARS]
            for name, is_dir in rows[:LIST_MAX_ENTRIES]
        ]
        cap_hit = truncated_scan or len(rows) > LIST_MAX_ENTRIES
        entries, ceiling_hit = _apply_response_ceiling(entries)
        ceiling_hit = ceiling_hit or cap_hit
        text = "\n".join(entries)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return ToolObservation(
            observation_id=_observation_id("list_files", relative, content_hash),
            tool="list_files",
            path=relative,
            content_hash=content_hash,
            text=text,
            lines=tuple(entries),
            truncated=ceiling_hit,
            metadata={"entry_count": len(entries)},
        )

    def _list_recursive(self, base: Path, relative: str, glob: str) -> ToolObservation:
        """Return a bounded tree of workspace-relative files matching a ``**`` glob."""

        matches: list[str] = []
        walked, walk_truncated = self._walk_files(base)
        cap_hit = walk_truncated
        for file_path in walked:
            if len(matches) >= LIST_MAX_ENTRIES:
                cap_hit = True
                break
            rel_file = "/".join(file_path.relative_to(self.workspace).parts)
            if not _glob_match(glob, file_path.name, relative_path=rel_file):
                continue
            try:
                # Sensitive files are denied by name in recursive listings too.
                _reject_sensitive_name(file_path.name)
            except FsToolError:
                continue
            matches.append(rel_file[:PATH_MAX_CHARS])
        matches.sort()
        matches, ceiling_hit = _apply_response_ceiling(matches)
        text = "\n".join(matches)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return ToolObservation(
            observation_id=_observation_id("list_files", f"{relative}::{glob}", content_hash),
            tool="list_files",
            path=relative,
            content_hash=content_hash,
            text=text,
            lines=tuple(matches),
            truncated=cap_hit or ceiling_hit,
            metadata={"entry_count": len(matches), "recursive": True},
        )

    def search_text(self, query: str, *, glob: str | None = None) -> ToolObservation:
        if not query or len(query) > QUERY_MAX_CHARS:
            raise FsToolError("query is empty or exceeds the length limit")
        if glob is not None and len(glob) > GLOB_MAX_CHARS:
            raise FsToolError("glob pattern exceeds the length limit")
        self._verify_root()
        needle = query.lower()
        matches: list[str] = []
        files_scanned = 0
        bytes_scanned = 0
        redacted_any = False
        decode_refused = False
        walked, walk_truncated = self._walk_files()
        scan_truncated = False
        for file_path in walked:
            if files_scanned >= SEARCH_MAX_FILES or bytes_scanned >= SEARCH_MAX_SCAN_BYTES:
                scan_truncated = True
                break
            relative = "/".join(file_path.relative_to(self.workspace).parts)
            if glob is not None and not _glob_match(glob, file_path.name):
                continue
            try:
                _reject_sensitive_name(file_path.name)
            except FsToolError:
                continue
            entry = file_path.lstat()
            # Re-confirm the entry is a real regular file (not a reparse point a
            # concurrent writer swapped in) immediately before reading it.
            if _is_reparse_point(entry) or not stat.S_ISREG(entry.st_mode):
                continue
            if entry.st_size > FILE_MAX_BYTES:
                continue
            if entry.st_nlink > 1:
                # A hard link can point at an inode outside the workspace.
                continue
            if bytes_scanned + entry.st_size > SEARCH_MAX_SCAN_BYTES:
                scan_truncated = True
                break
            data = file_path.read_bytes()
            files_scanned += 1
            bytes_scanned += len(data)
            if _looks_binary(data):
                continue
            try:
                decoded = _decode_strict(data)
            except StrictDecodeError:
                # A non-binary file that fails strict UTF-8 is a decode refusal.
                decode_refused = True
                continue
            # Sanitize the WHOLE file before matching so a match inside a secret
            # body cannot leak it, and search only the sanitized representation.
            sanitized_full, redacted = _sanitize_line_preserving(decoded)
            if redacted:
                redacted_any = True
            for line_number, line in enumerate(sanitized_full.split("\n"), start=1):
                if needle in line.lower():
                    snippet = line.strip()[:SNIPPET_MAX_CHARS]
                    matches.append(f"{relative}:{line_number}: {snippet}")
                    if len(matches) >= SEARCH_MAX_MATCHES:
                        break
            if len(matches) >= SEARCH_MAX_MATCHES:
                break
        raw_match_count = len(matches)
        matches, ceiling_hit = _apply_response_ceiling(matches)
        text = "\n".join(matches)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return ToolObservation(
            observation_id=_observation_id("search_text", query, content_hash),
            tool="search_text",
            path=".",
            content_hash=content_hash,
            text=text,
            lines=tuple(matches),
            truncated=(
                raw_match_count >= SEARCH_MAX_MATCHES
                or ceiling_hit
                or walk_truncated
                or scan_truncated
            ),
            incomplete=redacted_any or decode_refused,
            redacted=redacted_any,
            metadata={
                "match_count": len(matches),
                "files_scanned": files_scanned,
                "decode_refused": decode_refused,
            },
        )

    def _walk_files(self, base: Path | None = None) -> tuple[list[Path], bool]:
        """Collect regular files under ``base`` (default: the workspace root).

        Returns the collected files and whether the ``WALK_VISIT_LIMIT`` work
        bound was reached (so callers can report truncation rather than imply
        completeness).

        Walking is scoped to ``base`` so listing a subdirectory does not traverse
        the entire workspace, bounded by ``WALK_VISIT_LIMIT`` visited entries, and
        each directory is re-confirmed non-reparse before it is opened. This
        shrinks - but does not fully close - the window in which a concurrent
        writer swaps a directory for a junction between the walk and a later read;
        fully handle-relative (no-follow) enumeration remains future work and is
        noted in docs/milestone-v0.2.md.
        """

        root = base if base is not None else self.workspace
        collected: list[Path] = []
        stack: list[Path] = [root]
        visited = 0
        while stack:
            directory = stack.pop()
            # Re-confirm the directory is inside the workspace and not a reparse
            # point right before opening it, so a swapped parent is refused here.
            try:
                if directory != self.workspace:
                    dir_stat = directory.lstat()
                    if _is_reparse_point(dir_stat) or not stat.S_ISDIR(dir_stat.st_mode):
                        continue
                # os.scandir yields lazily, so the visit cap bounds total work even
                # inside a single directory holding millions of entries - iterdir()
                # would have materialized the whole directory before any check.
                with os.scandir(directory) as scan:
                    for entry in scan:
                        visited += 1
                        if visited > WALK_VISIT_LIMIT:
                            return sorted(collected), True
                        name = entry.name
                        if name.lower() in _DENIED_DIR_NAMES:
                            continue
                        try:
                            child_stat = entry.stat(follow_symlinks=False)
                        except OSError:
                            continue
                        if _is_reparse_point(child_stat):
                            continue
                        child = directory / name
                        if stat.S_ISDIR(child_stat.st_mode):
                            stack.append(child)
                        elif stat.S_ISREG(child_stat.st_mode) and child_stat.st_nlink <= 1:
                            collected.append(child)
            except OSError:
                continue
        return sorted(collected), False


def _glob_match(pattern: str, name: str, *, relative_path: str | None = None) -> bool:
    """Match a filename (or relative path) against a restricted glob.

    ``list_files``/``search_text`` match a bare filename, but callers routinely
    pass recursive patterns like ``**/*.py`` or ``src/**/*.xml``. A leading
    ``**/`` is stripped so the tail applies to the filename, the original pattern
    is tried, and when a ``relative_path`` is supplied an interior ``**`` is
    matched against the full path via :func:`fnmatch` so directory-prefixed
    recursive globs also work.
    """

    if not pattern:
        return True
    candidates = {pattern}
    stripped = pattern
    while stripped.startswith("**/"):
        stripped = stripped[3:]
        candidates.add(stripped)
    pure = PurePosixPath(name)
    if any(pure.match(candidate) for candidate in candidates if candidate):
        return True
    if relative_path is not None and "**" in pattern:
        # Translate ``**`` (any depth) to fnmatch's ``*`` against the full path.
        translated = pattern.replace("**/", "*/").replace("**", "*")
        if fnmatch(relative_path, translated) or fnmatch(relative_path, pattern.replace("**", "*")):
            return True
    return False


def _apply_response_ceiling(lines: list[str]) -> tuple[list[str], bool]:
    """Bound the serialized UTF-8 size of a multi-line observation to a backstop."""

    kept: list[str] = []
    size = 0
    for line in lines:
        projected = size + len(line.encode("utf-8")) + 1
        if projected > RESPONSE_MAX_BYTES:
            return kept, True
        kept.append(line)
        size = projected
    return kept, False


def _clip_to_byte_budget(text: str, budget: int) -> str:
    """Clip text so its UTF-8 encoding stays within ``budget`` bytes."""

    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    return encoded[:budget].decode("utf-8", errors="ignore")
