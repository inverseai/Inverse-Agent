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
import hmac
import re
import secrets
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

from inverse_agent.redaction import secret_spans
from inverse_agent.secure_fs import (
    SecureFsDeadlineError,
    SecureFsError,
    SecureFsPolicyError,
    SecureFsTooLargeError,
    SecureFsWorkspacePolicyError,
    SecureWorkspace,
)

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
FS_OPERATION_TIMEOUT_SECONDS = 10.0

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


class FsToolError(ValueError):
    """A read-tool request violated a path, size, or content policy."""


class PolicyViolationError(FsToolError):
    """A security-relevant refusal (confinement, links, sensitive files).

    Distinct from benign, retryable errors (bad line range, missing file): a
    policy violation terminates the investigation as an immediate refusal.
    """


class RequestValidationError(FsToolError):
    """A model request has an invalid shape and can be corrected without new evidence."""


class StrictDecodeError(FsToolError):
    """A file could not be strict-decoded as UTF-8 and was refused."""


@dataclass(frozen=True)
class ToolObservation:
    """A durable, model-visible result of one read-tool call.

    ``incomplete`` is set when content was redacted or otherwise could not be
    delivered faithfully. ``truncated`` records bounded omission; a read window
    can still ground a localized cited claim, while a truncated list/search
    cannot establish broad absence.
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


def _require_utf8(value: str, label: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RequestValidationError(f"{label} contains non-UTF-8 text") from exc


def _reject_component_policy(name: str) -> None:
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


def _reject_component(name: str) -> None:
    _reject_component_policy(name)
    _require_utf8(name, "path component")


def _reject_sensitive_name(name: str) -> None:
    for pattern in _DENIED_NAME_PATTERNS:
        if pattern.search(name):
            raise PolicyViolationError(f"file is denied by the sensitive-file policy: {name}")


def _relative_parts(raw_path: str, *, reject_sensitive_final: bool = False) -> tuple[str, ...]:
    if not raw_path:
        raise PolicyViolationError("path is empty")
    if "\x00" in raw_path:
        raise PolicyViolationError("path contains a null byte")
    normalized = raw_path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or (len(normalized) >= 2 and normalized[1] == ":"):
        raise PolicyViolationError("absolute paths are not permitted")
    parts = tuple(part for part in pure.parts if part not in ("", "."))
    for part in parts:
        _reject_component_policy(part)
    if reject_sensitive_final and parts:
        _reject_sensitive_name(parts[-1])
    if len(raw_path) > PATH_MAX_CHARS:
        raise RequestValidationError("path exceeds the length limit")
    _require_utf8(raw_path, "path")
    return parts


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


def _sanitize_line_preserving(
    text: str,
    *,
    deadline: float,
    redacted_line_window: tuple[int, int] | None = None,
) -> tuple[str, bool, tuple[int, ...]]:
    """Redact secrets over the FULL text while preserving the line count.

    Redacting line by line would defeat the multi-line private-key patterns
    (only the ``BEGIN`` header would match), leaking the key body. Instead the
    secret patterns run over the whole text and each replacement re-inserts the
    same number of newlines it covered, so downstream line numbers stay valid.
    """

    # Every ordinary pattern scans the original text, so an inner match can
    # never suppress a later enclosing match (for example a provider token used
    # as either the username or scheme of a credential-bearing URL). Private
    # keys use a linear marker-aware scan so nested or malformed blocks cannot
    # hide an outer BEGIN marker and leak the remaining key tail.
    def check_deadline() -> None:
        if time.monotonic() > deadline:
            raise FsToolError("source sanitization exceeded its deadline")

    spans = [(secret.start, secret.end) for secret in secret_spans(text, check=check_deadline)]
    if not spans:
        return text, False, ()

    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    newline_offsets = [index for index, character in enumerate(text) if character == "\n"]
    redacted_lines: set[int] = set()
    output_pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        if time.monotonic() > deadline:
            raise FsToolError("source sanitization exceeded its deadline")
        output_pieces.append(text[cursor:start])
        matched = text[start:end]
        output_pieces.append("[REDACTED_SECRET]" + "\n" * matched.count("\n"))
        if redacted_line_window is not None:
            first_line = bisect_left(newline_offsets, start) + 1
            last_character = max(start, end - 1)
            last_line = bisect_left(newline_offsets, last_character) + 1
            window_start, window_end = redacted_line_window
            redacted_lines.update(
                range(max(first_line, window_start), min(last_line, window_end) + 1)
            )
        cursor = end
    output_pieces.append(text[cursor:])
    return "".join(output_pieces), True, tuple(sorted(redacted_lines))


def _tool_error(exc: SecureFsError) -> FsToolError:
    if isinstance(exc, SecureFsPolicyError):
        return PolicyViolationError(str(exc))
    if isinstance(exc, SecureFsDeadlineError):
        return FsToolError(str(exc))
    return FsToolError(str(exc))


@dataclass(frozen=True)
class WorkspaceReader:
    """A source_read-scoped reader bound to one resolved workspace root."""

    workspace: Path
    secure: SecureWorkspace
    _identity_key: bytes = field(
        default_factory=lambda: secrets.token_bytes(32),
        repr=False,
        compare=False,
    )
    _identity_by_observation: dict[str, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    @classmethod
    def open(cls, workspace: Path) -> WorkspaceReader:
        try:
            secure = SecureWorkspace.open(workspace)
        except SecureFsError as exc:
            raise _tool_error(exc) from exc
        return cls(secure.workspace, secure)

    def evidence_identity(self, observation_id: str) -> str | None:
        """Return a run-local opaque file identity that is never model-visible."""

        return self._identity_by_observation.get(observation_id)

    def _remember_identity(self, observation_id: str, identity: tuple[int, int]) -> None:
        raw = f"{self.secure.root_identity[0]}:{self.secure.root_identity[1]}:{identity[0]}:{identity[1]}"
        digest = hmac.new(self._identity_key, raw.encode("ascii"), hashlib.sha256).hexdigest()
        self._identity_by_observation[observation_id] = digest

    @staticmethod
    def _deadline() -> float:
        return time.monotonic() + FS_OPERATION_TIMEOUT_SECONDS

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        max_lines: int = READ_MAX_LINES,
    ) -> ToolObservation:
        # Security-sensitive path policy takes precedence over correctable
        # request-shape errors when several arguments are invalid together.
        parts = _relative_parts(path, reject_sensitive_final=True)
        if not parts:
            raise PolicyViolationError("path must reference a file inside the workspace")
        relative = "/".join(parts)
        if start_line < 1:
            raise RequestValidationError("start_line must be >= 1")
        if not 1 <= max_lines <= READ_MAX_LINES:
            raise RequestValidationError(f"max_lines must be between 1 and {READ_MAX_LINES}")
        deadline = self._deadline()
        try:
            secure_read = self.secure.read_bytes(
                parts,
                maximum_bytes=FILE_MAX_BYTES,
                deadline=deadline,
            )
        except SecureFsError as exc:
            raise _tool_error(exc) from exc
        data = secure_read.data
        entry = secure_read.entry
        if _looks_binary(data):
            empty_hash = hashlib.sha256(b"").hexdigest()
            observation = ToolObservation(
                observation_id=_observation_id(
                    "read_file", relative, f"binary:{entry.size}:{start_line}:{max_lines}"
                ),
                tool="read_file",
                path=relative,
                content_hash=empty_hash,
                text="",
                incomplete=False,
                metadata={
                    "binary": True,
                    "size_bytes": entry.size,
                },
            )
            self._remember_identity(observation.observation_id, entry.identity)
            return observation
        decoded = _decode_strict(data)
        # Sanitize the WHOLE file before slicing so a window starting inside a
        # multi-line secret cannot leak the body, and redaction is line-preserving.
        sanitized_full, redacted, redacted_lines = _sanitize_line_preserving(
            decoded,
            deadline=deadline,
            redacted_line_window=(start_line, start_line + max_lines - 1),
        )
        sanitized_hash = hashlib.sha256(sanitized_full.encode("utf-8")).hexdigest()
        window_salt = f"{sanitized_hash}:{start_line}:{max_lines}"
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
        numbered = tuple(f"{start_line + offset}: {line}" for offset, line in enumerate(window))
        observation = ToolObservation(
            observation_id=_observation_id("read_file", relative, window_salt),
            tool="read_file",
            path=relative,
            content_hash=sanitized_hash,
            text=text,
            lines=numbered,
            start_line=start_line,
            truncated=truncated,
            incomplete=redacted,
            redacted=redacted,
            metadata={
                "total_lines": total_lines,
                "redacted_lines": redacted_lines,
            },
        )
        self._remember_identity(observation.observation_id, entry.identity)
        return observation

    def list_files(self, path: str = ".", *, glob: str | None = None) -> ToolObservation:
        # Resolve and enforce path policy before validating the optional glob,
        # so a malformed glob cannot hide a simultaneous traversal attempt.
        if path in ("", "."):
            base_parts: tuple[str, ...] = ()
            relative = "."
        else:
            base_parts = _relative_parts(path)
            if not base_parts:
                raise PolicyViolationError("list path must reference a workspace directory")
            relative = "/".join(base_parts)
        if glob is not None and len(glob) > GLOB_MAX_CHARS:
            raise RequestValidationError("glob pattern exceeds the length limit")
        if glob is not None:
            _require_utf8(glob, "glob")
        deadline = self._deadline()
        # A recursive glob (containing ``**``) returns a bounded file tree of
        # matching workspace-relative paths, so a model can discover nested files
        # in one call instead of drilling directory by directory.
        if glob is not None and "**" in glob:
            return self._list_recursive(base_parts, relative, glob, deadline=deadline)
        # Scan lazily and bound the number of entries examined so a directory
        # with millions of children cannot be fully materialized before the
        # LIST_MAX_ENTRIES cap applies.
        rows: list[tuple[str, bool]] = []
        visited = 0
        truncated_scan = False
        try:
            listing = self.secure.list_directory(
                base_parts,
                maximum_visits=min(WALK_VISIT_LIMIT, LIST_MAX_ENTRIES * 4),
                deadline=deadline,
            )
        except SecureFsError as exc:
            raise _tool_error(exc) from exc
        filtered_count = listing.filtered
        truncated_scan = listing.truncated or listing.refused > 0 or filtered_count > 0
        for entry in listing.entries:
            visited += 1
            if visited > WALK_VISIT_LIMIT or len(rows) >= LIST_MAX_ENTRIES * 4:
                truncated_scan = True
                break
            name = entry.name
            try:
                _reject_component(name)
            except FsToolError:
                filtered_count += 1
                continue
            if len(name) + (1 if entry.is_dir else 0) > PATH_MAX_CHARS:
                filtered_count += 1
                continue
            is_dir = entry.is_dir
            if not is_dir:
                if entry.link_count > 1:
                    filtered_count += 1
                    continue
                try:
                    _reject_sensitive_name(name)
                except FsToolError:
                    filtered_count += 1
                    continue
            if glob is not None and not is_dir and not _glob_match(glob, name):
                continue
            rows.append((name, is_dir))
        rows.sort()
        rendered_entries = [
            f"{name}{'/' if is_dir else ''}" for name, is_dir in rows[:LIST_MAX_ENTRIES]
        ]
        cap_hit = truncated_scan or filtered_count > 0 or len(rows) > LIST_MAX_ENTRIES
        rendered_entries, ceiling_hit = _apply_response_ceiling(rendered_entries)
        ceiling_hit = ceiling_hit or cap_hit
        text = "\n".join(rendered_entries)
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        return ToolObservation(
            observation_id=_observation_id("list_files", relative, content_hash),
            tool="list_files",
            path=relative,
            content_hash=content_hash,
            text=text,
            lines=tuple(rendered_entries),
            truncated=ceiling_hit,
            incomplete=listing.refused > 0 or filtered_count > 0,
            metadata={
                "entry_count": len(rendered_entries),
                "refused_entry_count": listing.refused,
                "filtered_entry_count": filtered_count,
                "glob": glob,
            },
        )

    def _list_recursive(
        self,
        base_parts: tuple[str, ...],
        relative: str,
        glob: str,
        *,
        deadline: float,
    ) -> ToolObservation:
        """Return a bounded tree of workspace-relative files matching a ``**`` glob."""

        matches: list[str] = []
        walked, walk_truncated, walk_incomplete, omitted_count = self._walk_files(
            base_parts, deadline=deadline
        )
        cap_hit = walk_truncated or walk_incomplete
        for relative_file in walked:
            if len(matches) >= LIST_MAX_ENTRIES:
                cap_hit = True
                break
            name = relative_file.rsplit("/", 1)[-1]
            if not _glob_match(glob, name, relative_path=relative_file):
                continue
            try:
                # Sensitive files are denied by name in recursive listings too.
                _reject_sensitive_name(name)
            except FsToolError:
                omitted_count += 1
                walk_incomplete = True
                cap_hit = True
                continue
            if len(relative_file) > PATH_MAX_CHARS:
                omitted_count += 1
                walk_incomplete = True
                cap_hit = True
                continue
            matches.append(relative_file)
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
            incomplete=walk_incomplete,
            metadata={
                "entry_count": len(matches),
                "recursive": True,
                "glob": glob,
                "omitted_entry_count": omitted_count,
            },
        )

    def search_text(self, query: str, *, glob: str | None = None) -> ToolObservation:
        if not query or len(query) > QUERY_MAX_CHARS:
            raise RequestValidationError("query is empty or exceeds the length limit")
        _require_utf8(query, "query")
        if glob is not None and len(glob) > GLOB_MAX_CHARS:
            raise RequestValidationError("glob pattern exceeds the length limit")
        if glob is not None:
            _require_utf8(glob, "glob")
        deadline = self._deadline()
        needle = query.lower()
        matches: list[str] = []
        files_scanned = 0
        bytes_scanned = 0
        redacted_any = False
        decode_refused = False
        read_refused = False
        oversized_skipped = 0
        binary_skipped = 0
        policy_race_refused = 0
        sensitive_skipped = 0
        walked, walk_truncated, walk_incomplete, omitted_count = self._walk_files(
            (), deadline=deadline
        )
        scan_truncated = False
        for relative in walked:
            if files_scanned >= SEARCH_MAX_FILES or bytes_scanned >= SEARCH_MAX_SCAN_BYTES:
                scan_truncated = True
                break
            name = relative.rsplit("/", 1)[-1]
            if glob is not None and not _glob_match(glob, name):
                continue
            try:
                _reject_sensitive_name(name)
            except FsToolError:
                sensitive_skipped += 1
                continue
            try:
                secure_read = self.secure.read_bytes(
                    tuple(relative.split("/")),
                    maximum_bytes=FILE_MAX_BYTES,
                    deadline=deadline,
                )
            except SecureFsDeadlineError as exc:
                raise _tool_error(exc) from exc
            except SecureFsWorkspacePolicyError as exc:
                raise _tool_error(exc) from exc
            except SecureFsPolicyError:
                # The path came from a handle-validated walk. A later link or
                # hard-link refusal is therefore a workspace mutation, not a
                # denied path requested by the model. Preserve the refusal as
                # incomplete search evidence instead of misclassifying the run.
                read_refused = True
                policy_race_refused += 1
                continue
            except SecureFsTooLargeError:
                oversized_skipped += 1
                continue
            except SecureFsError:
                read_refused = True
                continue
            if bytes_scanned + secure_read.entry.size > SEARCH_MAX_SCAN_BYTES:
                scan_truncated = True
                break
            data = secure_read.data
            files_scanned += 1
            bytes_scanned += len(data)
            if _looks_binary(data):
                binary_skipped += 1
                continue
            try:
                decoded = _decode_strict(data)
            except StrictDecodeError:
                # A non-binary file that fails strict UTF-8 is a decode refusal.
                decode_refused = True
                continue
            # Sanitize the WHOLE file before matching so a match inside a secret
            # body cannot leak it, and search only the sanitized representation.
            sanitized_full, redacted, _redacted_lines = _sanitize_line_preserving(
                decoded, deadline=deadline
            )
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
                or walk_incomplete
                or scan_truncated
                or read_refused
                or oversized_skipped > 0
                or binary_skipped > 0
                or sensitive_skipped > 0
            ),
            incomplete=(
                redacted_any
                or decode_refused
                or walk_incomplete
                or read_refused
                or oversized_skipped > 0
                or binary_skipped > 0
                or sensitive_skipped > 0
            ),
            redacted=redacted_any,
            metadata={
                "match_count": len(matches),
                "files_scanned": files_scanned,
                "decode_refused": decode_refused,
                "read_refused": read_refused,
                "oversized_skipped": oversized_skipped,
                "binary_skipped": binary_skipped,
                "policy_race_refused": policy_race_refused,
                "sensitive_skipped": sensitive_skipped,
                "walk_omitted_entry_count": omitted_count,
                "query": query,
                "glob": glob,
            },
        )

    def _walk_files(
        self,
        base_parts: tuple[str, ...],
        *,
        deadline: float,
    ) -> tuple[list[str], bool, bool, int]:
        """Collect regular files under a workspace-relative directory.

        Returns the collected files and whether the ``WALK_VISIT_LIMIT`` work
        bound was reached (so callers can report truncation rather than imply
        completeness).

        Every directory enumeration is anchored at the retained workspace root
        and every child is opened relative to its retained parent handle.
        """

        collected: list[str] = []
        stack: list[tuple[str, ...]] = [base_parts]
        visited = 0
        incomplete = False
        omitted_count = 0
        while stack:
            directory_parts = stack.pop()
            remaining = WALK_VISIT_LIMIT - visited
            if remaining <= 0:
                return sorted(collected), True, incomplete, omitted_count
            try:
                listing = self.secure.list_directory(
                    directory_parts,
                    maximum_visits=remaining,
                    deadline=deadline,
                )
            except SecureFsDeadlineError as exc:
                raise _tool_error(exc) from exc
            except SecureFsError as exc:
                raise _tool_error(exc) from exc
            visited += listing.visited
            directory_omissions = listing.refused + listing.filtered
            omitted_count += directory_omissions
            incomplete = incomplete or directory_omissions > 0
            for entry in listing.entries:
                name = entry.name
                if name.lower() in _DENIED_DIR_NAMES:
                    omitted_count += 1
                    incomplete = True
                    continue
                try:
                    _reject_component(name)
                except FsToolError:
                    omitted_count += 1
                    incomplete = True
                    continue
                child_parts = (*directory_parts, name)
                if entry.is_dir:
                    stack.append(child_parts)
                elif entry.is_file and entry.link_count <= 1:
                    try:
                        _reject_sensitive_name(name)
                    except FsToolError:
                        omitted_count += 1
                        incomplete = True
                        continue
                    relative_file = "/".join(child_parts)
                    if len(relative_file) > PATH_MAX_CHARS:
                        omitted_count += 1
                        incomplete = True
                        continue
                    collected.append(relative_file)
                elif entry.is_file:
                    omitted_count += 1
                    incomplete = True
            if listing.truncated:
                return sorted(collected), True, incomplete, omitted_count
        return sorted(collected), False, incomplete, omitted_count


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
