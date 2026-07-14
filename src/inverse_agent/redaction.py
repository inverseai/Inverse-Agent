"""Redaction and egress helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from unicodedata import category, normalize

__all__ = [
    "SOURCE_INSTRUCTION_REDACTION_MARKER",
    "RedactionResult",
    "SecretSpan",
    "SourceInstructionResult",
    "neutralize_source_instructions",
    "private_key_spans",
    "redact_text",
    "secret_spans",
]

_PEM_TOKEN = re.compile(
    r"(?=(?P<token>-----(?P<kind>BEGIN|END) |-----))",
    re.IGNORECASE | re.ASCII,
)
_PRIVATE_KEY_LABEL = re.compile(r"PRIVATE KEY", re.IGNORECASE | re.ASCII)
_PEM_MARKER_CLOSE_LENGTH = 5

_SOURCE_INSTRUCTION_PATTERNS = (
    re.compile(
        r"(?i)\b(?:reviewer|assistant|system|model|prompt)\b.{0,160}"
        r"\b(?:ignore|disregard|override|return|output|respond|pass|finding|instruction)\b"
    ),
    re.compile(
        r"(?i)\b(?:ignore|disregard|override|return|output|respond)\b.{0,160}"
        r"\b(?:reviewer|assistant|system|model|prompt|finding|pass)\b"
    ),
)
_SOURCE_COMMENT_MARKERS = ("//", "#", "/*", "<!--", "-- ")
SOURCE_INSTRUCTION_REDACTION_MARKER = "[untrusted source instruction redacted]"

_SECRET_PATTERNS = (
    (
        "key-value-secret",
        re.compile(
            r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})"
        ),
    ),
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer-token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    (
        "github-token",
        re.compile(r"\b(?:gh[oprsu]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("provider-token", re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{20,}\b|\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    (
        "credential-url",
        re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^\s:/@]+:[^\s/@]+@"),
    ),
)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    blocked: bool
    matches: tuple[str, ...]


@dataclass(frozen=True)
class SecretSpan:
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class SourceInstructionResult:
    text: str
    redacted: bool
    incomplete: bool
    redacted_lines: tuple[int, ...] = ()


def _instruction_probe(value: str) -> str:
    normalized = normalize("NFKC", value)
    return "".join(character for character in normalized if category(character) != "Cf")


def _instruction_line_indexes(lines: list[str]) -> set[int]:
    """Find single-line and short split-line instruction payloads."""

    probes = [_instruction_probe(line) for line in lines]
    matched = {
        index
        for index, probe in enumerate(probes)
        if any(pattern.search(probe) for pattern in _SOURCE_INSTRUCTION_PATTERNS)
    }
    # Prompt injections commonly split the authority claim and directive across
    # adjacent comments. Match bounded two/three-line windows while retaining the
    # original records for line-preserving replacement.
    for width in (2, 3):
        for start in range(0, len(probes) - width + 1):
            indexes = range(start, start + width)
            if any(index in matched for index in indexes):
                continue
            window = " ".join(probes[start : start + width])
            if any(pattern.search(window) for pattern in _SOURCE_INSTRUCTION_PATTERNS):
                matched.update(indexes)
    return matched


def neutralize_source_instructions(
    value: str,
    *,
    source: bool = False,
    check: Callable[[], None] | None = None,
    track_redacted_lines: bool = False,
    redacted_line_window: tuple[int, int] | None = None,
) -> SourceInstructionResult:
    """Neutralize reviewer-directed source text without changing line boundaries.

    ``source=True`` preserves executable text before a recognized comment marker.
    A matching executable line with no comment boundary is omitted in full because
    retaining only selected tokens would silently reinterpret untrusted source.
    """

    if check is not None:
        check()
    records = value.splitlines(keepends=True)
    line_values = [
        record[:-2]
        if record.endswith("\r\n")
        else record[:-1]
        if record.endswith(("\r", "\n"))
        else record
        for record in records
    ]
    instruction_lines = _instruction_line_indexes(line_values)
    redacted = False
    incomplete = False
    redacted_lines: list[int] = []
    output: list[str] = []
    for line_number, (record, line) in enumerate(zip(records, line_values, strict=True), start=1):
        if check is not None and line_number % 128 == 0:
            check()
        if record.endswith("\r\n"):
            ending = "\r\n"
        elif record.endswith(("\r", "\n")):
            ending = record[-1:]
        else:
            ending = ""
        if line_number - 1 not in instruction_lines:
            output.append(record)
            continue
        redacted = True
        if track_redacted_lines and (
            redacted_line_window is None
            or redacted_line_window[0] <= line_number <= redacted_line_window[1]
        ):
            redacted_lines.append(line_number)
        diff_prefix = line[:1] if line[:1] in {"+", "-", " "} else ""
        content = line[len(diff_prefix) :]
        if not source:
            output.append(f"{diff_prefix}{SOURCE_INSTRUCTION_REDACTION_MARKER}{ending}")
            continue
        comment_positions = sorted(
            (position, marker)
            for marker in _SOURCE_COMMENT_MARKERS
            if (position := content.find(marker)) >= 0
        )
        if not comment_positions:
            output.append(f"{diff_prefix}[untrusted source instruction line omitted]{ending}")
            incomplete = True
            continue
        position, marker = comment_positions[0]
        code_prefix = content[:position]
        output.append(
            f"{diff_prefix}{code_prefix}{marker} {SOURCE_INSTRUCTION_REDACTION_MARKER}{ending}"
        )
        incomplete = incomplete or bool(code_prefix.strip())
    if check is not None:
        check()
    return SourceInstructionResult(
        text="".join(output),
        redacted=redacted,
        incomplete=incomplete,
        redacted_lines=tuple(redacted_lines),
    )


def private_key_spans(
    text: str, *, check: Callable[[], None] | None = None
) -> tuple[tuple[int, int, bool], ...]:
    """Return conservative PEM spans as ``(start, end, complete)`` tuples.

    A regex that stops at the first END marker is unsafe for nested or malformed
    input: it can hide an outer BEGIN from an unterminated-key fallback and leak
    the remaining key tail. This single linear marker scan tracks nesting and
    redacts through EOF whenever marker labels do not close in LIFO order.
    """

    if check is not None:
        check()
    spans: list[tuple[int, int, bool]] = []
    labels: list[str] = []
    span_start: int | None = None
    pending: tuple[bool, int, int] | None = None

    def first_line_break(start: int, end: int) -> int | None:
        carriage_return = text.find("\r", start, end)
        line_feed = text.find("\n", start, end)
        candidates = tuple(index for index in (carriage_return, line_feed) if index >= 0)
        return min(candidates) if candidates else None

    def consume_marker(
        is_begin: bool,
        marker_start: int,
        label: str,
        marker_end: int,
    ) -> bool:
        nonlocal span_start
        if _PRIVATE_KEY_LABEL.search(label) is None:
            return True
        if is_begin:
            if span_start is None:
                span_start = marker_start
            labels.append(label)
            return True
        if span_start is None:
            return True
        if not labels or labels[-1] != label:
            spans.append((span_start, len(text), False))
            return False
        labels.pop()
        if not labels:
            spans.append((span_start, marker_end, True))
            span_start = None
        return True

    for token_index, token in enumerate(_PEM_TOKEN.finditer(text), start=1):
        if check is not None and token_index % 128 == 0:
            check()
        token_start = token.start("token")
        token_end = token.end("token")
        if pending is not None:
            is_begin, marker_start, label_start = pending
            line_break = first_line_break(label_start, token_start)
            if line_break is None:
                if not consume_marker(
                    is_begin,
                    marker_start,
                    text[label_start:token_start],
                    token_start + _PEM_MARKER_CLOSE_LENGTH,
                ):
                    return tuple(spans)
            else:
                label = text[label_start:line_break]
                if _PRIVATE_KEY_LABEL.search(label) is not None:
                    if span_start is None and is_begin:
                        span_start = marker_start
                    if span_start is not None:
                        spans.append((span_start, len(text), False))
                        return tuple(spans)
            pending = None

        kind = token.group("kind")
        if kind is not None:
            pending = (kind.upper() == "BEGIN", token_start, token_end)

    if pending is not None:
        is_begin, marker_start, label_start = pending
        line_break = first_line_break(label_start, len(text))
        label_end = line_break if line_break is not None else len(text)
        label = text[label_start:label_end]
        if _PRIVATE_KEY_LABEL.search(label) is not None:
            if span_start is None and is_begin:
                span_start = marker_start
            if span_start is not None:
                spans.append((span_start, len(text), False))
                return tuple(spans)
    if span_start is not None:
        spans.append((span_start, len(text), False))
    if check is not None:
        check()
    return tuple(spans)


def secret_spans(text: str, *, check: Callable[[], None] | None = None) -> tuple[SecretSpan, ...]:
    """Return every secret span found against the original, unmodified text."""

    found: list[SecretSpan] = []
    for start, end, complete in private_key_spans(text, check=check):
        found.append(
            SecretSpan(
                kind="private-key-block" if complete else "private-key-prefix",
                start=start,
                end=end,
            )
        )
    for name, pattern in _SECRET_PATTERNS:
        if check is not None:
            check()
        for match_index, match in enumerate(pattern.finditer(text), start=1):
            found.append(SecretSpan(kind=name, start=match.start(), end=match.end()))
            if check is not None and match_index % 128 == 0:
                check()
    if check is not None:
        check()
    return tuple(found)


def redact_text(text: str) -> RedactionResult:
    found = secret_spans(text)
    if not found:
        return RedactionResult(text=text, blocked=False, matches=())

    counts: dict[str, int] = {}
    for secret in found:
        counts[secret.kind] = counts.get(secret.kind, 0) + 1
    matches = tuple(f"{name}:{count}" for name, count in counts.items())
    spans = sorted((secret.start, secret.end) for secret in found)

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append("[REDACTED_SECRET]")
        cursor = end
    pieces.append(text[cursor:])
    return RedactionResult(text="".join(pieces), blocked=True, matches=matches)
