"""Redaction and egress helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["RedactionResult", "SecretSpan", "private_key_spans", "redact_text", "secret_spans"]

_PEM_TOKEN = re.compile(
    r"(?=(?P<token>-----(?P<kind>BEGIN|END) |-----))",
    re.IGNORECASE | re.ASCII,
)
_PRIVATE_KEY_LABEL = re.compile(r"PRIVATE KEY", re.IGNORECASE | re.ASCII)
_PEM_MARKER_CLOSE_LENGTH = 5

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
