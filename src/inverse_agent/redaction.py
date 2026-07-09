"""Redaction and egress helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

SECRET_PATTERNS = (
    (
        "key-value-secret",
        re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})"),
    ),
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("private-key-header", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)


@dataclass(frozen=True)
class RedactionResult:
    text: str
    blocked: bool
    matches: tuple[str, ...]


def redact_text(text: str) -> RedactionResult:
    matches: list[str] = []
    redacted = text
    for name, pattern in SECRET_PATTERNS:
        count = 0
        for _match in pattern.finditer(redacted):
            count += 1
        if count:
            matches.append(f"{name}:{count}")
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return RedactionResult(text=redacted, blocked=bool(matches), matches=tuple(matches))
