"""Redaction and egress helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

# The PEM body span is bounded to a realistic key size so that many unterminated
# "BEGIN PRIVATE KEY" markers in attacker-controlled workspace content cannot
# drive the lazy scan to EOF for each marker (O(n^2) catastrophic backtracking /
# ReDoS on the untrusted read tier). A real key body is well under this bound;
# the linear private-key-prefix pattern still catches a genuinely unterminated
# block, so coverage is unchanged.
_PRIVATE_KEY_BODY_MAX = 8192

SECRET_PATTERNS = (
    (
        "private-key-block",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
            rf".{{0,{_PRIVATE_KEY_BODY_MAX}}}?"
            r"-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        "private-key-prefix",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*$", re.DOTALL),
    ),
    (
        "key-value-secret",
        re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})"),
    ),
    ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer-token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    ("github-token", re.compile(r"\b(?:gh[oprsu]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")),
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
