"""Signed, short-lived approval capabilities bound to one exact action."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from secrets import token_hex
from typing import Protocol

from inverse_agent.models import CommandRule, Domain


class ApprovalError(ValueError):
    """Raised when an approval capability is missing, invalid, or expired."""


@dataclass(frozen=True)
class ApprovalClaims:
    approval_id: str
    challenge_id: str
    action_digest: str
    rule: str
    workspace: str
    domain: str
    approved_by: str
    issued_at: int
    expires_at: int


class ApprovalReplayStore(Protocol):
    def consume(self, approval_id: str) -> bool:
        """Return true only for the first successful consumption."""


class MemoryApprovalReplayStore:
    def __init__(self) -> None:
        self._consumed: set[str] = set()
        self._lock = threading.Lock()

    def consume(self, approval_id: str) -> bool:
        with self._lock:
            if approval_id in self._consumed:
                return False
            self._consumed.add(approval_id)
            return True


class SqliteApprovalReplayStore:
    """Process-safe replay protection that survives runner restarts."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS consumed_approvals "
                "(approval_id TEXT PRIMARY KEY, consumed_at INTEGER NOT NULL)"
            )

    def consume(self, approval_id: str) -> bool:
        with sqlite3.connect(self.path, timeout=30) as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO consumed_approvals(approval_id, consumed_at) VALUES (?, ?)",
                (approval_id, int(time.time())),
            )
            return cursor.rowcount == 1


def action_digest(
    *,
    workspace: Path,
    domain: Domain,
    rule: CommandRule,
    argv: tuple[str, ...],
) -> str:
    payload = {
        "argv": list(argv),
        "domain": domain.value,
        "rule": rule.name,
        "workspace": str(workspace.resolve()),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ApprovalAuthority:
    """Issues HMAC-authenticated capabilities that cannot be reused for another action."""

    def __init__(self, secret: bytes, replay_store: ApprovalReplayStore | None = None):
        if len(secret) < 32:
            raise ValueError("approval secret must contain at least 32 bytes")
        self._secret = secret
        self._replay_store = replay_store or MemoryApprovalReplayStore()

    def issue(
        self,
        *,
        workspace: Path,
        domain: Domain,
        rule: CommandRule,
        argv: tuple[str, ...],
        approved_by: str,
        challenge_id: str,
        ttl_seconds: int = 300,
        now: int | None = None,
        expires_at: int | None = None,
    ) -> tuple[str, ApprovalClaims]:
        if ttl_seconds <= 0 or ttl_seconds > 3600:
            raise ValueError("approval ttl must be between 1 and 3600 seconds")
        if len(challenge_id) != 32 or any(
            character not in "0123456789abcdef" for character in challenge_id
        ):
            raise ValueError("approval challenge_id must be 32 lowercase hexadecimal characters")
        issued_at = int(time.time() if now is None else now)
        resolved_expires_at = issued_at + ttl_seconds if expires_at is None else expires_at
        if resolved_expires_at <= issued_at or resolved_expires_at > issued_at + 3600:
            raise ValueError("approval expiry must be after issuance and at most 3600 seconds")
        claims = ApprovalClaims(
            approval_id=token_hex(16),
            challenge_id=challenge_id,
            action_digest=action_digest(
                workspace=workspace,
                domain=domain,
                rule=rule,
                argv=argv,
            ),
            rule=rule.name,
            workspace=str(workspace.resolve()),
            domain=domain.value,
            approved_by=approved_by,
            issued_at=issued_at,
            expires_at=resolved_expires_at,
        )
        payload = json.dumps(asdict(claims), sort_keys=True, separators=(",", ":")).encode()
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        token = f"{_encode(payload)}.{_encode(signature)}"
        return token, claims

    def verify(
        self,
        token: str,
        *,
        workspace: Path,
        domain: Domain,
        rule: CommandRule,
        argv: tuple[str, ...],
        expected_challenge_id: str,
        now: int | None = None,
        consume: bool = True,
    ) -> ApprovalClaims:
        try:
            payload_part, signature_part = token.split(".", 1)
            payload = _decode(payload_part)
            signature = _decode(signature_part)
            raw_claims = json.loads(payload)
            claims = ApprovalClaims(**raw_claims)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ApprovalError("malformed approval capability") from exc

        expected_signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected_signature):
            raise ApprovalError("invalid approval capability signature")
        current_time = int(time.time() if now is None else now)
        if claims.expires_at <= current_time:
            raise ApprovalError("approval capability expired")
        if len(claims.challenge_id) != 32 or any(
            character not in "0123456789abcdef" for character in claims.challenge_id
        ):
            raise ApprovalError("approval capability has an invalid challenge identity")
        if len(expected_challenge_id) != 32 or any(
            character not in "0123456789abcdef" for character in expected_challenge_id
        ):
            raise ApprovalError("approval request has an invalid challenge identity")
        if not hmac.compare_digest(claims.challenge_id, expected_challenge_id):
            raise ApprovalError("approval capability does not match the current challenge")
        expected_digest = action_digest(
            workspace=workspace,
            domain=domain,
            rule=rule,
            argv=argv,
        )
        if not hmac.compare_digest(claims.action_digest, expected_digest):
            raise ApprovalError("approval capability does not match this action")
        if consume and not self._replay_store.consume(claims.approval_id):
            raise ApprovalError("approval capability already consumed")
        return claims


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
