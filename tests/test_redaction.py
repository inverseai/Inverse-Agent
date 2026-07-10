import pytest

from inverse_agent.redaction import redact_text


@pytest.mark.parametrize(
    "secret",
    [
        "api_key=sk_test_secret_value",
        "AKIA1234567890ABCDEF",
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "-".join(("xoxb", "1234567890", "abcdefghijklmnopqrstuvwxyz")),
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "hf_abcdefghijklmnopqrstuvwxyz123456",
        "eyJabcdefghijk.abcdefghijklmnop.abcdefghijklmnop",
        "postgresql://alice:supersecret@example.test/db",
    ],
)
def test_redact_text_removes_known_secret_shapes(secret: str) -> None:
    result = redact_text(secret)
    assert result.blocked
    assert secret not in result.text
    assert "[REDACTED_SECRET]" in result.text


def test_redact_text_removes_complete_private_key_block() -> None:
    body = "SENSITIVE_PRIVATE_KEY_BODY_123456789"
    value = f"-----BEGIN PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----"
    result = redact_text(value)
    assert result.blocked
    assert body not in result.text
    assert "BEGIN PRIVATE KEY" not in result.text
    assert "END PRIVATE KEY" not in result.text


def test_redact_text_removes_truncated_private_key_block() -> None:
    value = "-----BEGIN PRIVATE KEY-----\nSENSITIVE_PARTIAL_BODY"
    result = redact_text(value)
    assert result.blocked
    assert "SENSITIVE_PARTIAL_BODY" not in result.text


def test_redact_text_allows_non_secret_text() -> None:
    result = redact_text("ordinary build output")
    assert not result.blocked
    assert result.matches == ()
    assert result.text == "ordinary build output"
