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


def test_redact_text_removes_nested_private_key_tail() -> None:
    value = (
        "-----BEGIN PRIVATE KEY-----\n"
        "OUTER_HEAD\n"
        "-----BEGIN PRIVATE KEY-----\n"
        "INNER_BODY\n"
        "-----END PRIVATE KEY-----\n"
        "OUTER_TAIL\n"
        "-----END PRIVATE KEY-----"
    )
    result = redact_text(value)
    assert result.blocked
    assert result.text == "[REDACTED_SECRET]"
    assert "OUTER_TAIL" not in result.text


@pytest.mark.parametrize(
    "label",
    [
        "PGP PRIVATE KEY BLOCK",
        "ED25519 PRIVATE KEY",
        "X" * 53 + " PRIVATE KEY",
    ],
)
def test_redact_text_removes_private_key_armor_label_variants(label: str) -> None:
    body = "SENSITIVE_PRIVATE_KEY_BODY"
    value = f"-----BEGIN {label}-----\n{body}\n-----END {label}-----"
    result = redact_text(value)
    assert result.blocked
    assert result.text == "[REDACTED_SECRET]"
    assert body not in result.text


def test_redact_text_keeps_offsets_after_unicode_case_expansion() -> None:
    body = "SENSITIVE_PRIVATE_KEY_BODY"
    value = f"Unicode prefix: ß -----BEGIN PRIVATE KEY-----\n{body}\n-----END PRIVATE KEY-----"
    result = redact_text(value)
    assert result.blocked
    assert body not in result.text
    assert result.text.startswith("Unicode prefix: ß ")


@pytest.mark.parametrize("dash_count", range(6, 10))
@pytest.mark.parametrize("shifted_marker", ["begin", "end"])
def test_redact_text_finds_overlapping_private_key_markers(
    dash_count: int, shifted_marker: str
) -> None:
    body = "SENSITIVE_PRIVATE_KEY_BODY"
    dashes = "-" * dash_count
    begin_dashes = dashes if shifted_marker == "begin" else "-----"
    end_dashes = dashes if shifted_marker == "end" else "-----"
    value = (
        f"{begin_dashes}BEGIN PRIVATE KEY-----\n"
        f"{body}\n"
        f"{end_dashes}END PRIVATE KEY-----\n"
        "safe_after = 1"
    )
    result = redact_text(value)
    assert result.blocked
    assert body not in result.text
    assert "PRIVATE KEY" not in result.text
    assert result.text.endswith("safe_after = 1")


def test_redact_text_allows_non_secret_text() -> None:
    result = redact_text("ordinary build output")
    assert not result.blocked
    assert result.matches == ()
    assert result.text == "ordinary build output"


def test_private_key_block_is_not_catastrophic() -> None:
    # A workspace file full of unterminated BEGIN markers must not drive the
    # marker scanner into superlinear work (ReDoS on the read tier).
    import time

    payload = "-----BEGIN A PRIVATE KEY-----\n" * 40000  # ~1.2 MB
    started = time.perf_counter()
    result = redact_text(payload)
    elapsed = time.perf_counter() - started
    assert result.blocked  # the marker scanner catches unterminated keys
    assert elapsed < 20.0, f"redaction took {elapsed:.1f}s on pathological input"


def test_private_key_marker_scan_is_linear_on_one_long_line() -> None:
    import time

    payload = "-----BEGIN PRIVATE KEY " * 38000 + "-----END PRIVATE KEY-----"
    started = time.perf_counter()
    result = redact_text(payload)
    elapsed = time.perf_counter() - started
    assert result.blocked
    assert elapsed < 2.5, f"redaction took {elapsed:.1f}s on same-line markers"


def test_private_key_block_still_redacts_real_key() -> None:
    body = "MIIEvQ" + "A" * 400
    value = f"x\n-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----\ny"
    result = redact_text(value)
    assert result.blocked
    assert body not in result.text
    assert "BEGIN RSA PRIVATE KEY" not in result.text
