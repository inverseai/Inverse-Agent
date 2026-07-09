from inverse_agent.redaction import redact_text


def test_redact_text_blocks_secret_like_values() -> None:
    result = redact_text("api_key=sk_test_secret_value")

    assert result.blocked
    assert "sk_test_secret_value" not in result.text
    assert "[REDACTED_SECRET]" in result.text
    assert result.matches == ("key-value-secret:1",)


def test_redact_text_blocks_other_known_secret_shapes() -> None:
    aws = redact_text("AKIA1234567890ABCDEF")
    private_key = redact_text("-----BEGIN RSA PRIVATE KEY-----")

    assert aws.matches == ("aws-access-key:1",)
    assert private_key.matches == ("private-key-header:1",)


def test_redact_text_allows_non_secret_text() -> None:
    result = redact_text("ordinary build output")

    assert not result.blocked
    assert result.matches == ()
    assert result.text == "ordinary build output"
