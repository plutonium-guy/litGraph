"""PiiScrubber — detects emails, phones, SSNs, Luhn-validated credit
cards, AWS access keys, JWTs, IPv4/v6. Mandatory hygiene layer for
any log pipeline that ships to third parties."""
from litgraph.evaluators import PiiScrubber, luhn_valid


def test_email_redacted():
    s = PiiScrubber()
    r = s.scrub("Contact alice@example.com for details.")
    assert "<EMAIL>" in r["scrubbed"]
    assert "alice@example.com" not in r["scrubbed"]
    assert r["replacements"][0]["kind"] == "EMAIL"


def test_ssn_redacted():
    s = PiiScrubber()
    r = s.scrub("My SSN is 123-45-6789.")
    assert "<SSN>" in r["scrubbed"]
    assert "123-45-6789" not in r["scrubbed"]


def test_aws_access_key_redacted():
    s = PiiScrubber()
    r = s.scrub("aws_access_key_id=AKIAIOSFODNN7EXAMPLE")
    assert "<AWS_ACCESS_KEY>" in r["scrubbed"]


def test_jwt_redacted():
    s = PiiScrubber()
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJzdWIiOiIxMjM0NSJ9."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    r = s.scrub(f"Bearer {jwt}")
    assert "<JWT>" in r["scrubbed"]


def test_credit_card_requires_luhn_pass():
    s = PiiScrubber()
    # Real Visa test number.
    r1 = s.scrub("CC: 4111 1111 1111 1111")
    assert "<CREDIT_CARD>" in r1["scrubbed"]
    # 16 digits that fail Luhn must NOT be redacted.
    r2 = s.scrub("Customer ID: 1234 5678 9012 3456")
    assert "<CREDIT_CARD>" not in r2["scrubbed"]


def test_luhn_validation_can_be_disabled():
    s = PiiScrubber(validate_luhn=False)
    r = s.scrub("1234 5678 9012 3456")
    assert "<CREDIT_CARD>" in r["scrubbed"]


def test_ipv4_redacted():
    s = PiiScrubber()
    r = s.scrub("Client IP: 192.168.1.1 connected")
    assert "<IPV4>" in r["scrubbed"]


def test_phone_redacted_us_format():
    s = PiiScrubber()
    r = s.scrub("Call +1-555-867-5309 for support.")
    assert "<PHONE>" in r["scrubbed"]


def test_multiple_pii_all_redacted():
    s = PiiScrubber()
    r = s.scrub(
        "alice@example.com +1-555-867-5309 AKIAIOSFODNN7EXAMPLE"
    )
    kinds = {rep["kind"] for rep in r["replacements"]}
    assert "EMAIL" in kinds
    assert "PHONE" in kinds
    assert "AWS_ACCESS_KEY" in kinds
    # Original values never appear in scrubbed output.
    assert "alice@example.com" not in r["scrubbed"]
    assert "AKIAIOSFODNN7EXAMPLE" not in r["scrubbed"]


def test_replacement_carries_original_and_offset():
    s = PiiScrubber()
    text = "Email: alice@example.com is done"
    r = s.scrub(text)
    rep = r["replacements"][0]
    assert rep["original"] == "alice@example.com"
    assert rep["start"] == 7
    # Caller can recover the original from the source text.
    assert text[rep["start"]:rep["start"] + len(rep["original"])] == rep["original"]


def test_no_pii_leaves_text_unchanged():
    s = PiiScrubber()
    r = s.scrub("Nothing sensitive here.")
    assert r["scrubbed"] == "Nothing sensitive here."
    assert r["replacements"] == []


def test_custom_patterns_append_to_defaults():
    s = PiiScrubber(
        extra_patterns=[("ORDER_ID", r"\bORD-\d{6}\b")],
    )
    r = s.scrub("Order ORD-123456 from alice@example.com")
    assert "<ORDER_ID>" in r["scrubbed"]
    assert "<EMAIL>" in r["scrubbed"]


def test_only_custom_drops_defaults():
    s = PiiScrubber(
        extra_patterns=[("INTERNAL", r"\bIDX-\d+\b")],
        only_custom=True,
    )
    r = s.scrub("alice@example.com referenced IDX-99")
    # Email not scrubbed because defaults were dropped.
    assert "alice@example.com" in r["scrubbed"]
    assert "<INTERNAL>" in r["scrubbed"]


def test_invalid_regex_in_extra_patterns_raises_value_error():
    try:
        PiiScrubber(extra_patterns=[("BAD", r"(unclosed")])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "invalid regex" in str(e).lower()


def test_standalone_luhn_helper():
    assert luhn_valid("4111 1111 1111 1111") is True
    assert luhn_valid("4111-1111-1111-1111") is True
    assert luhn_valid("4111111111111112") is False  # off by one
    assert luhn_valid("411") is False  # too short


if __name__ == "__main__":
    import traceback
    fns = [
        test_email_redacted,
        test_ssn_redacted,
        test_aws_access_key_redacted,
        test_jwt_redacted,
        test_credit_card_requires_luhn_pass,
        test_luhn_validation_can_be_disabled,
        test_ipv4_redacted,
        test_phone_redacted_us_format,
        test_multiple_pii_all_redacted,
        test_replacement_carries_original_and_offset,
        test_no_pii_leaves_text_unchanged,
        test_custom_patterns_append_to_defaults,
        test_only_custom_drops_defaults,
        test_invalid_regex_in_extra_patterns_raises_value_error,
        test_standalone_luhn_helper,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)
