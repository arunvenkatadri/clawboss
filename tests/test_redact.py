"""Tests for clawboss.redact — PII detection and redaction."""

from clawboss.redact import Redactor


class TestRedactorEmail:
    def test_redacts_email(self):
        r = Redactor(["email"])
        result = r.redact("Contact bob@example.com for info")
        assert "[EMAIL]" in result.text
        assert "bob@example.com" not in result.text
        assert result.redacted_count == 1
        assert "email" in result.categories_found

    def test_multiple_emails(self):
        r = Redactor(["email"])
        result = r.redact("alice@test.com and bob@test.com")
        assert result.redacted_count == 2


class TestRedactorPhone:
    def test_redacts_us_phone(self):
        r = Redactor(["phone"])
        result = r.redact("Call 555-123-4567")
        assert "[PHONE]" in result.text
        assert "555-123-4567" not in result.text

    def test_redacts_phone_with_parens(self):
        r = Redactor(["phone"])
        result = r.redact("Call (555) 123-4567")
        assert "[PHONE]" in result.text

    def test_redacts_phone_with_country_code(self):
        r = Redactor(["phone"])
        result = r.redact("Call +1-555-123-4567")
        assert "[PHONE]" in result.text


class TestRedactorSSN:
    def test_redacts_ssn(self):
        r = Redactor(["ssn"])
        result = r.redact("SSN: 123-45-6789")
        assert "[SSN]" in result.text
        assert "123-45-6789" not in result.text


class TestRedactorCreditCard:
    def test_redacts_credit_card(self):
        r = Redactor(["credit_card"])
        result = r.redact("Card: 4111-1111-1111-1111")
        assert "[CREDIT_CARD]" in result.text

    def test_redacts_card_no_dashes(self):
        r = Redactor(["credit_card"])
        result = r.redact("Card: 4111111111111111")
        assert "[CREDIT_CARD]" in result.text


class TestRedactorApiKey:
    def test_redacts_openai_key(self):
        r = Redactor(["api_key"])
        result = r.redact("Key: sk-abc123def456ghi789jkl012mno345pq")
        assert "[API_KEY]" in result.text

    def test_redacts_github_pat(self):
        r = Redactor(["api_key"])
        result = r.redact("Token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij")
        assert "[API_KEY]" in result.text

    def test_redacts_aws_key(self):
        r = Redactor(["api_key"])
        result = r.redact("AWS: AKIAIOSFODNN7EXAMPLE")
        assert "[API_KEY]" in result.text


class TestRedactorIpAddress:
    def test_redacts_ip(self):
        r = Redactor(["ip_address"])
        result = r.redact("Server at 192.168.1.100")
        assert "[IP_ADDR]" in result.text
        assert "192.168.1.100" not in result.text


class TestRedactorMultipleCategories:
    def test_redacts_multiple(self):
        r = Redactor(["email", "phone"])
        result = r.redact("Email bob@test.com or call 555-123-4567")
        assert "[EMAIL]" in result.text
        assert "[PHONE]" in result.text
        assert result.redacted_count == 2
        assert "email" in result.categories_found
        assert "phone" in result.categories_found

    def test_all_categories_by_default(self):
        r = Redactor()
        assert len(r.categories) >= 6

    def test_no_false_positives_on_clean_text(self):
        r = Redactor()
        result = r.redact("The quick brown fox jumps over the lazy dog")
        assert result.redacted_count == 0
        assert result.text == "The quick brown fox jumps over the lazy dog"


class TestRedactDict:
    def test_redacts_string_values(self):
        r = Redactor(["email"])
        cleaned, count = r.redact_dict({"name": "Bob", "email": "bob@test.com"})
        assert "[EMAIL]" in cleaned["email"]
        assert cleaned["name"] == "Bob"
        assert count == 1

    def test_non_string_values_untouched(self):
        r = Redactor(["email"])
        cleaned, _ = r.redact_dict({"count": 42, "flag": True})
        assert cleaned["count"] == 42
        assert cleaned["flag"] is True
