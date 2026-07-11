"""PII redaction — detect and replace sensitive data before it leaks to tools or agents."""

import asyncio

from clawboss import Policy, Redactor, Supervisor


async def send_email(to: str, body: str) -> str:
    """Simulate sending an email."""
    await asyncio.sleep(0.01)
    return f"Sent to {to}: {body}"


async def main():
    # --- Direct redaction on strings ---
    redactor = Redactor(categories=["email", "phone", "ssn", "credit_card", "api_key"])

    text = (
        "Contact jane@corp.com or 555-867-5309. "
        "SSN: 123-45-6789. Card: 4111 1111 1111 1111. "
        "Key: sk-abc123xyz456abcdefghijk"
    )
    result = redactor.redact(text)
    print("Original:", text)
    print("Redacted:", result.text)
    print(f"  {result.redacted_count} items found in categories: {result.categories_found}\n")

    # --- Redact all string values in a dict ---
    user_record = {
        "name": "Alice",
        "email": "alice@example.org",
        "phone": "(212) 555-0100",
        "notes": "AWS key is AKIAIOSFODNN7EXAMPLE",
    }
    cleaned, count = redactor.redact_dict(user_record)
    print("Dict redaction:")
    for k, v in cleaned.items():
        print(f"  {k}: {v}")
    print(f"  ({count} redactions)\n")

    # --- Policy-driven redaction in a supervised call ---
    policy = Policy(
        redact=["email", "phone", "ssn"],
        redact_direction="both",
        tool_timeout=5.0,
    )
    supervisor = Supervisor(policy)

    result = await supervisor.call(
        "send_email",
        send_email,
        to="bob@secret.io",
        body="Call me at 415-555-0199, SSN 987-65-4321",
    )
    print("Supervised call with auto-redaction:")
    print(f"  Result: {result.user_message()}")


if __name__ == "__main__":
    asyncio.run(main())
