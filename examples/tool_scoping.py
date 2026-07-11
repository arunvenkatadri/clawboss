"""Tool scoping — parameter-level permissions and rate limits."""

from clawboss import Policy, ScopeRule, ToolScope


def main():
    # Define scope rules for a file_write tool
    file_scope = ToolScope(
        tool_name="file_write",
        rules=[
            # Only allow writes to /tmp/ paths
            ScopeRule(param="path", constraint="allow", values=["/tmp/*", "/var/tmp/*"]),
            # Block writes to any .env file
            ScopeRule(param="path", constraint="block", values=["*.env", "*.secret"]),
            # Content must match a safe pattern (no shell commands)
            ScopeRule(param="content", constraint="match", values=[r"^[A-Za-z0-9\s.,!?]+$"]),
        ],
        max_calls_per_minute=10,
    )

    # Define scope rules for an HTTP tool
    http_scope = ToolScope(
        tool_name="http_request",
        rules=[
            # Only allow requests to approved domains
            ScopeRule(
                param="url",
                constraint="allow",
                values=["https://api.example.com/*", "https://internal.corp/*"],
            ),
            # Block any localhost/internal IPs
            ScopeRule(param="url", constraint="block", values=["*localhost*", "*127.0.0.1*"]),
        ],
        max_calls_per_minute=30,
    )

    # Attach scopes to a policy
    policy = Policy(
        tool_scopes=[file_scope, http_scope],
        tool_timeout=10.0,
    )

    print("Policy tool scopes:")
    for scope in policy.tool_scopes:
        print(f"  {scope.tool_name} (max {scope.max_calls_per_minute} calls/min)")
        for rule in scope.rules:
            print(f"    {rule.constraint} {rule.param}: {rule.values}")

    # --- Check calls against the scope ---
    print("\n--- file_write scope checks ---")

    # Allowed: writing to /tmp
    err = file_scope.check_args({"path": "/tmp/output.txt", "content": "Hello world"})
    print(f"Write to /tmp/output.txt: {'PASS' if err is None else 'BLOCKED: ' + err}")

    # Blocked: writing to a .env file
    err = file_scope.check_args({"path": "/tmp/.env", "content": "SECRET=abc"})
    print(f"Write to /tmp/.env: {'PASS' if err is None else 'BLOCKED: ' + err}")

    # Blocked: path outside allowed directories
    err = file_scope.check_args({"path": "/etc/passwd", "content": "root"})
    print(f"Write to /etc/passwd: {'PASS' if err is None else 'BLOCKED: ' + err}")

    print("\n--- http_request scope checks ---")

    # Allowed: approved domain
    err = http_scope.check_args({"url": "https://api.example.com/v1/data"})
    print(f"GET api.example.com: {'PASS' if err is None else 'BLOCKED: ' + err}")

    # Blocked: localhost
    err = http_scope.check_args({"url": "https://localhost:8080/admin"})
    print(f"GET localhost:8080: {'PASS' if err is None else 'BLOCKED: ' + err}")

    # Blocked: unapproved domain
    err = http_scope.check_args({"url": "https://evil.com/steal"})
    print(f"GET evil.com: {'PASS' if err is None else 'BLOCKED: ' + err}")


if __name__ == "__main__":
    main()
