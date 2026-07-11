"""Human-in-the-loop approval — queue dangerous tool calls for review."""

import asyncio

from agenthandler import ApprovalQueue, Policy, Supervisor


async def delete_database(db_name: str) -> str:
    """Simulate a destructive operation."""
    await asyncio.sleep(0.01)
    return f"Deleted database '{db_name}'"


async def main():
    # Policy that requires confirmation for dangerous tools
    policy = Policy(
        require_confirm=["delete_database"],
        tool_timeout=5.0,
    )
    supervisor = Supervisor(policy)
    queue = ApprovalQueue()

    session_id = "sess_demo_001"

    # --- Submit a dangerous call for approval ---
    request = queue.submit(
        tool_name="delete_database",
        tool_args={"db_name": "production"},
        session_id=session_id,
    )
    print(f"Submitted: {request.tool_name}({request.tool_args})")
    print(f"  ID: {request.approval_id}")
    print(f"  Status: {request.status.value}\n")

    # List pending approvals
    pending = queue.list_pending(session_id=session_id)
    print(f"Pending approvals: {len(pending)}")

    # --- Approve path ---
    approved = queue.approve(request.approval_id, approved_by="admin@corp.com")
    print(f"\nApproved by: {approved.resolved_by}")
    print(f"  Status: {approved.status.value}")

    # Now execute the approved call
    result = await supervisor.call("delete_database", delete_database, db_name="production")
    print(f"  Execution: {result.user_message()}\n")

    # --- Deny path ---
    request2 = queue.submit(
        tool_name="delete_database",
        tool_args={"db_name": "analytics"},
        session_id=session_id,
    )
    denied = queue.deny(
        request2.approval_id,
        reason="Not authorized for analytics DB",
        denied_by="security@corp.com",
    )
    print(f"Denied: {denied.tool_name}({denied.tool_args})")
    print(f"  Reason: {denied.deny_reason}")
    print(f"  Status: {denied.status.value}")

    # Final summary
    all_requests = queue.list_all(session_id=session_id)
    print(f"\nTotal requests: {len(all_requests)}")
    for req in all_requests:
        print(f"  [{req.status.value}] {req.tool_name}({req.tool_args})")


if __name__ == "__main__":
    asyncio.run(main())
