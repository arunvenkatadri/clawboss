"""POML pipeline building example.

Demonstrates generating pipelines from natural language via PipelineBuilder,
and manually parsing POML documents with parse_pipeline_poml().

Run:
    python examples/poml_pipeline.py
"""

import asyncio

from clawboss import MemoryStore, SessionManager
from clawboss.pipeline_poml import PipelineBuilder, parse_pipeline_poml

# -- Simulated tools --


async def sql_query(sql: str = "", **kwargs) -> dict:
    """Execute a SQL query."""
    return {"rows": [{"cnt": 15}], "row_count": 1}


async def escalate(**kwargs) -> str:
    """Escalate to the on-call team."""
    return "Paged on-call engineer via PagerDuty"


async def log_ok(**kwargs) -> str:
    """Log that everything is fine."""
    return "All clear - logged"


async def report(**kwargs) -> str:
    """Write a summary report."""
    return "Report written to #incidents"


TOOL_REGISTRY = {
    "sql.query": sql_query,
    "escalate": escalate,
    "log_ok": log_ok,
    "report": report,
}


# A fake LLM that returns a canned POML response
async def fake_llm(prompt: str) -> str:
    """Simulate an LLM that generates POML from a description."""
    return """\
<!--
metadata:
  name: alert-monitor
  triggers: check alerts, monitor
-->

<task>
Check the alerts table for critical alerts.
If there are more than 10, escalate to the on-call team.
Otherwise log that everything is fine. Then write a summary.
</task>

<pipeline>
  <step tool="sql.query">
    SELECT count(*) as cnt FROM alerts WHERE severity='critical'
  </step>

  <threshold key="rows.0.cnt" value="10">
    <above tool="escalate">Notify on-call team</above>
    <below tool="log_ok">Log all clear</below>
  </threshold>

  <step tool="report">Write summary</step>
</pipeline>"""


async def main() -> None:
    store = MemoryStore()
    mgr = SessionManager(store)

    # --- 1. PipelineBuilder.create() — description to runnable pipeline ---
    print("=== PipelineBuilder.create() ===\n")
    builder = PipelineBuilder(fake_llm, TOOL_REGISTRY, mgr)
    pipeline = await builder.create("Check alerts, escalate if more than 10")
    result = await pipeline.run()
    print(f"  Completed: {result.completed}")
    print(f"  Steps: {len(result.steps)}")
    print(f"  Final output: {result.final_output}")

    # --- 2. PipelineBuilder.create_poml() — get POML text for review ---
    print("\n=== PipelineBuilder.create_poml() ===\n")
    poml = await builder.create_poml("Check alerts, escalate if more than 10")
    print(poml)

    # --- 3. PipelineBuilder.refine() — iterate on existing POML ---
    print("\n=== PipelineBuilder.refine() ===\n")
    refined = await builder.refine(poml, "Add a final report step after the threshold")
    print(f"  Refined POML length: {len(refined)} chars")
    print(f"  Contains <pipeline>: {'<pipeline>' in refined}")

    # --- 4. parse_pipeline_poml() — manual POML parsing ---
    print("\n=== parse_pipeline_poml() — manual parsing ===\n")
    raw_poml = """\
<!--
metadata:
  name: manual-pipeline
  triggers: on-demand
-->

<task>
Run a quick status check and report.
</task>

<pipeline>
  <step tool="sql.query">
    SELECT count(*) as cnt FROM alerts WHERE severity='critical'
  </step>

  <threshold key="rows.0.cnt" value="10">
    <above tool="escalate">Page the on-call engineer</above>
    <below tool="log_ok">All systems nominal</below>
  </threshold>

  <step tool="report">Write incident summary</step>
</pipeline>"""

    pipeline = parse_pipeline_poml(raw_poml, TOOL_REGISTRY, mgr, "manual-agent")
    result = await pipeline.run()
    print(f"  Completed: {result.completed}")
    print(f"  Steps: {len(result.steps)}")
    for step in result.steps:
        print(f"    - {step.name}: {step.result.output}")


if __name__ == "__main__":
    asyncio.run(main())
