"""POML pipeline parser and builder — natural language to supervised pipelines.

Parses POML documents with ``<pipeline>`` tags into executable Pipeline objects,
and generates POML from natural language descriptions via an LLM.

POML pipeline format::

    <!--
    metadata:
      name: alert-monitor
      triggers: check alerts, monitor
    -->

    <task>
    Check the alerts table for critical alerts.
    If there are more than 10, escalate.
    </task>

    <pipeline>
      <step tool="sql.query">
        SELECT count(*) as cnt FROM alerts WHERE severity='critical'
      </step>

      <threshold key="rows.0.cnt" value="10">
        <above tool="escalate">Notify on-call team</above>
        <below tool="log_ok">Log all clear</below>
      </threshold>

      <condition if="'error' in output">
        <then tool="handle_error">Fix the error</then>
        <else tool="continue">Keep going</else>
      </condition>

      <step tool="report">Write summary</step>
    </pipeline>

Usage:
    # Parse POML into a Pipeline
    from clawboss.pipeline_poml import parse_pipeline_poml, PipelineBuilder

    pipeline = parse_pipeline_poml(poml_text, tool_registry, mgr, "agent-1", policy)
    result = await pipeline.run()

    # Generate POML from natural language
    builder = PipelineBuilder(my_llm, tool_registry)
    pipeline = await builder.create("Check alerts, escalate if > 10")
    result = await pipeline.run()
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Callable, Coroutine, Dict, Optional

from .pipeline import Pipeline
from .session import SessionManager

# ---------------------------------------------------------------------------
# Tool registry type
# ---------------------------------------------------------------------------

ToolRegistry = Dict[str, Callable[..., Coroutine]]


# ---------------------------------------------------------------------------
# POML pipeline parser
# ---------------------------------------------------------------------------


def parse_pipeline_poml(
    poml_text: str,
    tools: ToolRegistry,
    manager: SessionManager,
    agent_id: str,
    policy_dict: Optional[Dict[str, Any]] = None,
    stateless: bool = False,
) -> Pipeline:
    """Parse a POML document with ``<pipeline>`` tags into an executable Pipeline.

    Args:
        poml_text: The POML text containing a <pipeline> block.
        tools: Registry mapping tool names to async callables.
        manager: SessionManager for the pipeline session.
        agent_id: Agent identifier.
        policy_dict: Policy configuration.
        stateless: Whether the pipeline session is stateless.

    Returns:
        A configured Pipeline ready to .run().
    """
    # Extract the <pipeline>...</pipeline> block
    pipeline_match = re.search(r"<pipeline>(.*?)</pipeline>", poml_text, re.DOTALL)
    if not pipeline_match:
        raise ValueError("No <pipeline> block found in POML text")

    pipeline_xml = pipeline_match.group(1).strip()

    # Wrap in a root element for XML parsing
    try:
        root = ET.fromstring(f"<root>{pipeline_xml}</root>")
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in <pipeline> block: {e}") from e

    pipeline = Pipeline(manager, agent_id, policy_dict=policy_dict, stateless=stateless)

    for node in root:
        if node.tag == "step":
            _parse_step(node, tools, pipeline)
        elif node.tag == "threshold":
            _parse_threshold(node, tools, pipeline)
        elif node.tag == "condition":
            _parse_condition(node, tools, pipeline)
        else:
            raise ValueError(f"Unknown pipeline tag: <{node.tag}>")

    return pipeline


def _parse_step(node: ET.Element, tools: ToolRegistry, pipeline: Pipeline) -> None:
    """Parse a <step tool="name">sql or description</step>."""
    tool_name = node.attrib.get("tool", "")
    if not tool_name:
        raise ValueError("<step> requires a 'tool' attribute")
    if tool_name not in tools:
        raise ValueError(f"Tool '{tool_name}' not found in registry")

    content = (node.text or "").strip()
    name = node.attrib.get("name", tool_name)

    # Build kwargs from attributes (excluding tool and name)
    kwargs: Dict[str, Any] = {}
    for k, v in node.attrib.items():
        if k not in ("tool", "name"):
            kwargs[k] = v

    # If content looks like SQL, pass as sql kwarg. Otherwise it's just a label.
    if content and _looks_like_sql(content):
        kwargs["sql"] = content

    chain = node.attrib.get("chain", "true").lower() != "false"
    pipeline.add_step(tool_name, tools[tool_name], name=name, chain_input=chain, **kwargs)


def _parse_threshold(node: ET.Element, tools: ToolRegistry, pipeline: Pipeline) -> None:
    """Parse a <threshold key="..." value="..."><above>/<below></threshold>."""
    key = node.attrib.get("key", "")
    value_str = node.attrib.get("value", "0")
    if not key:
        raise ValueError("<threshold> requires a 'key' attribute")
    try:
        threshold_val = float(value_str)
    except ValueError as e:
        raise ValueError(f"<threshold> value must be numeric: {value_str}") from e

    above_step = None
    below_step = None

    for child in node:
        tool_name = child.attrib.get("tool", "")
        if not tool_name or tool_name not in tools:
            raise ValueError(f"<{child.tag}> requires a valid 'tool' attribute, got '{tool_name}'")
        if child.tag == "above":
            above_step = (tool_name, tools[tool_name])
        elif child.tag == "below":
            below_step = (tool_name, tools[tool_name])

    if above_step is None:
        raise ValueError("<threshold> requires an <above> child")

    pipeline.add_threshold(
        key=key,
        threshold=threshold_val,
        above_step=above_step,
        below_step=below_step,
    )


def _parse_condition(node: ET.Element, tools: ToolRegistry, pipeline: Pipeline) -> None:
    """Parse a <condition if="expr"><then>/<else></condition>."""
    expr = node.attrib.get("if", "")
    if not expr:
        raise ValueError("<condition> requires an 'if' attribute")

    then_step = None
    else_step = None

    for child in node:
        tool_name = child.attrib.get("tool", "")
        if not tool_name or tool_name not in tools:
            raise ValueError(f"<{child.tag}> requires a valid 'tool' attribute, got '{tool_name}'")
        if child.tag == "then":
            then_step = (tool_name, tools[tool_name])
        elif child.tag == "else":
            else_step = (tool_name, tools[tool_name])

    if then_step is None:
        raise ValueError("<condition> requires a <then> child")

    # Build a safe predicate from the expression
    predicate = _make_predicate(expr)
    pipeline.add_condition(predicate=predicate, then_step=then_step, else_step=else_step)


def _make_predicate(expr: str) -> Callable[[Any], bool]:
    """Build a predicate function from a simple expression string.

    Supported expressions:
    - "output.rows.0.cnt >= 10" — dot-notation comparison
    - "'error' in output" — containment check
    - "output == 'done'" — equality

    Uses a restricted eval with no builtins for safety.
    """

    def _predicate(output: Any) -> bool:
        # Make 'output' available in the expression
        safe_globals: Dict[str, Any] = {"__builtins__": {}}
        safe_locals: Dict[str, Any] = {"output": output}

        # Support dot notation: output.rows.0.cnt → resolve manually
        resolved_expr = _resolve_dot_notation(expr, output)

        try:
            return bool(eval(resolved_expr, safe_globals, safe_locals))  # noqa: S307
        except Exception:
            return False

    return _predicate


def _resolve_dot_notation(expr: str, output: Any) -> str:
    """Replace output.x.y.z with the actual resolved value in the expression."""

    # Find patterns like output.foo.bar.0.baz
    def _replace(match: re.Match) -> str:
        path = match.group(0)
        parts = path.split(".")
        val = output
        for part in parts[1:]:  # skip 'output'
            if isinstance(val, dict):
                val = val.get(part)
            elif isinstance(val, (list, tuple)):
                try:
                    val = val[int(part)]
                except (IndexError, ValueError):
                    return "None"
            else:
                return "None"
        return repr(val)

    return re.sub(r"output(?:\.\w+)+", _replace, expr)


def _looks_like_sql(text: str) -> bool:
    """Check if text looks like a SQL statement."""
    normalized = text.strip().upper()
    return normalized.startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "CREATE"))


# ---------------------------------------------------------------------------
# POML pipeline generation prompt
# ---------------------------------------------------------------------------

PIPELINE_GENERATION_PROMPT = """\
You are a pipeline builder for AI agents. Given a natural language description \
of what the pipeline should do, produce a POML pipeline definition.

Available tools:
{tools}
{schema_section}
The POML format uses XML-style tags inside a <pipeline> block:

```xml
<pipeline>
  <step tool="tool_name">
    SQL query or description of what this step does
  </step>

  <threshold key="rows.0.cnt" value="10">
    <above tool="tool_name">What to do if above</above>
    <below tool="tool_name">What to do if below</below>
  </threshold>

  <condition if="'error' in output">
    <then tool="tool_name">What to do if true</then>
    <else tool="tool_name">What to do if false</else>
  </condition>
</pipeline>
```

Rules:
- tool attributes must match one of the available tools listed above
- <step> content: if it's a SQL query, write correct SQL using the database schema above. \
Use the actual table and column names from the schema.
- <threshold> key uses dot notation to extract a value from the previous step's output. \
For SQL query results, the format is rows.0.column_name
- <condition> if uses simple Python expressions with 'output' as the variable
- Steps chain automatically — each step receives the previous step's output
- Add chain="false" to a step to ignore previous output
- Keep pipelines focused — prefer fewer steps that do more over many trivial steps

Respond with ONLY the POML text. Include a metadata comment and task description.\
"""

PIPELINE_REFINEMENT_PROMPT = """\
You are refining an existing POML pipeline based on user feedback.

Current pipeline:
```
{current_poml}
```

Available tools:
{tools}

The user wants the following changes:
{feedback}

Apply the changes and return the complete updated POML. Respond with ONLY the POML text.\
"""


# ---------------------------------------------------------------------------
# PipelineBuilder — natural language to POML to Pipeline
# ---------------------------------------------------------------------------


class PipelineBuilder:
    """Build pipelines from natural language using any LLM.

    Same bring-your-own-LLM pattern as SkillBuilder. Pass any async
    callable that takes a prompt string and returns a string.

    Args:
        llm: Async callable — your LLM function.
        tools: Registry mapping tool names to async callables.
        manager: SessionManager for created pipelines.
        db_schema: Database schema dict from SqlConnector.discover_schema().
                   Included in the LLM prompt so it writes correct SQL.
        system_prompt: Override the default generation prompt.
    """

    def __init__(
        self,
        llm: Callable[[str], Coroutine[Any, Any, str]],
        tools: ToolRegistry,
        manager: SessionManager,
        db_schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ):
        self._llm = llm
        self._tools = tools
        self._manager = manager
        self._db_schema = db_schema
        self._system_prompt = system_prompt or PIPELINE_GENERATION_PROMPT

    def _tools_description(self) -> str:
        """Format tool registry as a description for the LLM."""
        lines = []
        for name in sorted(self._tools.keys()):
            fn = self._tools[name]
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            lines.append(f"- {name}: {doc}" if doc else f"- {name}")
        return "\n".join(lines)

    def _schema_description(self) -> str:
        """Format the database schema for the LLM prompt."""
        if self._db_schema is None:
            return ""

        # SQL schema
        if "tables" in self._db_schema:
            from .connectors import SqlConnector

            connector = SqlConnector.__new__(SqlConnector)
            return connector.schema_to_text(self._db_schema)

        # MongoDB schema
        if "collections" in self._db_schema:
            lines = []
            for coll in self._db_schema["collections"]:
                header = f"Collection: {coll['name']}"
                if "document_count" in coll:
                    header += f" ({coll['document_count']} documents)"
                lines.append(header)
                for f in coll.get("fields", []):
                    lines.append(f"  - {f['name']} ({f.get('type', 'unknown')})")
                lines.append("")
            return "\n".join(lines)

        return ""

    def _build_prompt(self) -> str:
        """Build the full LLM prompt with tools and schema."""
        schema_text = self._schema_description()
        schema_section = ""
        if schema_text:
            schema_section = f"\nDatabase schema:\n{schema_text}\n"
        return self._system_prompt.format(
            tools=self._tools_description(),
            schema_section=schema_section,
        )

    async def create(
        self,
        description: str,
        agent_id: str = "pipeline-agent",
        policy_dict: Optional[Dict[str, Any]] = None,
        stateless: bool = False,
    ) -> Pipeline:
        """Generate a pipeline from a natural language description.

        Args:
            description: What the pipeline should do, in plain English.
            agent_id: Agent identifier for the pipeline session.
            policy_dict: Policy configuration.
            stateless: Whether the session is stateless.

        Returns:
            A configured Pipeline ready to .run().
        """
        prompt = self._build_prompt()
        prompt += f"\n\nUser's description:\n{description}"

        raw = await self._llm(prompt)
        poml = self._extract_poml(raw)

        return parse_pipeline_poml(
            poml,
            self._tools,
            self._manager,
            agent_id,
            policy_dict=policy_dict,
            stateless=stateless,
        )

    async def create_poml(self, description: str) -> str:
        """Generate POML text from a description without parsing it.

        Useful for reviewing/editing the POML before running.
        """
        prompt = self._build_prompt()
        prompt += f"\n\nUser's description:\n{description}"

        raw = await self._llm(prompt)
        return self._extract_poml(raw)

    async def refine(
        self,
        current_poml: str,
        feedback: str,
    ) -> str:
        """Refine existing POML based on feedback. Returns updated POML text."""
        prompt = PIPELINE_REFINEMENT_PROMPT.format(
            current_poml=current_poml,
            tools=self._tools_description(),
            feedback=feedback,
        )
        raw = await self._llm(prompt)
        return self._extract_poml(raw)

    @staticmethod
    def _extract_poml(text: str) -> str:
        """Extract POML from LLM output, stripping markdown fences."""
        text = text.strip()
        text = re.sub(r"^```(?:xml|poml)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        return text.strip()
