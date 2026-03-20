# Clawboss

A way of controlling deployed agents to keep your (and their) sanity.

Clawboss wraps your AI agent's tool calls with **timeouts**, **token budgets**, **iteration limits**, **circuit breakers**, and **audit logging**. Zero dependencies. Works with any agent framework.

## Install

```bash
pip install clawboss
```

## Quick start

```python
import asyncio
from clawboss import Supervisor, Policy

# Define limits
policy = Policy(
    max_iterations=5,       # max tool call rounds
    tool_timeout=15.0,      # seconds per tool call
    token_budget=10_000,    # total token cap
)

supervisor = Supervisor(policy)

# Your tool function (any async callable)
async def web_search(query: str) -> str:
    # ... your implementation ...
    return f"Results for: {query}"

async def main():
    # Supervise a tool call
    result = await supervisor.call("web_search", web_search, query="python async")

    if result.succeeded:
        print(result.output)
    else:
        print(f"Failed: {result.error.user_message()}")

    # Track token usage from your LLM calls
    supervisor.record_tokens(1500)

    # Finish and get final stats
    snapshot = supervisor.finish()
    print(f"Used {snapshot.tokens_used} tokens in {snapshot.iterations} iterations")

asyncio.run(main())
```

## What it does

| Feature | What it prevents |
|---------|-----------------|
| **Tool timeout** | A single tool call hanging forever |
| **Token budget** | Runaway LLM costs blowing through your budget |
| **Iteration limit** | Agent loops that never converge |
| **Circuit breaker** | Hammering a tool that keeps failing |
| **Dead man's switch** | Agent going silent (no activity for N seconds) |
| **Confirmation gates** | Dangerous tools running without human approval |
| **Audit log** | Not knowing what your agent did |

## Policy from config

Load policy from a dictionary (YAML, JSON, database — whatever you use):

```python
policy = Policy.from_dict({
    "max_iterations": 10,
    "tool_timeout": 30,
    "token_budget": 50000,
    "on_timeout": "return_error",
    "on_budget_exceeded": "respond_with_best_effort",
    "require_confirm": ["delete_file", "send_email"],
})
```

## Sync support

No event loop? No problem:

```python
result = supervisor.call_sync("calculator", my_sync_fn, x=42)
```

## Audit logging

Every supervised action is recorded. Write to JSONL, stdout, or implement your own sink:

```python
from clawboss import Supervisor, Policy, AuditLog, JsonlAuditSink

# Log to file
sink = JsonlAuditSink.file("audit.jsonl")
audit = AuditLog("request-123", [sink])
supervisor = Supervisor(policy, audit)

# Or log to stdout
sink = JsonlAuditSink.stdout()
```

Custom sink — implement the `AuditSink` interface:

```python
from clawboss import AuditSink, AuditEntry

class MyDatabaseSink(AuditSink):
    def write(self, entry: AuditEntry) -> None:
        db.insert(entry.to_dict())
```

## Circuit breaker

Per-tool circuit breakers stop your agent from hammering a broken tool:

```
CLOSED  →  failures < threshold, calls pass through
OPEN    →  failures >= threshold, calls blocked
HALF_OPEN → after reset period, allow one test call
```

```python
policy = Policy(
    circuit_breaker_threshold=3,   # open after 3 consecutive failures
    circuit_breaker_reset=60.0,    # try again after 60 seconds
)
```

## Failure handlers

Control what happens when limits are hit:

```python
from clawboss import Policy, OnFailure, Action

policy = Policy(
    on_timeout=OnFailure(Action.RETURN_ERROR),
    on_budget_exceeded=OnFailure(Action.RESPOND_WITH_BEST_EFFORT),
    on_max_iterations=OnFailure(Action.RETURN_ERROR, retries=2),
)
```

Actions:
- `RETURN_ERROR` — stop and return the error
- `RESPOND_WITH_BEST_EFFORT` — return what you have so far
- `KILL` — hard stop, no graceful handling

## Skill Builder

Create skills from natural language. Bring your own LLM — pass any async function that takes a prompt and returns text.

```python
from clawboss import SkillBuilder, SkillStore

# Bring your own LLM (OpenAI, Anthropic, local, whatever)
async def my_llm(prompt: str) -> str:
    response = await openai.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

builder = SkillBuilder(my_llm)

# Describe what you want in plain English
skill = await builder.create(
    "A skill that researches topics on the web, limited to 5 searches, "
    "with a 30 second timeout, and asks before deleting anything"
)

# Inspect what was generated
print(skill.name)           # "web_research"
print(skill.supervision)    # {"max_iterations": 5, "tool_timeout": 30, ...}
print(skill.instructions)   # ["Always cite sources", ...]

# Refine it with feedback
skill = await builder.refine(skill, "Add a rule about preferring recent sources")

# Save it
store = SkillStore("~/.clawboss/skills")
store.save(skill)
```

### Managing skills

```python
store = SkillStore("~/.clawboss/skills")

# List all skills
for s in store.list():
    print(f"{s['name']}: {s['description']}")

# Load a full skill
skill = store.get("web_research")

# Delete
store.delete("web_research")

# Export to POML format (for frameworks that use it)
poml_text = store.export_poml("web_research")

# Export all skills as .poml files
store.export_all_poml("./poml_output/")
```

### Skill format

Skills are stored as JSON and can be exported to POML. The format includes:

- **name, description, triggers** — identity and activation
- **role, task, instructions, examples** — what the agent should do
- **tools** — what tools are available (with parameter schemas)
- **supervision** — clawboss limits (maps directly to `Policy.from_dict()`)

## API

### `Policy`

Dataclass with all configuration. Every field has a sensible default.

### `Supervisor(policy, audit=None)`

- `call(tool_name, fn, **kwargs)` — supervise an async tool call
- `call_sync(tool_name, fn, **kwargs)` — supervise a sync tool call
- `record_iteration()` — record an agent loop iteration
- `record_tokens(n)` — record token usage
- `budget()` — get current `BudgetSnapshot`
- `finish()` — mark request complete, return final snapshot

### `SupervisedResult`

- `output` — the tool's return value (if succeeded)
- `error` — `ClawbossError` (if failed)
- `succeeded` — bool
- `duration_ms` — how long the call took
- `budget` — `BudgetSnapshot` at time of completion
- `user_message()` — always returns a string (output or error message)

### `SkillBuilder(llm)`

- `create(description)` — generate a skill from natural language
- `refine(skill, feedback)` — modify a skill with natural language feedback

### `SkillStore(directory)`

- `save(skill)` — save a skill to disk
- `get(name)` — load a skill by name
- `list()` — list all skills (name + description)
- `delete(name)` — delete a skill
- `export_poml(name)` — export a skill as POML text
- `export_all_poml(output_dir)` — export all skills as `.poml` files

### `Skill`

- `to_dict()` / `from_dict(d)` — serialize/deserialize
- `to_poml()` — render as POML format
- `to_json()` — serialize to JSON string

## License

Apache 2.0
