# Clawboss

[![CI](https://github.com/arunvenkatadri/Clawboss/actions/workflows/ci.yml/badge.svg)](https://github.com/arunvenkatadri/Clawboss/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen.svg)]()

**Stop your AI agents from going rogue.** Clawboss wraps tool calls with timeouts, budgets, circuit breakers, and audit logging so one bad tool call doesn't drain your wallet or loop forever.

Zero dependencies. Works with **any agent framework** — LangChain, CrewAI, AutoGen, OpenClaw, your own custom loop, whatever. Just wrap your tool calls. Includes durable sessions that survive restarts, a REST control plane, and a [dashboard](#dashboard) for managing everything in one place.

## Why

You deploy an agent. It calls a flaky API in a loop. 47 times. At $0.03 per call. At 3am. Nobody's watching.

Or: your agent decides to "keep researching" and burns through your entire token budget in one conversation. Or: a tool hangs for 90 seconds and your user stares at a spinner.

Clawboss is the guardrail layer between your agent and its tools. Every tool call goes through supervision — timeouts, budgets, circuit breakers — so you can deploy agents without white-knuckling it.

### No arbitrary code downloads

Most agent platforms want you to install skills from a community marketplace — arbitrary code that runs unsandboxed in your agent's process. One bad plugin and your agent has full access to your filesystem, credentials, and network.

Clawboss takes a different approach. **You define skills and agents declaratively** — what tools are available, what parameters they accept, what supervision limits apply. No downloading stranger code. No hoping someone reviewed that community plugin before you installed it. You control exactly what your agents can do, and every tool call goes through supervision whether you built it or someone else did.

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

## Dashboard

Open `dashboard.html` in a browser for a full management UI:

- **Agents** — create, edit, delete, pause/resume/stop agents with supervision policies
- **Skills** — define reusable capabilities (tool collections) and assign them to agents
- **Sessions** — live view of running agent sessions from the REST API, with pause/resume/stop controls, budget usage, and audit logs
- **Chat** — open a conversation with any agent directly from the dashboard
- **Costs** — track spend, set budgets with hard stops, view usage over time
- **Policies** — see all active supervision rules at a glance

The Sessions tab connects to the REST control plane (`uvicorn clawboss.server:app`) and shows real-time session data. Agent cards show live status and controls work against the real API.

<img width="1498" height="953" alt="Screenshot 2026-03-25 at 6 05 30 PM" src="https://github.com/user-attachments/assets/11a5047c-6328-43bc-a6cf-d56a9b0b45da" />

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
| **Durable sessions** | Agent dies mid-task, loses all progress |
| **REST control plane** | No way to pause/resume/stop agents remotely |

## Works with any agent framework

Clawboss doesn't care what framework you use. It supervises tool calls — any async or sync callable. If your agent calls tools, Clawboss can wrap them.

```python
# LangChain? Wrap your tools.
# CrewAI? Wrap your tools.
# AutoGen? Wrap your tools.
# Custom loop? Wrap your tools.
# OpenClaw? There's a built-in bridge (see below).

result = await supervisor.call("my_tool", my_tool_fn, **kwargs)
```

## Durable sessions

Long-running agents survive process restarts. Clawboss checkpoints supervisor state (iterations, token usage, circuit breaker states) to a pluggable store after every operation.

```python
from clawboss import SessionManager, SqliteStore

store = SqliteStore("sessions.db")  # or MemoryStore() for testing
mgr = SessionManager(store)

# Start a session
session_id = mgr.start("my-agent", {
    "max_iterations": 20,
    "tool_timeout": 30,
    "token_budget": 50000,
})

# Get the supervisor and use it in your agent loop
sv = mgr.get_supervisor(session_id)
result = await sv.call("web_search", search_fn, query="python async")
sv.record_tokens(1500)

# Pause — the supervisor raises AgentPaused on next call()
mgr.pause(session_id)

# Resume later (even after a crash / restart)
sv = mgr.resume(session_id)   # budget, iterations, circuit breakers all restored
result = await sv.call("web_search", search_fn, query="continue research")

# Stop when done
mgr.stop(session_id)
```

### Pluggable storage

Implement the `StateStore` protocol for your own backend:

```python
from clawboss import StateStore, Checkpoint

class RedisStore:
    def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...
    def load_checkpoint(self, session_id: str) -> Checkpoint | None: ...
    def list_sessions(self) -> list[Checkpoint]: ...
    def delete_session(self, session_id: str) -> bool: ...
```

Ships with `SqliteStore` (production default, stdlib sqlite3) and `MemoryStore` (testing).

### Stateless sessions

Not every agent needs crash recovery. Pass `stateless=True` to skip auto-checkpointing — you still get supervision, audit logging, and pause/stop controls, but no disk writes on each tool call and no crash recovery.

```python
# In-memory only — no checkpoints, no crash recovery
session_id = mgr.start("quick-agent", policy_dict, stateless=True)
```

Via the REST API:

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "quick-agent", "policy": {...}, "stateless": true}'
```

Stateless sessions can be paused and stopped normally. They cannot be resumed after a process restart — if the process dies, the session is gone.

## REST control plane

Manage agent sessions remotely over HTTP. Optional dependency — install with:

```bash
pip install clawboss[server]
```

Start the server:

```bash
uvicorn clawboss.server:app
```

With API key auth:

```bash
# Pick any string as your secret — there's no signup or external service
CLAWBOSS_API_KEY=my-secret-key uvicorn clawboss.server:app
```

Clients pass the key as a Bearer token:

```bash
curl -H "Authorization: Bearer my-secret-key" http://localhost:8000/sessions
```

WebSocket connections pass it as a query param: `ws://localhost:8000/sessions/{id}/events?token=my-secret-key`

If `CLAWBOSS_API_KEY` is not set, auth is disabled (open access — fine for local dev). You can also pass `api_key=` directly to `create_app()` in code.

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/sessions` | Create a new agent session |
| GET | `/sessions` | List all sessions with status |
| GET | `/sessions/{id}` | Session detail (budget, checkpoint time, policy) |
| POST | `/sessions/{id}/pause` | Pause an agent |
| POST | `/sessions/{id}/resume` | Resume a paused agent |
| POST | `/sessions/{id}/stop` | Stop an agent |
| GET | `/sessions/{id}/audit` | Audit log entries for this session |
| WS | `/sessions/{id}/events` | Stream status changes and audit events |

```bash
# Create a session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "researcher", "policy": {"max_iterations": 10, "token_budget": 50000}}'

# Pause it
curl -X POST http://localhost:8000/sessions/{id}/pause

# Check status
curl http://localhost:8000/sessions/{id}
```

## Security model

Clawboss is designed to supervise untrusted agent behavior. The stateful session layer enforces several invariants:

**Policy is immutable.** The supervision policy (timeouts, budgets, confirmation gates) is frozen at `start()` and cannot be changed by the agent. `resume()` always rebuilds from the original policy — even if the stored checkpoint is tampered with, the agent cannot weaken its own supervision.

**Payload is untrusted.** The `payload` field is agent-writable storage for intermediate work. It is validated for size (1 MB limit) and serializability, but its *contents* should be treated like user input. If your agent reads from payload after a resume, sanitize it.

**Session IDs are cryptographic.** 128-bit random IDs via `secrets.token_hex` — not guessable or enumerable.

**The REST API supports API key auth.** Set `CLAWBOSS_API_KEY` to enable Bearer token authentication on all endpoints. CORS is restricted to localhost by default. Always enable auth before exposing the server to untrusted networks.

**SQLite files are owner-only.** The default `SqliteStore` creates database files with `0600` permissions.

**Audit logs survive crashes.** Entries are persisted to the checkpoint store on `pause()` and `stop()`, so you don't lose the trail if the process dies.

**Sessions can expire.** Call `SqliteStore.delete_expired(max_age_seconds)` to clean up old sessions.

## OpenClaw integration

Clawboss includes a built-in bridge for [OpenClaw](https://github.com/openclaw/openclaw). Expose your supervised tools to OpenClaw over HTTP — all supervision (timeouts, budgets, circuit breakers) applies automatically.

```python
from clawboss import OpenClawBridge, Skill, ToolDefinition, ToolParameter

# Define your skill with tools and supervision limits
skill = Skill(
    name="web_research",
    description="Research topics on the web",
    tools=[
        ToolDefinition(
            name="web_search",
            description="Search the web",
            parameters=[
                ToolParameter(name="query", type="string",
                              description="Search query", required=True),
            ],
        ),
    ],
    supervision={"tool_timeout": 15, "max_iterations": 5, "token_budget": 10000},
)

# Start the bridge
bridge = OpenClawBridge(port=9229)
bridge.register_skill(skill, {"web_search": my_search_fn})
bridge.serve()  # GET /tools, POST /execute/{name}
```

Then install the TypeScript plugin from `openclaw-plugin/` into OpenClaw. The plugin auto-discovers tools from the bridge and registers them. See `examples/openclaw_bridge.py` for a full working example.

You can also convert schemas without running a bridge:

```python
from clawboss import to_openclaw_tool_schema, to_openclaw_manifest

schema = to_openclaw_tool_schema(tool_def)    # OpenClaw JSON Schema format
manifest = to_openclaw_manifest(skill)         # openclaw.plugin.json content
```

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
CLOSED    ->  failures < threshold, calls pass through
OPEN      ->  failures >= threshold, calls blocked
HALF_OPEN ->  after reset period, allow one test call
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
<img width="1509" height="944" alt="Screenshot 2026-03-25 at 6 06 28 PM" src="https://github.com/user-attachments/assets/e1e28e88-0ce5-41fb-b0a6-32d16dd59f86" />


Skills are stored as JSON and can be exported to POML. The format includes:

- **name, description, triggers** — identity and activation
- **role, task, instructions, examples** — what the agent should do
- **tools** — what tools are available (with parameter schemas)
- **supervision** — clawboss limits (maps directly to `Policy.from_dict()`)

## API

### `Policy`

Dataclass with all configuration. Every field has a sensible default.

### `Supervisor(policy, audit=None, store=None, session_id=None, agent_id=None)`

- `call(tool_name, fn, **kwargs)` — supervise an async tool call
- `call_sync(tool_name, fn, **kwargs)` — supervise a sync tool call
- `record_iteration()` — record an agent loop iteration
- `record_tokens(n)` — record token usage
- `budget()` — get current `BudgetSnapshot`
- `finish()` — mark request complete, return final snapshot
- `to_checkpoint_data()` — export state for persistence
- `restore_from_checkpoint(checkpoint)` — rebuild from a checkpoint

### `SupervisedResult`

- `output` — the tool's return value (if succeeded)
- `error` — `ClawbossError` (if failed)
- `succeeded` — bool
- `duration_ms` — how long the call took
- `budget` — `BudgetSnapshot` at time of completion
- `user_message()` — always returns a string (output or error message)

### `OpenClawBridge(policy, audit, host, port)`

- `register_tool(tool, fn)` — register a tool with its async callable
- `register_skill(skill, tool_impls)` — register all tools from a skill
- `serve()` — start the bridge (blocking)
- `serve_background()` — start the bridge in a background thread
- `shutdown()` — stop the bridge

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

### `SessionManager(store)`

- `start(agent_id, policy_dict, payload)` — create a new session, returns `session_id`
- `pause(session_id)` — pause (supervisor raises `AgentPaused` on next call)
- `resume(session_id)` — rehydrate supervisor from last checkpoint
- `stop(session_id)` — stop and finalize
- `status(session_id)` — get current checkpoint
- `list_sessions()` — list all sessions
- `get_supervisor(session_id)` — get the active supervisor
- `get_audit_entries(session_id)` — get audit log entries
- `update_payload(session_id, payload)` — update opaque agent payload

### `StateStore` (protocol)

- `save_checkpoint(checkpoint)` — persist a checkpoint
- `load_checkpoint(session_id)` — load by ID (returns `None` if missing)
- `list_sessions()` — list all checkpoints
- `delete_session(session_id)` — delete a checkpoint

Implementations: `SqliteStore(db_path)`, `MemoryStore()`

## Contributing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check .
ruff format --check .

# Type check
mypy clawboss/
```

## License

Apache 2.0
