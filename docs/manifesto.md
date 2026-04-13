# The Long-Duration Agent Manifesto

Most agent frameworks are built for agents that run for minutes. A user asks a question, the agent calls a few tools, returns an answer, and exits. The assumptions built into these frameworks — ephemeral state, no crash recovery, a single trace through the code — are fine for that use case.

But the interesting agents don't look like that. The interesting agents run for days. They watch data streams. They take notes. They escalate to humans when they're unsure. They recover from crashes and resume where they left off. They reflect on whether what they just did actually advanced their goal.

Those agents need a different foundation.

## What's broken about short-duration thinking

When you build an agent to run for minutes, certain problems are invisible:

- **Crashes don't matter.** If it crashes, the user retries. No state to lose.
- **Budgets don't matter much.** A few bad tool calls won't drain your account.
- **Drift isn't a concern.** The task is so short the agent can't really go off-course.
- **Audit is for debugging, not compliance.** You read logs when something breaks, not because regulators require it.
- **Supervision happens in the prompt.** You tell the LLM "don't do X," and hope.

None of those assumptions hold when the agent runs for 72 hours.

- Crashes absolutely happen over 72 hours, and losing that much progress is unacceptable.
- A loop for 72 hours at $0.03/call destroys budgets.
- Drift is the dominant failure mode: agents that were on-task hour 1 are nowhere near their goal hour 40.
- Compliance cares about 72 hours of audit data, not five minutes.
- Prompt-level supervision fails under real adversarial pressure. You need code-level enforcement.

## What long-duration agents need

After building Clawboss, I'm convinced the minimum viable infrastructure for production long-duration agents looks like this:

### Durable state, or it didn't happen

Every tool call has to checkpoint somewhere. When the process dies (and it will), a new process picks up the session at the exact point it left off. Budget counters, circuit breaker states, tool call history — all restored. No losing 6 hours of research because a container got recycled.

### Crash loop protection

If the agent keeps crashing and being resumed, stop it. At some point "try again" becomes "you're in a bad state, halt before you bankrupt me." A simple counter with a configurable limit does the job.

### Policy that can't be rewritten

The agent cannot be trusted to enforce its own limits. Put the policy in a place the agent can't touch — signed with an HMAC, checked on every resume, separated from the agent-writable session payload. If someone edits the SQLite file directly, the resume fails.

### Guardrails on every tool call

Not just timeouts and budgets. 16 of them, layered: rule-based checks that are cheap and fast, then LLM-backed checks for the things rules can't catch. Prompt injection detection. Intent drift detection. Schema validation. URL allowlists. Output safety classification. Resource quotas. Ensemble decisions for the most critical actions.

Every single tool call goes through this gauntlet. Every blocked call is audited. Every LLM-backed check is opt-in and zero-overhead when disabled.

### Reflection, not just looping

`while not done: agent.step()` isn't enough. You need structured cycles: think → act → observe → reflect. Each phase is its own LLM call, each phase is audited, and the reflection output feeds into the next think phase. This is how agents stay on-task over hours.

### Human-in-the-loop that actually works

Not "the agent paused and exited." Approval queues that wait patiently, surface via REST/WebSocket/dashboard, and let a human click "approve" or "deny" whenever they get to it. The agent sits in its session, ready to resume.

### Observability built in

Per-tool metrics. Per-session metrics. Recent call log. Anomaly detection against session history. Everything exportable to OpenTelemetry so it plugs into whatever your SRE team uses. When an agent does something weird at 3am, you find out at 3:01am.

### Stream-first input

The interesting long-duration agents react to events, they don't poll. Kafka, Kinesis, Redis Streams — agents should be able to subscribe to a topic and run their pipeline on every message. Crash-safe, at-least-once delivery, automatic commit after successful processing.

### Time-travel and replay

When something goes wrong after 40 hours, you need to reconstruct what happened step by step. Every tool call, every decision, every guardrail fire. Stored once in the audit log, replayable on demand. "Why did the agent do X?" becomes a solvable question.

## What managed agent platforms are missing

Anthropic, OpenAI, and the rest are building managed agent platforms. They'll scale beautifully, integrate natively with their models, and handle the boring infrastructure. That's great.

But they don't give you:

- **Your infrastructure, your data.** Healthcare, finance, government — your agent data can't leave your environment.
- **Framework agnostic.** Any LLM, any framework, any model. Not locked to one vendor.
- **Opinionated guardrails.** They offer generic tooling; long-duration agents need specific protections.
- **Declarative safety.** You define in code what the agent can and can't do. That code is auditable, version-controllable, testable.
- **Transparent supervision.** Not a black box — every decision is logged with the reason.

## What Clawboss is

Clawboss is the open-source foundation for long-duration agents. Everything in this manifesto is already built:

- Durable sessions with crash recovery and policy checksums
- 16 guardrails (8 deterministic, 8 LLM-backed)
- Structured reflection loops for multi-step reasoning
- Kafka/Kinesis/Redis streaming input
- Triggers and scheduling (cron, interval, webhook, DB watch)
- Human-in-the-loop approval flow
- Session replay through the audit log
- Observability with OpenTelemetry export
- REST control plane with auth
- Dashboard UI for the whole stack
- Zero required dependencies for the core library

You run it on your infrastructure. Any LLM framework. Any database. Any model.

## Why this matters

The next two years of AI will be defined by what agents can do without a human watching every move. That requires an infrastructure layer that doesn't yet exist in the managed platforms. Clawboss is one answer.

If you're building long-duration agents and the managed platforms aren't giving you what you need, try Clawboss. Fork it, break it, file issues. This is where long-running agents live now.

---

**Code:** [github.com/arunvenkatadri/clawboss](https://github.com/arunvenkatadri/clawboss)
**Install:** `pip install clawboss`
**Author:** Arun Venkatadri
