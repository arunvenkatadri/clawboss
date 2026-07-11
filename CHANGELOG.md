# Changelog

All notable changes to AgentHandler are documented here.

## [0.92.0] — 2026-07-11

### Added
- **Dashboard**: Live agent chat with full supervision, LLM-powered agent builder, visual pipeline editor with drag-to-reorder, budget overrides per agent, localStorage persistence.
- **POML security hardening**: Replaced unsafe `eval()` in condition expressions with an AST-based whitelist evaluator. Function calls, attribute access, comprehensions, lambdas, and imports are rejected at parse time.
- **PipelineBuilder validation**: `create_poml()` and `refine()` now validate that LLM output contains a parseable `<pipeline>` block before returning.
- 28 new POML tests (expression safety, parse error edge cases, builder validation).
- Test count badge in README.

### Changed
- **mypy --strict** enabled across all 29 source files with zero errors. Full type annotations on every function.
- Python 3.13 compatibility: replaced deprecated `asyncio.get_event_loop()` in approval tests.

## [0.91.0] — 2026-06-10

### Added
- **Cost attribution**: `PricingTable` with built-in model pricing (Claude Opus/Sonnet/Haiku, GPT-4o family, Gemini Pro/Flash).
- `Observer` extended with `input_tokens`, `output_tokens`, `cost_usd`, and `model` per call.
- `cost_summary()` with breakdowns by agent, session, tool, and model.
- REST endpoints: `GET /metrics/costs`, `GET /metrics/pricing`.
- Dashboard Costs tab wired to live data from the API.
- `SessionManager` defaults to `PricingTable.default()` — cost attribution works out of the box.
- 22 new tests (630 total).

## [0.90.0] — 2026-06-09

### Added
- **Session replay**: `SessionReplay` reconstructs agent timelines from audit logs. `frames()`, `state_at()`, `filter()`, `summary()`, `to_timeline()`. REST endpoint: `GET /sessions/{id}/replay`.
- **Reflection loops**: Structured think → act → observe → reflect cycles via `ReflectionLoop`. Each phase is a separate LLM call, fully supervised and audited.
- **Long-duration agent manifesto** (`docs/manifesto.md`).
- Example: `examples/long_running_agent.py`.

## [0.89.0] — 2026-06-08

### Added
- **16 guardrails** — 8 deterministic + 8 LLM-backed:
  - Deterministic: `SchemaValidator`, `CategoryRateLimit`, `RecursionDetector`, `IdempotencyGuard`, `ResourceQuota`, `OutputLengthLimit`, `UrlGuard`, `ActiveHours`.
  - LLM-backed: `PromptInjectionDetector`, `SafetyClassifier`, `IntentDriftDetector`, `SemanticPiiRedactor`, `AnomalyScorer`, `GoalVerifier`, `ExplanationRequired`, `EnsembleDecision`.
- All guardrails hook into `Supervisor` via `pre_guardrails` / `post_guardrails`.
- Uniform `GuardrailResult(allowed, reason, replacement_output)` interface.

## [0.88.0] — 2026-06-07

### Added
- **LLM decisions**: `Pipeline.add_llm_decision()` for structured JSON decision steps. LLM reads previous output + session context, returns structured JSON for routing.
- **Stateful context**: `Pipeline.with_context(session_id)` for multi-run agents. Rolling history (last 20 runs) in session payload.
- **Stream input injection**: `Pipeline.run(initial_input=msg)` seeds the first step with a stream message payload.

## [0.87.0] — 2026-06-06

### Added
- **Streaming inputs**: `KafkaStreamConnector`, `KinesisStreamConnector`, `RedisStreamConnector`. At-least-once delivery. Optional extras: `agenthandler[kafka]`, `agenthandler[kinesis]`, `agenthandler[redis]`, `agenthandler[streams]`.

## [0.86.0] — 2026-06-05

### Added
- **Triggers and scheduling**: `Scheduler` with interval, cron, and DB watch. `WebhookTrigger` with REST endpoints for fire/enable/disable/remove/history.
- Dashboard trigger mode selection in agent creation flow.

### Fixed
- Dashboard: merged agent and session creation into one flow.
- Dashboard: wired Agents tab to REST API for persistence across refreshes.
- Dashboard: fixed live agent cards to use same renderer as mock agents.
- Dashboard: added policies to creation flow, triggers to edit modal.

## [0.85.0] — 2026-06-04

### Added
- **Schema auto-discovery**: `SqlConnector.discover_schema()` introspects tables, columns, types, row counts. `schema_to_text()` for human-readable format. `MongoConnector.discover_schema()` samples documents.
- REST endpoint: `GET /pipelines/schema`.
- `PipelineBuilder` accepts `db_schema` so the LLM writes correct SQL.

## [0.84.0] — 2026-06-03

### Added
- Dashboard pipeline editor: natural language "Describe" mode + manual "Edit POML" mode.
- REST endpoints: `POST /pipelines/generate`, `POST /pipelines/validate`, `POST /pipelines/run`, `GET /pipelines/tools`.

## [0.83.0] — 2026-06-02

### Added
- **PipelineBuilder**: Natural language → POML → Pipeline via any LLM. `create()`, `create_poml()`, `refine()`.
- POML pipeline format: `<step>`, `<threshold>`, `<condition>` tags with XML parsing.

## [0.82.0] — 2026-06-01

### Added
- **Database connectors**: `SqlConnector` (SQLite/PostgreSQL/MySQL) and `MongoConnector`. Read-only by default. Parameterized queries, configurable `max_rows`.
- **Conditional routing**: `Pipeline.add_condition()` and `Pipeline.add_threshold()` for branching pipelines.

## [0.81.0] — 2026-05-31

### Added
- **Pipeline orchestration**: `Pipeline` class with `add_step()`, sequential execution, output chaining. Every step goes through full supervision.

## [0.80.0] — 2026-05-30

### Added
- OAuth2 support (GitHub, Google) via `register_oauth_routes()`.
- International PII patterns: UK/EU phone numbers, UK National Insurance, German Tax IDs, IBANs, passport numbers.
- Optional NLP augmentation for PII detection via spaCy NER.

### Fixed
- Race conditions in `SessionManager`: per-session locking for concurrent lifecycle operations.
- Policy integrity: HMAC-SHA256 checksums prevent post-start policy tampering.
- Async timeout test flakiness (10ms→50ms thresholds).

## [0.79.0] — 2026-05-29

### Added
- **Privacy shielding**: `Redactor` with 9 regex categories (email, phone, SSN, credit card, API key, IP, national_id, IBAN, passport). Inbound/outbound/both directions.
- **Real-time dashboard**: WebSocket updates for session status changes.
- **Observability**: `Observer` with tool-level metrics, OpenTelemetry export.

## [0.78.0] — 2026-05-28

### Added
- **Human-in-the-loop approval**: `ApprovalQueue` with submit/approve/deny. `require_confirm` policy field. REST endpoints and WebSocket notifications. Dashboard approval cards.

### Changed
- npm package renamed to `agenthandler-ai` (name `agenthandler` was taken).

## [0.77.0] — 2026-05-27

### Added
- **Durable sessions**: `SessionManager` with `start()`, `pause()`, `resume()`, `stop()`, `restart()`. `SqliteStore` and `MemoryStore`.
- **REST control plane**: FastAPI app with 20+ endpoints for session CRUD, metrics, audit logs.
- **API key auth**: Bearer token authentication via `AGENTHANDLER_API_KEY`.
- **Stateless sessions**: `stateless=True` for in-memory-only supervision.
- **Crash loop protection**: `max_resumes` with automatic failure marking.
- Lifecycle example: `examples/session_lifecycle.py`.
- CI pipeline with pytest, mypy, ruff.
