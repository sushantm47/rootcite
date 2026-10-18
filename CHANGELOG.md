# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0]

### Added
- OpenAI-compatible model adapter with tool calling, supporting Groq, OpenAI, Together,
  vLLM, and Ollama without additional dependencies.
- Demo UI served at `/`, and a `GET /v1/scenarios` endpoint with sample incidents.
- Public demo mode: per-client sliding-window rate limiting and disabled write endpoints.
- `Deploy demo` workflow publishing the application to a Hugging Face Space.
- HTTP 503 with `Retry-After` for transient model failures, and 502 for rejected requests.
- Model and provider details in `GET /health`.

### Changed
- Renamed the project from OpsPilot to RootCite, including the Python package, command-line
  tool, `ROOTCITE_` settings prefix, and metrics namespace.
- An API key alone now selects the OpenAI-compatible provider and its default model.
- The AWS CDK stack takes the Bedrock model ID as required deploy-time context.

## [0.2.0]

### Added
- Incident agent implemented as a state machine with step, tool-call, and token budgets.
- Tool registry with argument validation, and side-effect tools held for human approval.
- Guardrails: prompt-injection detection, untrusted-data isolation, secret and PII redaction.
- Citation verifier with confidence capping and escalation rules.
- FastAPI service, MCP server, and command-line interface.
- Retrieval and agent evaluation harness with a CI quality gate.
- Docker image, Docker Compose environment, and AWS CDK stack.

## [0.1.0]

### Added
- Domain model and ports for embedders, search indexes, LLMs, and telemetry.
- Markdown chunking with contextual headers, BM25, and hybrid retrieval with Reciprocal
  Rank Fusion.
- In-memory and PostgreSQL/pgvector indexes.
- Retries with full jitter, circuit breaker, and semantic cache.
- Structured logging and CloudWatch Embedded Metric Format metrics.
- Stage-aware configuration.
