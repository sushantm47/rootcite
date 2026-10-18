# RootCite operational runbook

Owner: RootCite maintainers · Dashboards: CloudWatch namespace `RootCite` · Logs: JSON, search by `request_id`

## Alarm: P99LatencyAlarm (ALB p99 > 20 s for 5 minutes)

1. Check `Latency` and `ToolCalls` by `Stage`. A jump in tool calls means the model is
   looping; confirm `max_agent_steps` / `max_tool_calls` are set as expected.
2. Check model-side latency: Bedrock `InvocationLatency` and throttling metrics.
3. Mitigate: lower `ROOTCITE_MAX_AGENT_STEPS`, or scale out tasks if CPU is high.

## Alarm: Target5xxAlarm (5xx > 5/min for 3 minutes)

1. Search logs for `level=ERROR` and group by `logger`.
2. 503s reporting that `claude` or `llm` is unavailable mean the model's circuit breaker is
   open: check the provider's status and throttling quotas (Bedrock, or the
   OpenAI-compatible provider). The breaker probes again after 30 s. Other 503s with
   `Retry-After: 30` are transient model failures that exhausted their retries.
3. Database errors: check RDS connections and CPU; the pool max is 10 per task.
4. If it started with a deploy, roll back (ECS keeps the previous task definition).

## Alarm: Grounding regression (`Grounded` avg < 0.9 for 1 hour)

1. Check whether a prompt or model change shipped (`PROMPT_VERSION`, `ROOTCITE_MODEL`).
2. Check `ingest` ran: an empty or stale index produces uncited answers.
3. Run `rootcite eval agent` against the golden set and compare with the last report.

## Routine: re-index the knowledge base

`POST /v1/ingest` or `rootcite ingest`. Chunk IDs are derived from the source path and
position, so re-running upserts instead of duplicating.

## Public demo: visitors report "Rate limit reached"

The demo allows `ROOTCITE_RATE_LIMIT_PER_MINUTE` diagnoses per client per minute (default
10). Sustained 429s from many clients usually indicate scripted traffic; lower the limit in
the Space's variables. If the model provider's free quota is exhausted, diagnoses return 503
until it resets.
