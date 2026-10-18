# Live demo

The demo is the same application as the production design, configured for free public
hosting:

| | Production design | Public demo |
|---|---|---|
| Hosting | ECS Fargate behind an ALB (AWS CDK) | Render web service (Docker, free plan) |
| Model | Claude on Amazon Bedrock | `openai/gpt-oss-20b` on Groq's free tier |
| Index | PostgreSQL + pgvector | In-memory (rebuilt from `data/knowledge` at startup) |
| Telemetry | CloudWatch (planned adapter) | Sample fixtures for five fictional services |
| Deploys | CDK | GitHub Actions, after CI passes |

Only adapters and settings differ. The agent, tools, guardrails, and verifier are identical.

## What visitors see

`GET /` serves a single page: choose one of five sample incidents (or edit the alert), run
the diagnosis, and see the root cause, confidence, recommended actions, guardrail warnings,
blocked actions, the tools the agent called, and every piece of evidence with the cited ones
highlighted. All API output is rendered as text, never as HTML.

## Safeguards

`ROOTCITE_PUBLIC_DEMO=true` turns on:

- **A per-visitor rate limit** (10 diagnoses a minute by default, `ROOTCITE_RATE_LIMIT_PER_MINUTE`),
  answered with HTTP 429 and `Retry-After`.
- **A daily budget of live runs** (100 by default, `ROOTCITE_DAILY_RUN_LIMIT`), which keeps
  the demo within the model provider's free quota. Answers served from the semantic cache
  don't count, and `render.yaml` keeps cached answers for 24 hours, so repeated runs of the
  sample incidents cost nothing.
- **No writes:** `POST /v1/ingest` returns 403.
- The usual limits still apply: input validation, step, tool-call, and token budgets, and
  blocked side-effect tools.

If the model is unavailable, the API answers 503 with `Retry-After` instead of failing
silently. With no key at all, the demo runs the deterministic offline planner.

## Hosting the UI on a static site

The API also accepts requests from the origins listed in `ROOTCITE_ALLOWED_ORIGINS`, so the
demo UI can be served from a static site that never sleeps. Such a page can wake the API in
the background (a `GET /health` request uses no model tokens), show a recorded example while
the service starts, and enable live runs once `/health` responds.

## Deployment (one-time setup)

The service is defined in `render.yaml` (Docker runtime, free plan, health check on
`/health`, demo settings). Render's free plan needs no credit card.

1. Sign in at https://render.com with GitHub.
2. Choose **New → Blueprint**, select this repository, and apply `render.yaml`. When asked
   for `ROOTCITE_LLM_API_KEY`, enter a Groq API key.
3. In the new service, open **Settings → Deploy Hook** and copy the URL.
4. In the GitHub repository, **Settings → Secrets and variables → Actions**, add the secret
   `RENDER_DEPLOY_HOOK_URL` with that URL.

From then on, every push to `main` that passes CI triggers the **Deploy demo** workflow,
which calls the deploy hook. Render builds the image and replaces the running service once
the health check passes. The workflow can also be run manually from the Actions tab.

Free services sleep after 15 minutes without traffic. The first request after that takes
about a minute while the service starts.
