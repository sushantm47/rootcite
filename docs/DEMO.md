# Live demo

The demo is the same application as the production design, configured for free public
hosting:

| | Production design | Public demo |
|---|---|---|
| Hosting | ECS Fargate behind an ALB (AWS CDK) | Hugging Face Space (Docker, free CPU) |
| Model | Claude on Amazon Bedrock | `openai/gpt-oss-20b` on Groq's free tier |
| Index | PostgreSQL + pgvector | In-memory (rebuilt from `data/knowledge` at startup) |
| Telemetry | CloudWatch (planned adapter) | Sample fixtures for five fictional services |
| Deploys | CDK | GitHub Actions on every push |

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
- **No writes:** `POST /v1/ingest` returns 403.
- The usual limits still apply: input validation, step, tool-call, and token budgets, and
  blocked side-effect tools.

If the model is unavailable, the API answers 503 with `Retry-After` instead of failing
silently. With no key at all, the demo runs the deterministic offline planner.

## Deployment (one-time setup)

1. Create a free account at https://huggingface.co and a new **Space**: SDK **Docker**,
   template **Blank**, hardware **CPU basic (free)**.
2. In the Space's **Settings → Variables and secrets**, add the secret
   `ROOTCITE_LLM_API_KEY` with a Groq API key.
3. Create a Hugging Face access token with **write** permission (Settings → Access Tokens).
4. In the GitHub repo, **Settings → Secrets and variables → Actions**:
   - secret `HF_TOKEN`: the token from step 3
   - variable `HF_SPACE`: `<hf-username>/<space-name>`
5. Push to `main`, or run the **Deploy demo** workflow from the Actions tab.

The workflow copies `src/`, `data/`, `Dockerfile`, and `pyproject.toml` into the Space with
the Space's own README (its configuration card, in `deploy/huggingface/README.md`), adds the
demo settings, and pushes. Hugging Face builds the image and starts it on port 8000.

Free Spaces sleep after a period without visitors. The first request after that takes
about a minute while the Space wakes up.
