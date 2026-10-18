"""FastAPI application factory.

Run locally:  uvicorn rootcite.api.app:create_app --factory --reload
"""

from __future__ import annotations

import logging
import math
import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from rootcite import __version__
from rootcite.api.demo import SCENARIOS, client_key
from rootcite.api.schemas import (
    AlertRequest,
    DiagnosisResponse,
    HealthResponse,
    IngestResponse,
    ScenarioOut,
)
from rootcite.config import Settings, Stage
from rootcite.container import Container, build_container
from rootcite.domain.errors import (
    DependencyUnavailableError,
    InvalidRequestError,
    LLMRequestError,
    QuotaExceededError,
    RetryableError,
)
from rootcite.observability.logging import configure_logging, request_id_var
from rootcite.resilience.rate_limit import SlidingWindowLimiter

log = logging.getLogger(__name__)


def _default_container() -> Container:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    return build_container(settings)


@lru_cache(maxsize=1)
def demo_page() -> str:
    return files("rootcite.api").joinpath("static/demo.html").read_text(encoding="utf-8")


def create_app(container: Container | None = None) -> FastAPI:
    # Bind a non-Optional name so the closures below are well-typed.
    c: Container = container if container is not None else _default_container()
    # Pre-prod convenience; prod ingests through the deployment pipeline instead.
    if c.settings.stage in (Stage.LOCAL, Stage.BETA) and c.index.count() == 0:
        c.ingestion.ingest_directory(c.settings.knowledge_dir)

    app = FastAPI(title="RootCite", version=__version__)
    app.state.container = c
    if c.settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(c.settings.allowed_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
            expose_headers=["retry-after", "x-request-id"],
            max_age=86_400,
        )
    rate = c.settings.rate_limit_per_minute
    limiter = SlidingWindowLimiter(rate, window_seconds=60) if rate > 0 else None

    @app.middleware("http")
    async def rate_limit(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Only the expensive calls (POST /v1/...) count toward the per-visitor limit.
        if limiter is not None and request.method == "POST" and request.url.path.startswith("/v1/"):
            peer = request.client.host if request.client else None
            allowed, retry_after = limiter.check(client_key(request.headers, peer))
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Too many requests. Please wait a minute and try again."},
                    headers={"retry-after": str(max(1, math.ceil(retry_after)))},
                )
        return await call_next(request)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(InvalidRequestError)
    async def invalid_request(_: Request, exc: InvalidRequestError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(DependencyUnavailableError)
    async def unavailable(_: Request, exc: DependencyUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": str(exc)},
            headers={"retry-after": str(int(exc.retry_after_seconds))},
        )

    @app.exception_handler(RetryableError)
    async def transient(_: Request, exc: RetryableError) -> JSONResponse:
        log.warning("dependency failed after retries", extra={"error": str(exc)})
        return JSONResponse(
            status_code=503,
            content={"error": "A dependency is temporarily unavailable. Try again shortly."},
            headers={"retry-after": "30"},
        )

    @app.exception_handler(QuotaExceededError)
    async def quota_exceeded(_: Request, exc: QuotaExceededError) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"error": str(exc), "reason": "daily_budget"},
            headers={"retry-after": str(max(1, math.ceil(exc.retry_after_seconds)))},
        )

    @app.exception_handler(LLMRequestError)
    async def llm_rejected(_: Request, exc: LLMRequestError) -> JSONResponse:
        log.error("model provider rejected the request", extra={"error": str(exc)})
        return JSONResponse(
            status_code=502, content={"error": "The model provider rejected the request."}
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo() -> HTMLResponse:
        return HTMLResponse(demo_page())

    @app.get("/v1/scenarios", response_model=list[ScenarioOut])
    def scenarios() -> list[ScenarioOut]:
        return [ScenarioOut(**s) for s in SCENARIOS]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        offline = c.settings.llm_provider == "offline"
        return HealthResponse(
            status="ok",
            stage=c.settings.stage.value,
            indexed_chunks=c.index.count(),
            version=__version__,
            llm_provider=c.settings.llm_provider,
            model="offline planner" if offline else c.settings.model,
            public_demo=c.settings.public_demo,
            repo_url=c.settings.repo_url,
            live_runs_remaining=c.run_budget.remaining() if c.run_budget else None,
        )

    @app.post("/v1/diagnoses", response_model=DiagnosisResponse)
    def create_diagnosis(body: AlertRequest) -> DiagnosisResponse:
        diagnosis = c.diagnosis.diagnose(body.to_domain())
        return DiagnosisResponse.from_domain(diagnosis)

    @app.post("/v1/ingest", response_model=IngestResponse)
    def ingest() -> IngestResponse:
        if c.settings.public_demo:
            raise HTTPException(status_code=403, detail="Ingestion is disabled on the public demo.")
        report = c.ingestion.ingest_directory(c.settings.knowledge_dir)
        return IngestResponse(
            documents=report.documents,
            chunks=report.chunks,
            indexed_total=c.index.count(),
        )

    return app
