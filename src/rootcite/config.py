"""Stage-aware settings (local -> beta -> gamma -> prod), loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rootcite.domain.errors import ConfigurationError


DEFAULT_OPENAI_MODEL = "openai/gpt-oss-20b"  # on Groq's free tier, with tool calling


class Stage(StrEnum):
    LOCAL = "local"
    BETA = "beta"
    GAMMA = "gamma"
    PROD = "prod"


@dataclass(frozen=True)
class Settings:
    stage: Stage = Stage.LOCAL
    llm_provider: str = "offline"  # offline | anthropic | bedrock | openai
    model: str = "claude-sonnet-5"
    # For llm_provider=openai: any OpenAI-compatible API (Groq's free tier by default).
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = field(default="", repr=False)  # never printed in logs or errors
    aws_region: str = "us-east-1"
    embedder: str = "hashing"  # hashing | sentence-transformers
    index: str = "memory"  # memory | postgres
    database_url: str = ""
    knowledge_dir: Path = Path("data/knowledge")
    telemetry_dir: Path = Path("data/telemetry")
    retrieval_k: int = 5
    max_agent_steps: int = 6
    max_tool_calls: int = 10
    token_budget: int = 60_000
    cache_similarity: float = 0.95
    cache_ttl_seconds: int = 900
    log_level: str = "INFO"
    # Public demo mode: rate-limits the API per visitor and disables write endpoints.
    public_demo: bool = False
    rate_limit_per_minute: int = 0  # 0 = off; public demo defaults to 10
    repo_url: str = ""  # shown on the demo page
    allowed_origins: tuple[str, ...] = ()  # browser origins allowed to call the API (CORS)
    daily_run_limit: int = 0  # live agent runs per UTC day; 0 = unlimited, demo default 100

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        e = dict(os.environ) if env is None else env

        def get(name: str, default: str) -> str:
            return e.get(f"ROOTCITE_{name}", default)

        api_key = get("LLM_API_KEY", "")
        # With a key and no explicit provider, use the OpenAI-compatible API.
        provider = get("LLM_PROVIDER", "openai" if api_key else "offline")
        default_model = DEFAULT_OPENAI_MODEL if provider == "openai" else cls.model
        public_demo = get("PUBLIC_DEMO", "false").lower() in {"1", "true", "yes"}
        default_rate = "10" if public_demo else "0"
        settings = cls(
            stage=Stage(get("STAGE", "local")),
            llm_provider=provider,
            llm_base_url=get("LLM_BASE_URL", cls.llm_base_url),
            llm_api_key=api_key,
            model=get("MODEL", default_model),
            aws_region=e.get("AWS_REGION", cls.aws_region),
            embedder=get("EMBEDDER", "hashing"),
            index=get("INDEX", "memory"),
            database_url=e.get("DATABASE_URL", "") or _dsn_from_parts(e),
            knowledge_dir=Path(get("KNOWLEDGE_DIR", str(cls.knowledge_dir))),
            telemetry_dir=Path(get("TELEMETRY_DIR", str(cls.telemetry_dir))),
            retrieval_k=int(get("RETRIEVAL_K", str(cls.retrieval_k))),
            max_agent_steps=int(get("MAX_AGENT_STEPS", str(cls.max_agent_steps))),
            max_tool_calls=int(get("MAX_TOOL_CALLS", str(cls.max_tool_calls))),
            token_budget=int(get("TOKEN_BUDGET", str(cls.token_budget))),
            cache_similarity=float(get("CACHE_SIMILARITY", str(cls.cache_similarity))),
            cache_ttl_seconds=int(get("CACHE_TTL_SECONDS", str(cls.cache_ttl_seconds))),
            log_level=get("LOG_LEVEL", cls.log_level),
            public_demo=public_demo,
            repo_url=get("REPO_URL", ""),
            allowed_origins=tuple(
                o.strip().rstrip("/") for o in get("ALLOWED_ORIGINS", "").split(",") if o.strip()
            ),
            daily_run_limit=int(get("DAILY_RUN_LIMIT", "100" if public_demo else "0")),
            rate_limit_per_minute=int(get("RATE_LIMIT_PER_MINUTE", default_rate)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.llm_provider not in {"offline", "anthropic", "bedrock", "openai"}:
            raise ConfigurationError(f"unknown llm_provider: {self.llm_provider}")
        if self.llm_provider == "openai" and not self.llm_api_key:
            raise ConfigurationError("llm_provider=openai requires ROOTCITE_LLM_API_KEY")
        if self.daily_run_limit < 0:
            raise ConfigurationError("daily_run_limit must be >= 0")
        if self.rate_limit_per_minute < 0:
            raise ConfigurationError("rate_limit_per_minute must be >= 0")
        if self.index not in {"memory", "postgres"}:
            raise ConfigurationError(f"unknown index: {self.index}")
        if self.index == "postgres" and not self.database_url:
            raise ConfigurationError("index=postgres requires DATABASE_URL")
        if self.stage in {Stage.GAMMA, Stage.PROD}:
            if self.llm_provider == "offline":
                raise ConfigurationError(f"stage {self.stage} cannot use the offline LLM")
            if self.index == "memory":
                raise ConfigurationError(f"stage {self.stage} needs a durable index (postgres)")
        if not 0 < self.cache_similarity <= 1:
            raise ConfigurationError("cache_similarity must be in (0, 1]")


def _dsn_from_parts(env: dict[str, str]) -> str:
    """Build a DSN from DB_HOST/DB_USER/DB_PASSWORD (how ECS injects Secrets Manager values)."""
    host = env.get("DB_HOST")
    if not host:
        return ""
    user = env.get("DB_USER", "rootcite")
    password = env.get("DB_PASSWORD", "")
    name = env.get("DB_NAME", "rootcite")
    port = env.get("DB_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"
