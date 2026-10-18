"""Composition root: the only place that chooses concrete adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO

from rootcite.adapters.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from rootcite.adapters.memory_index import InMemoryIndex
from rootcite.adapters.offline_llm import OfflineLLM
from rootcite.adapters.telemetry import FixtureTelemetry
from rootcite.agent.graph import AgentConfig, IncidentAgent
from rootcite.agent.toolkit import build_default_tools
from rootcite.agent.tools import ToolRegistry
from rootcite.config import Settings
from rootcite.observability.metrics import MetricsLogger
from rootcite.ports import Embedder, LLMClient, SearchIndex, TelemetrySource
from rootcite.resilience.circuit_breaker import CircuitBreaker
from rootcite.resilience.resilient_llm import ResilientLLM
from rootcite.resilience.semantic_cache import SemanticCache
from rootcite.retrieval.hybrid import HybridRetriever
from rootcite.services.diagnosis import DiagnosisService
from rootcite.services.ingestion import IngestionService


@dataclass
class Container:
    settings: Settings
    embedder: Embedder
    index: SearchIndex
    retriever: HybridRetriever
    telemetry: TelemetrySource
    tools: ToolRegistry
    llm: LLMClient
    agent: IncidentAgent
    diagnosis: DiagnosisService
    ingestion: IngestionService


def _embedder(settings: Settings) -> Embedder:
    if settings.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder()
    return HashingEmbedder()


def _index(settings: Settings, dimension: int) -> SearchIndex:
    if settings.index == "postgres":
        from rootcite.adapters.pgvector_index import PgVectorIndex

        return PgVectorIndex(settings.database_url, dimension)
    return InMemoryIndex(dimension)


def _llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "offline":
        return OfflineLLM()
    if settings.llm_provider == "openai":
        from rootcite.adapters.openai_compat import OpenAICompatLLM

        compat = OpenAICompatLLM(
            settings.model, settings.llm_api_key, base_url=settings.llm_base_url
        )
        return ResilientLLM(compat, CircuitBreaker("llm", failure_threshold=5, reset_timeout=30))
    from rootcite.adapters.claude import ClaudeLLM

    inner = ClaudeLLM(
        settings.model, provider=settings.llm_provider, aws_region=settings.aws_region
    )
    return ResilientLLM(inner, CircuitBreaker("claude", failure_threshold=5, reset_timeout=30))


def build_container(
    settings: Settings,
    *,
    llm: LLMClient | None = None,
    telemetry: TelemetrySource | None = None,
    metrics_stream: TextIO | None = None,
) -> Container:
    """Wire the application. Tests and the CLI inject fakes via the keyword arguments."""
    embedder = _embedder(settings)
    index = _index(settings, embedder.dimension)
    retriever = HybridRetriever(index, embedder)
    telemetry = telemetry or FixtureTelemetry(settings.telemetry_dir)
    tools = build_default_tools(telemetry, retriever)
    llm = llm or _llm(settings)
    agent = IncidentAgent(
        llm,
        retriever,
        tools,
        AgentConfig(
            retrieval_k=settings.retrieval_k,
            max_steps=settings.max_agent_steps,
            max_tool_calls=settings.max_tool_calls,
            token_budget=settings.token_budget,
        ),
    )
    cache = SemanticCache(
        embedder, threshold=settings.cache_similarity, ttl_seconds=settings.cache_ttl_seconds
    )
    return Container(
        settings=settings,
        embedder=embedder,
        index=index,
        retriever=retriever,
        telemetry=telemetry,
        tools=tools,
        llm=llm,
        agent=agent,
        diagnosis=DiagnosisService(
            agent,
            cache,
            metrics_factory=lambda operation: MetricsLogger(
                operation, stream=metrics_stream, dimensions={"Stage": settings.stage.value}
            ),
        ),
        ingestion=IngestionService(index, embedder),
    )
