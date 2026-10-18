from __future__ import annotations

import io
from collections.abc import Callable

import pytest

from rootcite.config import Settings
from rootcite.container import Container, build_container
from rootcite.ports import LLMClient
from support import KNOWLEDGE_DIR, TELEMETRY_DIR


@pytest.fixture
def settings() -> Settings:
    return Settings(knowledge_dir=KNOWLEDGE_DIR, telemetry_dir=TELEMETRY_DIR)


@pytest.fixture
def make_container(settings: Settings) -> Callable[..., Container]:
    def factory(llm: LLMClient | None = None) -> Container:
        container = build_container(settings, llm=llm, metrics_stream=io.StringIO())
        container.ingestion.ingest_directory(settings.knowledge_dir)
        return container

    return factory
