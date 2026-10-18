.PHONY: install lint typecheck test eval run docker-up mcp clean

install:            ## Install with dev extras
	pip install -e ".[dev]"

lint:               ## Ruff lint
	ruff check src tests infra

typecheck:          ## mypy on the application package
	mypy

test:               ## Unit + integration tests with coverage
	pytest --cov --cov-report=term-missing

eval:               ## Offline evals (retrieval is a CI quality gate)
	rootcite eval retrieval --min-hit-rate 0.9 --out reports/retrieval.json
	rootcite eval agent --out reports/agent.json

run:                ## API on http://localhost:8000/docs
	uvicorn rootcite.api.app:create_app --factory --reload

docker-up:          ## API + Postgres/pgvector
	docker compose up --build

mcp:                ## MCP server over stdio (for Claude Desktop / Claude Code)
	python -m rootcite.mcp_server

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage reports build *.egg-info
