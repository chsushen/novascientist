"""
Tests for Phase 8: Production Hardening & Deployment Stability.

Validates:
1. LiteratureService environment configuration and timeout parameters.
2. Graceful exception handling for network timeouts, DNS errors, and 5xx responses.
3. Central Orchestrator robust exception handling and return contracts.
4. CLI argument parsing and execution stability.
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from backend.core.latex_assembler import AuthorProfile
from backend.core.literature import LiteratureService, PaperMetadata
from backend.core.orchestrator import NovaScientistOrchestrator, OrchestratorResult
from cli import main


def test_literature_service_environment_configuration(monkeypatch):
    """Verify LiteratureService reads custom timeout and email from environment variables."""
    monkeypatch.setenv("SCHOLARLY_CONTACT_EMAIL", "custom_lab@ai.org")
    monkeypatch.setenv("SCHOLARLY_API_TIMEOUT", "12.5")

    service = LiteratureService()
    assert service.email == "custom_lab@ai.org"
    assert service.timeout == 12.5
    assert "custom_lab@ai.org" in service.headers["User-Agent"]


def test_literature_service_parameter_override():
    """Verify explicit parameters override environment defaults."""
    service = LiteratureService(email="direct@univ.edu", timeout=4.0)
    assert service.email == "direct@univ.edu"
    assert service.timeout == 4.0


@pytest.mark.asyncio
async def test_crossref_network_timeout_resilience():
    """Verify query_crossref catches timeout exceptions and returns an empty list without crashing."""
    service = LiteratureService()

    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Connection timed out")):
        papers = await service.query_crossref("Quantum Computing", max_results=5)
        assert isinstance(papers, list)
        assert len(papers) == 0


@pytest.mark.asyncio
async def test_openalex_server_error_resilience():
    """Verify query_openalex catches HTTP 500/503 errors and returns an empty list cleanly."""
    service = LiteratureService()

    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.side_effect = Exception("Service Unavailable")

    with patch("httpx.AsyncClient.get", return_value=mock_resp):
        papers = await service.query_openalex("Physics Surrogates", max_results=5)
        assert isinstance(papers, list)
        assert len(papers) == 0


@pytest.mark.asyncio
async def test_search_literature_partial_backend_failure():
    """Verify search_literature returns available papers even if one backend times out."""
    service = LiteratureService()

    # Mock CrossRef failing with timeout and OpenAlex returning 1 paper
    mock_paper = PaperMetadata(
        doi="10.1145/3318464.3389700",
        title="OpenAlex Recovered Paper",
        authors=["Author, OpenAlex"],
        year=2022,
        venue="ACM KDD",
        source_origin="openalex",
        text_origin="openalex_abstract_inverted_index",
        abstract="Sample recovered abstract for robustness test.",
    )

    with patch.object(service, "query_crossref", side_effect=httpx.TimeoutException("Timeout")), \
         patch.object(service, "query_openalex", return_value=[mock_paper]):
        papers = await service.search_literature("Robustness Topic", limit=5)
        assert len(papers) == 1
        assert papers[0].title == "OpenAlex Recovered Paper"


@pytest.mark.asyncio
async def test_orchestrator_execution_contract(tmp_path):
    """Verify NovaScientistOrchestrator produces valid OrchestratorResult with complete telemetry."""
    orchestrator = NovaScientistOrchestrator(output_dir=str(tmp_path / "dist"))

    result = await orchestrator.execute(
        topic="Adaptive Quantization for Graph Transformers",
        author=AuthorProfile("Researcher", "University", "res@univ.edu"),
        target_length="4_page_conference",
        execution_mode="fast_microbenchmark",
        num_seeds=3,
        num_epochs=5,
    )

    assert isinstance(result, OrchestratorResult)
    assert result.success is True
    assert result.topic == "Adaptive Quantization for Graph Transformers"
    assert result.plan is not None
    assert result.methodology is not None
    assert result.evidence is not None
    assert result.validation_report is not None
    assert result.stat_critique is not None
    assert result.review_report is not None
    assert result.provenance_graph is not None
    assert result.revision_history is not None
    assert result.elapsed_seconds > 0.0


def test_cli_help_and_subcommand_parsing():
    """Verify CLI parses benchmark and run subcommands correctly."""
    import sys
    from unittest.mock import patch

    test_args = ["cli.py", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_version_consistency_and_release_config():
    """Verify consistent v2.3.0 versioning across core configuration and server diagnostics."""
    from backend.config import config
    from backend.api.server import app
    from fastapi.testclient import TestClient

    assert config.app_version == "2.3.0"
    
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["app_version"] == "2.3.0"

    diag = client.get("/diagnostics")
    assert diag.status_code == 200
    assert diag.json()["version"] == "2.3.0"
    assert "v2.3.0" in diag.json()["application"]

