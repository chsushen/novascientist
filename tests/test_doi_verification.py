"""Comprehensive Tests for Hardened DOI Verification, Normalization, Resolution, and Metadata Validation (Phase 2)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx

from backend.core.doi_verifier import (
    DOIVerificationResult,
    DOIVerificationStatus,
    DOIVerifier,
    calculate_verified_doi_rate,
    extract_metadata_from_response,
    is_title_match,
    is_year_match,
    normalize_doi,
    normalize_title_for_comparison,
    validate_doi_syntax,
)
from backend.core.evidence_agent import EvidenceScope, LiteratureAgent, SourceRecord
from backend.core.literature import PaperMetadata


# ---------------------------------------------------------------------------
# Test 1 — DOI Normalization Formats
# ---------------------------------------------------------------------------
def test_doi_normalization_all_formats():
    assert normalize_doi("https://doi.org/10.1109/TPAMI.2021.3099999") == "10.1109/TPAMI.2021.3099999"
    assert normalize_doi("http://doi.org/10.1109/TPAMI.2021.3099999") == "10.1109/TPAMI.2021.3099999"
    assert normalize_doi("https://dx.doi.org/10.1145/3534678.3539001") == "10.1145/3534678.3539001"
    assert normalize_doi("http://dx.doi.org/10.1145/3534678.3539001") == "10.1145/3534678.3539001"
    assert normalize_doi("doi:10.1016/j.jcp.2021.110660") == "10.1016/j.jcp.2021.110660"
    assert normalize_doi("DOI:10.1016/j.jcp.2021.110660") == "10.1016/j.jcp.2021.110660"
    assert normalize_doi("10.1038/s42254-021-00314-5") == "10.1038/s42254-021-00314-5"
    assert normalize_doi("  10.1038/s42254-021-00314-5.  ") == "10.1038/s42254-021-00314-5"
    assert normalize_doi("10.1038/s42254-021-00314-5;") == "10.1038/s42254-021-00314-5"
    assert normalize_doi("10.1038/s42254-021-00314-5,") == "10.1038/s42254-021-00314-5"
    assert normalize_doi("") is None
    assert normalize_doi(None) is None
    assert normalize_doi("not-a-doi") is None


# ---------------------------------------------------------------------------
# Test 2 — Invalid Syntax
# ---------------------------------------------------------------------------
def test_doi_invalid_syntax():
    assert validate_doi_syntax("9.1109/invalid") is False
    assert validate_doi_syntax("10/missingprefix") is False
    assert validate_doi_syntax("10.123") is False
    assert validate_doi_syntax("https://example.com/not-a-doi") is False
    assert validate_doi_syntax("") is False
    assert validate_doi_syntax(None) is False


# ---------------------------------------------------------------------------
# Test 3 — Valid Syntax but Unresolved DOI
# ---------------------------------------------------------------------------
def test_doi_valid_syntax_unresolved():
    assert validate_doi_syntax("10.1109/TPAMI.2021.3099999") is True
    paper = PaperMetadata(
        doi="10.1109/TPAMI.2021.3099999",
        title="Unresolved Paper",
        authors=["Author, A."],
        year=2022,
        venue="TPAMI",
        source_origin="openalex",
    )
    assert paper.doi_syntax_valid is True
    assert paper.doi_resolved is False
    assert paper.doi_verification_status == "syntax_valid_only"


# ---------------------------------------------------------------------------
# Test 4 — HTTP 404
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_404_unresolvable(monkeypatch):
    verifier = DOIVerifier()
    mock_resp = MagicMock(status_code=404, url="https://doi.org/10.1109/404", headers={})
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1109/404", expected_title="Missing Paper")
    assert result.doi_syntax_valid is True
    assert result.doi_resolved is False
    assert result.doi_verification_status == DOIVerificationStatus.UNRESOLVABLE
    assert result.http_status == 404
    assert "404" in (result.error_type or "")


# ---------------------------------------------------------------------------
# Test 5 — HTTP 410 Gone
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_410_gone_unresolvable(monkeypatch):
    verifier = DOIVerifier()
    mock_resp = MagicMock(status_code=410, url="https://doi.org/10.1109/410", headers={})
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1109/410")
    assert result.doi_resolved is False
    assert result.doi_verification_status == DOIVerificationStatus.UNRESOLVABLE
    assert result.http_status == 410
    assert result.error_type == "http_410_gone"


# ---------------------------------------------------------------------------
# Test 6 — HTTP 429 Rate Limited
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_429_rate_limited(monkeypatch):
    verifier = DOIVerifier()
    mock_resp = MagicMock(status_code=429, url="https://doi.org/10.1109/429", headers={})
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1109/429")
    assert result.doi_resolved is False
    assert result.doi_verification_status == DOIVerificationStatus.UNRESOLVABLE
    assert result.error_type == "http_429_rate_limited"


# ---------------------------------------------------------------------------
# Test 7 — HTTP 5xx Server Error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_5xx_server_error(monkeypatch):
    verifier = DOIVerifier()
    mock_resp = MagicMock(status_code=503, url="https://doi.org/10.1109/503", headers={})
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1109/503")
    assert result.doi_resolved is False
    assert result.doi_verification_status == DOIVerificationStatus.UNRESOLVABLE
    assert "server_error" in (result.error_type or "")


# ---------------------------------------------------------------------------
# Test 8 — Timeout
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_timeout(monkeypatch):
    verifier = DOIVerifier()
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1109/timeout")
    assert result.doi_resolved is False
    assert result.doi_verification_status == DOIVerificationStatus.UNRESOLVABLE
    assert result.error_type == "timeout"


# ---------------------------------------------------------------------------
# Test 9 — Connection / DNS Failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_connection_dns_failure(monkeypatch):
    verifier = DOIVerifier()
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.ConnectError("Failed to resolve host")
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1109/dnsfail")
    assert result.doi_resolved is False
    assert result.doi_verification_status == DOIVerificationStatus.UNRESOLVABLE
    assert "ConnectError" in (result.error_type or "")


# ---------------------------------------------------------------------------
# Test 10 — Real Redirect Behavior Using Mocked Redirect Chain
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_real_redirect_behavior(monkeypatch):
    verifier = DOIVerifier()

    redirect_resp = MagicMock(status_code=302, url="https://doi.org/10.1109/TPAMI.2021.123")
    final_resp = MagicMock(
        status_code=200,
        url="https://ieeexplore.ieee.org/document/1234567",
        history=[redirect_resp],
        headers={"content-type": "application/json"},
    )
    final_resp.json.return_value = {
        "title": "Adaptive Quantization for GNNs",
        "published-print": {"date-parts": [[2022, 5]]},
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = final_resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi(
        doi="10.1109/TPAMI.2021.123",
        expected_title="Adaptive Quantization for GNNs",
        expected_year=2022,
    )

    assert result.doi_resolved is True
    assert result.http_status == 200
    assert result.final_url == "https://ieeexplore.ieee.org/document/1234567"
    assert result.doi_verification_status == DOIVerificationStatus.VERIFIED


# ---------------------------------------------------------------------------
# Test 11 — Successful Resolution with Matching Title
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_successful_resolution_matching_title(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/doc/1",
        headers={"content-type": "application/json"},
    )
    resp.json.return_value = {"title": "Graph Convolutional Networks for Classification"}
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1000/1", expected_title="Graph Convolutional Networks for Classification")
    assert result.doi_resolved is True
    assert result.doi_metadata_match is True
    assert result.doi_verification_status == DOIVerificationStatus.VERIFIED


# ---------------------------------------------------------------------------
# Test 12 — Successful Resolution with Mismatching Title
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_successful_resolution_mismatching_title(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/doc/2",
        headers={"content-type": "application/json"},
    )
    resp.json.return_value = {"title": "Synthesizing Organic Polymers in Petrochemistry"}
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1000/2", expected_title="Machine Learning for Robotics")
    assert result.doi_resolved is True
    assert result.doi_metadata_match is False
    assert result.doi_verification_status == DOIVerificationStatus.METADATA_MISMATCH


# ---------------------------------------------------------------------------
# Test 13 — Successful Resolution with Missing Title Metadata (Case C)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_successful_resolution_missing_title_metadata(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/landing.html",
        headers={"content-type": "text/html; charset=utf-8"},
    )
    resp.json.side_effect = ValueError("No JSON in HTML")
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1000/3", expected_title="Some Expected Title")
    assert result.doi_resolved is True
    # CRITICAL: When metadata was expected but unavailable, doi_metadata_match must be False and status cannot be VERIFIED!
    assert result.doi_metadata_match is False
    assert result.doi_verification_status == DOIVerificationStatus.RESOLVED_METADATA_UNAVAILABLE


# ---------------------------------------------------------------------------
# Test 14 — Matching Expected Year
# ---------------------------------------------------------------------------
def test_is_year_match_rules():
    assert is_year_match(2023, 2023) is True
    assert is_year_match(2023, 2022) is True  # Within 1 year tolerance
    assert is_year_match(2022, 2023) is True  # Within 1 year tolerance
    assert is_year_match(2024, 2020) is False  # Diff > 1 year
    assert is_year_match(None, 2023) is True  # No constraint
    assert is_year_match(2023, None) is False  # Missing year when expected


# ---------------------------------------------------------------------------
# Test 15 — Mismatching Expected Year
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_mismatching_expected_year(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/doc/4",
        headers={"content-type": "application/json"},
    )
    resp.json.return_value = {
        "title": "Deep Graph Learning",
        "published-online": {"date-parts": [[2016, 1]]},
    }
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1000/4", expected_title="Deep Graph Learning", expected_year=2024)
    assert result.doi_resolved is True
    assert result.doi_metadata_match is False
    assert result.doi_verification_status == DOIVerificationStatus.METADATA_MISMATCH


# ---------------------------------------------------------------------------
# Test 16 — Missing Resolved Year When Expected Year is Supplied
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_missing_resolved_year_when_expected_year_supplied(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/doc/5",
        headers={"content-type": "application/json"},
    )
    # Title is present, but year fields are absent
    resp.json.return_value = {"title": "Graph Convolutional Networks"}
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1000/5", expected_title="Graph Convolutional Networks", expected_year=2023)
    assert result.doi_resolved is True
    assert result.doi_metadata_match is False
    assert result.doi_verification_status == DOIVerificationStatus.RESOLVED_METADATA_UNAVAILABLE


# ---------------------------------------------------------------------------
# Test 17 — Title + Year Both Matching
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_title_and_year_both_matching(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/doc/6",
        headers={"content-type": "application/json"},
    )
    resp.json.return_value = {
        "title": "Diffusion Convolutional Recurrent Neural Network",
        "issued": {"date-parts": [[2018, 4]]},
    }
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi(
        doi="10.1000/6",
        expected_title="Diffusion Convolutional Recurrent Neural Network: Traffic Forecasting",
        expected_year=2018,
    )
    assert result.doi_resolved is True
    assert result.doi_metadata_match is True
    assert result.doi_verification_status == DOIVerificationStatus.VERIFIED
    assert result.resolved_year == 2018


# ---------------------------------------------------------------------------
# Test 18 — Title Matching But Year Mismatching
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_title_matching_but_year_mismatching(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/doc/7",
        headers={"content-type": "application/json"},
    )
    resp.json.return_value = {
        "title": "Quantum Tensor Networks",
        "issued": {"date-parts": [[2012, 1]]},
    }
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi(
        doi="10.1000/7",
        expected_title="Quantum Tensor Networks",
        expected_year=2024,
    )
    assert result.doi_resolved is True
    assert result.doi_metadata_match is False
    assert result.doi_verification_status == DOIVerificationStatus.METADATA_MISMATCH


# ---------------------------------------------------------------------------
# Test 19 — Empty DOI Population
# ---------------------------------------------------------------------------
def test_calculate_verified_doi_rate_empty_population():
    assert calculate_verified_doi_rate([]) is None
    sources_no_doi = [
        SourceRecord(source_id="s1", title="No DOI", authors=["A"], year=2020, doi="", url="", venue="V")
    ]
    assert calculate_verified_doi_rate(sources_no_doi) is None


# ---------------------------------------------------------------------------
# Test 20 — Mixed Verified / Unverified DOI Rate
# ---------------------------------------------------------------------------
def test_calculate_verified_doi_rate_mixed_sources():
    sources = [
        SourceRecord(source_id="s1", title="P1", authors=["A"], year=2021, doi="10.1000/1", url="", venue="V", doi_verification_status="verified"),
        SourceRecord(source_id="s2", title="P2", authors=["A"], year=2022, doi="10.1000/2", url="", venue="V", doi_verification_status="verified"),
        SourceRecord(source_id="s3", title="P3", authors=["A"], year=2023, doi="10.1000/3", url="", venue="V", doi_verification_status="syntax_valid_only"),
        SourceRecord(source_id="s4", title="P4", authors=["A"], year=2024, doi="10.1000/4", url="", venue="V", doi_verification_status="resolved_metadata_unavailable"),
        SourceRecord(source_id="s5", title="P5", authors=["A"], year=2025, doi="10.1000/5", url="", venue="V", doi_verification_status="unresolvable"),
    ]
    rate = calculate_verified_doi_rate(sources)
    # Exactly 2 verified out of 5 = 0.4
    assert rate == 0.4


# ---------------------------------------------------------------------------
# Test 21 — Cache Isolation When Expected Title Differs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_cache_isolation_differing_expected_title(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/doc/cache",
        headers={"content-type": "application/json"},
    )
    resp.json.return_value = {"title": "Title Alpha"}
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    r1 = await verifier.verify_doi("10.1000/cache", expected_title="Title Alpha")
    r2 = await verifier.verify_doi("10.1000/cache", expected_title="Title Completely Different")

    assert r1.doi_verification_status == DOIVerificationStatus.VERIFIED
    assert r2.doi_verification_status == DOIVerificationStatus.METADATA_MISMATCH


# ---------------------------------------------------------------------------
# Test 22 — Cache Isolation When Expected Year Differs
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_cache_isolation_differing_expected_year(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(
        status_code=200,
        url="https://publisher.org/doc/cacheyear",
        headers={"content-type": "application/json"},
    )
    resp.json.return_value = {"title": "Title Shared", "issued": {"date-parts": [[2020]]}}
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    r1 = await verifier.verify_doi("10.1000/cacheyear", expected_title="Title Shared", expected_year=2020)
    r2 = await verifier.verify_doi("10.1000/cacheyear", expected_title="Title Shared", expected_year=2030)

    assert r1.doi_verification_status == DOIVerificationStatus.VERIFIED
    assert r2.doi_verification_status == DOIVerificationStatus.METADATA_MISMATCH


# ---------------------------------------------------------------------------
# Test 23 — Production Literature Pipeline Propagates DOI Status
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_production_literature_pipeline_propagates_doi_status(monkeypatch):
    from backend.core.literature import LiteratureService
    service = LiteratureService()

    async def mock_search(topic, limit=5):
        return [
            PaperMetadata(
                doi="10.1109/TPAMI.2023.999",
                title="Dynamic Quantization for Graph Neural Networks",
                authors=["Smith, J."],
                year=2023,
                venue="IEEE TPAMI",
                abstract="We investigate dynamic quantization for graph neural networks.",
                source_origin="openalex",
                text_origin="openalex_abstract_inverted_index",
            )
        ]

    monkeypatch.setattr(service, "search_literature", mock_search)

    # Mock DOI verifier
    mock_verifier = DOIVerifier()
    async def mock_verify(doi, expected_title=None, expected_year=None):
        return DOIVerificationResult(
            doi=doi,
            doi_normalized="10.1109/TPAMI.2023.999",
            doi_syntax_valid=True,
            doi_resolved=True,
            doi_metadata_match=True,
            doi_verification_status=DOIVerificationStatus.VERIFIED,
            http_status=200,
            final_url="https://ieeexplore.ieee.org/document/999",
            resolved_title="Dynamic Quantization for Graph Neural Networks",
            resolved_year=2023,
        )
    monkeypatch.setattr(mock_verifier, "verify_doi", mock_verify)

    agent = LiteratureAgent(lit_service=service, doi_verifier=mock_verifier)
    evidence = await agent.gather_evidence("Dynamic Quantization", limit=1)

    assert len(evidence.sources) == 1
    src = evidence.sources[0]
    assert src.doi_syntax_valid is True
    assert src.doi_resolved is True
    assert src.doi_metadata_match is True
    assert src.doi_verification_status == "verified"
    assert src.doi_final_url == "https://ieeexplore.ieee.org/document/999"
    assert src.doi_http_status == 200
    assert evidence.verified_doi_rate == 1.0


# ---------------------------------------------------------------------------
# Test 24 — DOI Resolution Failure Does Not Fabricate Metadata
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_doi_resolution_failure_does_not_fabricate_metadata(monkeypatch):
    verifier = DOIVerifier()
    mock_client = AsyncMock()
    mock_client.get.side_effect = httpx.TimeoutException("Timeout")
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1000/timeout_fail", expected_title="Expected Title", expected_year=2023)
    assert result.doi_resolved is False
    assert result.resolved_title is None
    assert result.resolved_year is None
    assert result.doi_metadata_match is False
    assert result.doi_verification_status == DOIVerificationStatus.UNRESOLVABLE


# ---------------------------------------------------------------------------
# Negative Integrity Invariant Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_negative_integrity_http_200_missing_metadata_not_verified(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(status_code=200, url="https://publisher.org/page", headers={"content-type": "text/html"})
    resp.json.side_effect = ValueError("No JSON")
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1000/no_meta", expected_title="My Paper", expected_year=2024)
    # INVARIANT: HTTP 200 + missing metadata != VERIFIED
    assert result.doi_verification_status != DOIVerificationStatus.VERIFIED
    assert result.doi_verification_status == DOIVerificationStatus.RESOLVED_METADATA_UNAVAILABLE


@pytest.mark.asyncio
async def test_negative_integrity_http_200_title_matches_year_mismatches_not_verified(monkeypatch):
    verifier = DOIVerifier()
    resp = MagicMock(status_code=200, url="https://publisher.org/doc", headers={"content-type": "application/json"})
    resp.json.return_value = {"title": "Adaptive GNN", "issued": {"date-parts": [[2010]]}}
    mock_client = AsyncMock()
    mock_client.get.return_value = resp
    monkeypatch.setattr(httpx, "AsyncClient", MagicMock(return_value=mock_client))
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None

    result = await verifier.verify_doi("10.1000/year_mismatch", expected_title="Adaptive GNN", expected_year=2024)
    # INVARIANT: Title matches + Year mismatches != VERIFIED
    assert result.doi_verification_status != DOIVerificationStatus.VERIFIED
    assert result.doi_verification_status == DOIVerificationStatus.METADATA_MISMATCH
