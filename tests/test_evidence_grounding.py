"""Phase 1 Focused Tests: Evidence-Grounded Literature and Claim Extraction."""

import pytest
from backend.core.evidence_agent import (
    ClaimRecord,
    EvidenceBundle,
    EvidenceScope,
    LiteratureAgent,
    SourceRecord,
    VerificationStatus,
)
from backend.core.literature import PaperMetadata


class MockLiteratureService:
    """Mock literature service providing controlled PaperMetadata fixtures."""
    def __init__(self, papers):
        self.papers = papers

    async def search_literature(self, topic: str, limit: int = 5):
        return self.papers[:limit]


@pytest.mark.asyncio
async def test_real_evidence_claim_generation():
    """Test 1: Real Evidence Grounding from accessible abstract text."""
    real_abstract = (
        "We propose dynamic block-floating quantization for graph attention networks. "
        "Our method achieves 4.1x memory reduction while preserving node classification accuracy within 0.5% of dense FP32 baselines."
    )
    paper = PaperMetadata(
        doi="10.1109/TPAMI.2023.101",
        title="Dynamic Quantization for Graph Neural Networks",
        authors=["Smith, Jane", "Doe, John"],
        year=2023,
        venue="IEEE TPAMI",
        citation_count=120,
        abstract=real_abstract,
    )
    mock_service = MockLiteratureService([paper])
    agent = LiteratureAgent(lit_service=mock_service)
    
    evidence = await agent.gather_evidence("Dynamic Graph Attention Networks", limit=1)
    
    assert len(evidence.sources) == 1
    src = evidence.sources[0]
    assert src.retrieved_text_available is True
    assert src.evidence_scope == EvidenceScope.ABSTRACT
    assert len(src.claims) >= 1
    
    for claim in src.claims:
        assert claim.source_id == "src_001"
        assert claim.supporting_text != ""
        assert claim.supporting_text in real_abstract
        assert claim.supporting_location == "abstract"
        assert claim.evidence_scope == EvidenceScope.ABSTRACT
        assert claim.verification_status == VerificationStatus.VERIFIED
        assert claim.confidence is None  # No hardcoded fake confidence


@pytest.mark.asyncio
async def test_metadata_only_source_safety():
    """Test 2: Metadata-only source produces zero verified claims."""
    metadata_only_paper = PaperMetadata(
        doi="10.1145/3318464.3389700",
        title="Scalable Graph Processing under Strict Budgets",
        authors=["Brown, Alan"],
        year=2022,
        venue="ACM SIGMOD",
        citation_count=45,
        abstract=None,  # No accessible text
        full_text=None,
    )
    mock_service = MockLiteratureService([metadata_only_paper])
    agent = LiteratureAgent(lit_service=mock_service)
    
    evidence = await agent.gather_evidence("Scalable Graph Processing", limit=1)
    
    assert len(evidence.sources) == 1
    src = evidence.sources[0]
    assert src.retrieved_text_available is False
    assert src.evidence_scope == EvidenceScope.METADATA_ONLY
    assert src.retrieved_text is None
    # CRITICAL: Zero claims synthesized from metadata alone!
    assert len(src.claims) == 0
    assert len(evidence.claims) == 0


def test_missing_supporting_passage_rejection():
    """Test 3: Schema rejects verified claims with empty supporting passages."""
    with pytest.raises(ValueError, match="non-empty supporting_text"):
        ClaimRecord(
            claim_id="claim_test_001",
            claim_text="Graph transformers reduce peak memory.",
            source_id="src_001",
            supporting_text="",  # Empty supporting passage
            supporting_location="abstract",
            evidence_scope=EvidenceScope.ABSTRACT,
            verification_status=VerificationStatus.VERIFIED,
        )

    with pytest.raises(ValueError, match="evidence_scope is 'metadata_only'"):
        ClaimRecord(
            claim_id="claim_test_002",
            claim_text="Graph transformers reduce peak memory.",
            source_id="src_001",
            supporting_text="Some text passage",
            supporting_location="abstract",
            evidence_scope=EvidenceScope.METADATA_ONLY,  # Incompatible scope
            verification_status=VerificationStatus.VERIFIED,
        )


@pytest.mark.asyncio
async def test_claim_does_not_exceed_evidence():
    """Test 4: Extracted claim does not exaggerate or strengthen passage assertions."""
    weak_result_abstract = (
        "Method A improves F1 by approximately 2% on small benchmark datasets. "
        "However, performance degradation is observed on large graph topologies due to memory bandwidth bottlenecks."
    )
    paper = PaperMetadata(
        doi="10.1007/978-3-030-12345",
        title="Empirical Evaluation of Lightweight Graph Solvers",
        authors=["Lee, K."],
        year=2024,
        venue="ECML-PKDD",
        citation_count=10,
        abstract=weak_result_abstract,
    )
    mock_service = MockLiteratureService([paper])
    agent = LiteratureAgent(lit_service=mock_service)
    
    evidence = await agent.gather_evidence("Lightweight Graph Solvers", limit=1)
    assert len(evidence.claims) >= 1
    
    for claim in evidence.claims:
        # Verify claim strictly mirrors the passage and does NOT insert unsupported hyperbole
        assert "dramatically improves" not in claim.claim_text
        assert "revolutionary" not in claim.claim_text
        assert "flawless" not in claim.claim_text
        assert claim.supporting_text in weak_result_abstract


@pytest.mark.asyncio
async def test_no_synthetic_template_claims():
    """Test 5: Explicit test that title, venue, year, and topic never trigger synthetic claims without text."""
    empty_paper = PaperMetadata(
        doi="10.1109/ICCV.2023.9999",
        title="Universal Neural Quantum Operator for Continuous Dynamics",
        authors=["Einstein, Albert", "Bohr, Niels"],
        year=2024,
        venue="CVPR",
        citation_count=500,
        abstract="   ",  # Whitespace only
        full_text="",
    )
    mock_service = MockLiteratureService([empty_paper])
    agent = LiteratureAgent(lit_service=mock_service)
    
    evidence = await agent.gather_evidence("Quantum Operator", limit=1)
    assert len(evidence.claims) == 0
    assert evidence.sources[0].evidence_scope == EvidenceScope.METADATA_ONLY


def test_conflict_safety():
    """Test 6: Unrelated claims are NOT automatically marked as contradictory."""
    agent = LiteratureAgent()
    c1 = ClaimRecord(
        claim_id="c1",
        claim_text="Quantization reduces memory footprint.",
        source_id="s1",
        supporting_text="Quantization reduces memory footprint.",
        supporting_location="abstract",
        evidence_scope=EvidenceScope.ABSTRACT,
        polarity="supports",
        verification_status=VerificationStatus.VERIFIED,
        tags=["quantization", "memory"],
    )
    c2 = ClaimRecord(
        claim_id="c2",
        claim_text="Transformer attention scales quadratically with length.",
        source_id="s2",
        supporting_text="Transformer attention scales quadratically with length.",
        supporting_location="abstract",
        evidence_scope=EvidenceScope.ABSTRACT,
        polarity="supports",
        verification_status=VerificationStatus.VERIFIED,
        tags=["transformer", "attention"],
    )
    conflicts = agent._detect_conflicts([c1, c2])
    # Both claims support their respective domains; no artificial conflict
    assert len(conflicts) == 0


@pytest.mark.asyncio
async def test_search_failure_test_a_crossref_failure(monkeypatch):
    """Test A: When CrossRef fails, OpenAlex results are returned with proper provenance."""
    from backend.core.literature import LiteratureService
    service = LiteratureService()

    async def mock_crossref_fail(*args, **kwargs):
        raise ConnectionError("CrossRef API timeout")

    async def mock_openalex_success(*args, **kwargs):
        return [
            PaperMetadata(
                doi="10.1016/j.artint.2023.103999",
                title="Graph Neural Networks in Production",
                authors=["Zhang, Wei"],
                year=2023,
                venue="Artificial Intelligence",
                abstract="We demonstrate that quantized GNN operators reduce cache misses by 60% and improve throughput.",
                source_origin="openalex",
                text_origin="openalex_abstract_inverted_index",
            )
        ]

    monkeypatch.setattr(service, "query_crossref", mock_crossref_fail)
    monkeypatch.setattr(service, "query_openalex", mock_openalex_success)

    papers = await service.search_literature("Graph Neural Networks", limit=5)
    assert len(papers) == 1
    assert papers[0].source_origin == "openalex"
    assert papers[0].text_origin == "openalex_abstract_inverted_index"

    agent = LiteratureAgent(lit_service=service)
    evidence = await agent.gather_evidence("Graph Neural Networks", limit=5)
    assert len(evidence.sources) == 1
    assert evidence.sources[0].source_origin == "openalex"
    assert evidence.sources[0].text_origin == "openalex_abstract_inverted_index"
    assert len(evidence.claims) >= 1


@pytest.mark.asyncio
async def test_search_failure_test_b_openalex_failure(monkeypatch):
    """Test B: When OpenAlex fails, CrossRef results are returned with proper provenance."""
    from backend.core.literature import LiteratureService
    service = LiteratureService()

    async def mock_crossref_success(*args, **kwargs):
        return [
            PaperMetadata(
                doi="10.1109/TPAMI.2024.1111111",
                title="Low-Bit Quantization for Continuous Surrogates",
                authors=["Davis, Ronald"],
                year=2024,
                venue="IEEE TPAMI",
                abstract="We establish that 8-bit integer quantization bounds gradient variance and accelerates inference.",
                source_origin="crossref",
                text_origin="crossref_abstract",
            )
        ]

    async def mock_openalex_fail(*args, **kwargs):
        raise RuntimeError("OpenAlex service 503 unavailable")

    monkeypatch.setattr(service, "query_crossref", mock_crossref_success)
    monkeypatch.setattr(service, "query_openalex", mock_openalex_fail)

    papers = await service.search_literature("Continuous Surrogates", limit=5)
    assert len(papers) == 1
    assert papers[0].source_origin == "crossref"
    assert papers[0].text_origin == "crossref_abstract"

    agent = LiteratureAgent(lit_service=service)
    evidence = await agent.gather_evidence("Continuous Surrogates", limit=5)
    assert len(evidence.sources) == 1
    assert evidence.sources[0].source_origin == "crossref"
    assert evidence.sources[0].text_origin == "crossref_abstract"
    assert len(evidence.claims) >= 1


@pytest.mark.asyncio
async def test_search_failure_test_c_both_apis_fail(monkeypatch):
    """Test C: When both CrossRef and OpenAlex fail, return strictly empty results with ZERO fabrication."""
    from backend.core.literature import LiteratureService
    service = LiteratureService()

    async def mock_crossref_fail(*args, **kwargs):
        raise ConnectionError("Network unreachable")

    async def mock_openalex_fail(*args, **kwargs):
        raise TimeoutError("OpenAlex timeout")

    monkeypatch.setattr(service, "query_crossref", mock_crossref_fail)
    monkeypatch.setattr(service, "query_openalex", mock_openalex_fail)

    papers = await service.search_literature("Arbitrary Research Topic", limit=5)
    # ZERO fabricated papers
    assert papers == []

    agent = LiteratureAgent(lit_service=service)
    evidence = await agent.gather_evidence("Arbitrary Research Topic", limit=5)

    # ZERO fabricated sources or claims
    assert evidence.sources == []
    assert evidence.claims == []
    assert evidence.total_sources_retrieved == 0
    assert evidence.conflicting_claims == []
    for s in evidence.sources:
        assert s.source_origin != "test_fixture"


@pytest.mark.asyncio
async def test_search_failure_test_d_metadata_only_paper():
    """Test D: Metadata-only paper with text_origin='none' produces ZERO claims."""
    paper = PaperMetadata(
        doi="10.1000/182",
        title="Metadata Only Paper Without Accessible Abstract",
        authors=["Author, A."],
        year=2024,
        venue="General Journal",
        abstract=None,
        full_text=None,
        source_origin="crossref",
        text_origin="none",
    )
    mock_service = MockLiteratureService([paper])
    agent = LiteratureAgent(lit_service=mock_service)
    evidence = await agent.gather_evidence("Some Topic", limit=1)

    assert len(evidence.sources) == 1
    src = evidence.sources[0]
    assert src.evidence_scope == EvidenceScope.METADATA_ONLY
    assert src.text_origin == "none"
    assert src.retrieved_text_available is False
    assert len(src.claims) == 0
    assert len(evidence.claims) == 0


def test_search_failure_test_e_test_fixture_isolation():
    """Test E: Test fixtures only exist in tests/fixtures and are stamped with test_fixture provenance."""
    from tests.fixtures.literature.fixtures import get_canonical_test_papers, TEST_FIXTURE
    assert TEST_FIXTURE is True
    fixtures = get_canonical_test_papers()
    assert len(fixtures) > 0
    for paper in fixtures:
        assert paper.source_origin == "test_fixture"
        assert paper.text_origin == "test_fixture"


def test_production_safety_no_hardcoded_fallback_path():
    """Test F / Production Safety Test: Verify production code contains ZERO hardcoded fallback paper lists."""
    import inspect
    from backend.core import literature

    source_code = inspect.getsource(literature)
    assert "get_fallback_curated_papers" not in source_code
    assert "get_fallback_curated_papers" not in dir(literature.LiteratureService)
    # Check that search_literature does not have any fallback assignment
    assert "get_fallback" not in source_code

