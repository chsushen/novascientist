"""Unit tests for LiteratureService and BibTeX generator."""

import pytest
import asyncio
from backend.core.literature import LiteratureService, PaperMetadata


def test_paper_metadata_bibkey_generation():
    paper = PaperMetadata(
        doi="10.1109/TPAMI.2021.3099999",
        title="Adaptive Quantization and Memory-Bounded Graph Neural Networks",
        authors=["Kipf, Thomas", "Welling, Max"],
        year=2022,
        venue="IEEE TPAMI",
        abstract="We investigate adaptive low-bit quantization for graph neural networks.",
        source_origin="openalex",
        text_origin="openalex_abstract_inverted_index",
    )
    assert "kipf2022" in paper.bibkey
    assert paper.url == "https://doi.org/10.1109/TPAMI.2021.3099999"
    assert paper.source_origin == "openalex"
    assert paper.text_origin == "openalex_abstract_inverted_index"

    # Verify that without accessible text, text_origin defaults to 'none'
    no_text_paper = PaperMetadata(
        doi="10.1109/TPAMI.2021.3099999",
        title="Adaptive Quantization",
        authors=["Kipf, Thomas"],
        year=2022,
        venue="IEEE TPAMI",
        source_origin="openalex",
    )
    assert no_text_paper.text_origin == "none"


def test_escape_latex():
    raw = "Deep Learning & AI: A 95% Survey of #1 Methods"
    escaped = LiteratureService.escape_latex(raw)
    assert r"\&" in escaped
    assert r"\%" in escaped
    assert r"\#" in escaped


def test_generate_bibtex():
    papers = [
        PaperMetadata(
            doi="10.1109/TC.2024.12345",
            title="Resource-Constrained Representation Learning",
            authors=["Dally, William", "Horowitz, Mark"],
            year=2024,
            venue="IEEE Transactions on Computers",
            bibkey="dally2024_resource",
            source_origin="crossref",
            text_origin="crossref_abstract",
        )
    ]
    service = LiteratureService()
    bibtex = service.generate_bibtex(papers)
    assert "@article{dally2024_resource," in bibtex
    assert "author    = {Dally, William and Horowitz, Mark}," in bibtex
    assert "doi       = {10.1109/TC.2024.12345}," in bibtex


@pytest.mark.asyncio
async def test_search_literature_safe_empty_on_failure(monkeypatch):
    """Verify that when external APIs fail or are unreachable, empty list is returned with ZERO fabrication."""
    service = LiteratureService()
    
    # Mock both queries to simulate network failure / 0 results
    async def mock_fail(*args, **kwargs):
        return []
    
    monkeypatch.setattr(service, "query_crossref", mock_fail)
    monkeypatch.setattr(service, "query_openalex", mock_fail)

    papers = await service.search_literature("Dynamic Graph Attention", limit=3)
    assert papers == []  # Must be strictly empty, never synthetic fallbacks
