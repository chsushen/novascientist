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
    )
    assert "kipf2022" in paper.bibkey
    assert paper.url == "https://doi.org/10.1109/TPAMI.2021.3099999"


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
        )
    ]
    service = LiteratureService()
    bibtex = service.generate_bibtex(papers)
    assert "@article{dally2024_resource," in bibtex
    assert "author    = {Dally, William and Horowitz, Mark}," in bibtex
    assert "doi       = {10.1109/TC.2024.12345}," in bibtex


@pytest.mark.asyncio
async def test_search_literature_fallback():
    service = LiteratureService()
    # Test search with fallback resilience
    papers = await service.search_literature("Low-Compute Dynamic Graph Representation", limit=3)
    assert len(papers) >= 3
    for p in papers:
        assert p.doi.startswith("10.")
        assert len(p.authors) > 0
