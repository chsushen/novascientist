"""Isolated test fixtures for deterministic testing of Literature and Evidence agents.

CRITICAL ARCHITECTURAL RULE:
These fixtures are strictly for test execution and must NEVER be imported,
referenced, or returned by production code in backend/.
"""

from typing import List
from backend.core.literature import PaperMetadata

TEST_FIXTURE = True


def get_canonical_test_papers() -> List[PaperMetadata]:
    """Return isolated deterministic test papers clearly stamped with test_fixture provenance."""
    return [
        PaperMetadata(
            doi="10.1109/TPAMI.2021.3099999",
            title="Adaptive Quantization and Memory-Bounded Graph Neural Networks",
            authors=["Kipf, Thomas", "Welling, Max", "Hamilton, William L."],
            year=2022,
            venue="IEEE Transactions on Pattern Analysis and Machine Intelligence",
            citation_count=412,
            abstract="We investigate adaptive low-bit integer quantization for graph neural networks operating under constrained memory budgets. By bounding gradient variance through straight-through estimation and dynamic tile alignment, we achieve 4x memory reduction while preserving node classification accuracy within 0.8% of dense FP32 baselines.",
            url="https://doi.org/10.1109/TPAMI.2021.3099999",
            source_origin="test_fixture",
            text_origin="test_fixture",
        ),
        PaperMetadata(
            doi="10.1145/3534678.3539001",
            title="Dynamic Graph Compression under Strict Memory Budgets",
            authors=["Leskovec, Jure", "You, Jiaxuan", "Ying, Rex"],
            year=2023,
            venue="ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
            citation_count=185,
            abstract="Large-scale graph representation learning is constrained by the memory wall during multi-hop neighborhood aggregation. We propose dynamic graph compression with stochastic cache-line scheduling, reducing peak working memory footprint from 390 MB to 72 MB on edge commodity hardware.",
            url="https://doi.org/10.1145/3534678.3539001",
            source_origin="test_fixture",
            text_origin="test_fixture",
        ),
        PaperMetadata(
            doi="10.1109/TC.2023.3289012",
            title="Resource-Constrained Representation Learning on Embedded Vector Processors",
            authors=["Dally, William", "Horowitz, Mark", "Keutzer, Kurt"],
            year=2024,
            venue="IEEE Transactions on Computers",
            citation_count=98,
            abstract="Evaluating continuous neural operators on edge hardware induces severe memory bandwidth saturation. We demonstrate that 64-byte aligned SIMD register caching eliminates non-contiguous cache misses and achieves a 4.1x arithmetic throughput acceleration on vector execution units.",
            url="https://doi.org/10.1109/TC.2023.3289012",
            source_origin="test_fixture",
            text_origin="test_fixture",
        ),
    ]


class MockLiteratureService:
    """Mock literature service for testing literature and evidence agents with fixtures."""

    def __init__(self, papers: List[PaperMetadata] = None) -> None:
        self.papers = papers if papers is not None else get_canonical_test_papers()

    async def search_literature(self, topic: str, limit: int = 5) -> List[PaperMetadata]:
        return self.papers[:limit]
