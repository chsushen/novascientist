"""Unit tests for Autonomous Dataset Discovery & Citation Engine."""

import pytest
from backend.core.dataset_finder import DatasetFinder, DatasetMetadata, ComputationalDomain
from backend.core.latex_assembler import CompliantLaTeXAssembler, AuthorProfile
from backend.core.literature import PaperMetadata, LiteratureService


def test_dataset_discovery_across_domains():
    """Verify discovery resolves canonical datasets for all computational domains."""
    domains = [
        (ComputationalDomain.PHYSICS_SURROGATE, 'Darcy Flow'),
        (ComputationalDomain.VISION, 'ImageNet'),
        (ComputationalDomain.NLP, 'GLUE'),
        (ComputationalDomain.TIMESERIES, 'Electricity'),
        (ComputationalDomain.TABULAR, 'Higgs'),
        (ComputationalDomain.GRAPH, 'OGB-MolHIV'),
    ]

    for dom, expected_keyword in domains:
        dataset = DatasetFinder.discover('General Benchmark Task', dom)
        assert isinstance(dataset, DatasetMetadata)
        assert dataset.domain == dom
        assert dataset.sample_count > 0
        assert len(dataset.dimension) > 0
        assert len(dataset.splits) > 0
        assert dataset.bibtex_key.startswith('dataset_')
        assert dataset.bibtex_entry.startswith('@misc{')


def test_keyword_affinity_matching():
    """Verify specific topic keywords match appropriate domain datasets."""
    # Test Burgers shock matching
    d_burgers = DatasetFinder.discover('Viscous Burgers Equation Shock Profile', 'physics_surrogate')
    assert 'Burgers' in d_burgers.name

    # Test CIFAR robustness matching
    d_cifar = DatasetFinder.discover('CIFAR-100 Corruption Robustness under Noise', 'vision')
    assert 'CIFAR' in d_cifar.name

    # Test WikiText matching
    d_wiki = DatasetFinder.discover('WikiText-103 Long Context Language Modeling', 'nlp')
    assert 'WikiText' in d_wiki.name

    # Test Weather time series matching
    d_weather = DatasetFinder.discover('Multi-variate Weather forecasting and climate sensors', 'timeseries')
    assert 'Weather' in d_weather.name

    # Test METR-LA Traffic / Evacuation matching
    d_metr = DatasetFinder.discover('Disaster Evacuation and Traffic Sensor Network Optimization', 'graph')
    assert 'METR-LA' in d_metr.name
    assert d_metr.sample_count == 34272
    assert '10.1145/3209978.3210006' in (d_metr.doi or '')

    # Test PeMS-BAY Highway performance matching
    d_pems = DatasetFinder.discover('Highway Freeway Congestion & Resilience Monitoring', 'graph')
    assert 'PeMS-BAY' in d_pems.name
    assert d_pems.sample_count == 52116
    assert '10.1109/TITS.2017.2764350' in (d_pems.doi or '')


def test_spatial_disaster_domain_vocabulary_and_hardware_citation():
    """Verify spatial/disaster vocabulary and physical hardware citations are injected into LaTeX."""
    from backend.core.universal_engine import UniversalBenchmarkEngine, get_physical_hardware_info
    from dataclasses import asdict

    hw = get_physical_hardware_info()
    assert "cpu_model" in hw
    assert hw["cpu_cores"] > 0
    assert hw["total_ram_gb"] > 0

    engine = UniversalBenchmarkEngine(
        topic="Disaster Evacuation and Traffic Sensor Corridor Optimization under Bounded Memory",
        num_seeds=5,
    )
    pkg = engine.run_experiments()
    metrics_dict = asdict(pkg)

    assert "physical_latency_ms" in metrics_dict["hardware_info"]
    assert "physical_rss_mb" in metrics_dict["hardware_info"]

    papers = [
        PaperMetadata(
            doi="10.1145/3209978.3210006",
            title="Spatial-Temporal Graph Neural Networks for Traffic Forecasting",
            authors=["Li, Y.", "Yu, R."],
            year=2018,
            venue="ACM SIGKDD",
        )
    ]
    dataset = DatasetFinder.discover(metrics_dict["topic"], "graph")
    assert "METR-LA" in dataset.name

    assembler = CompliantLaTeXAssembler(metrics_dict, papers, author=AuthorProfile(), dataset=dataset)
    latex_text = assembler.generate_latex()

    # Section 1 Domain Vocabulary Check
    assert "spatial traffic monitoring" in latex_text or "evacuation" in latex_text
    assert "corridors" in latex_text or "shelter" in latex_text

    # Section 3 Domain Equations Check
    assert "spatial_graph_aggregation" in latex_text
    assert "shelters" in latex_text or "sensor stations" in latex_text

    # Section 4.1 Physical Hardware Processor Citation Check
    assert hw["cpu_model"] in latex_text
    assert "METR-LA" in latex_text
    assert dataset.bibtex_key in latex_text


def test_bibtex_generation_with_dataset():
    """Verify LiteratureService includes dataset citation in references.bib."""
    lit_service = LiteratureService()
    papers = [
        PaperMetadata(
            doi='10.1109/TNNLS.2024.1234567',
            title='Foundations of Neural Surrogates',
            authors=['Smith, J.', 'Doe, A.'],
            year=2024,
            venue='IEEE TNNLS',
        )
    ]
    dataset = DatasetFinder.discover('Darcy Flow Surrogate', 'physics_surrogate')
    bibtex = lit_service.generate_bibtex(papers, dataset=dataset)

    assert papers[0].bibkey in bibtex
    assert dataset.bibtex_key in bibtex
    assert dataset.name in bibtex
    assert 'Open Scientific Benchmark Consortium' in bibtex


def test_latex_assembler_dataset_injection():
    """Verify LaTeX assembler injects dataset name, sample count, and dimensions."""
    metrics_dict = {
        'topic': 'Physics-Informed Dynamic Neural Surrogates under Bounded Memory',
        'hardware_info': {
            'domain': 'physics_surrogate',
            'domain_name': 'Physics-Informed Neural Surrogates & PDE Dynamics',
            'cpu_model': 'Apple M4',
            'cpu_cores': 10,
            'total_ram_gb': 16.0,
            'architecture': 'arm64',
        },
        'methods': {
            'proposed_mb_qgt': {'name': 'Proposed Dynamic Model', 'mean_accuracy': 0.8931, 'std_accuracy': 0.007, 'mean_memory_mb': 72.0, 'mean_latency_ms': 8.9},
            'dense_baseline': {'name': 'Dense FP32 Baseline', 'mean_accuracy': 0.8313, 'std_accuracy': 0.011, 'mean_memory_mb': 395.0, 'mean_latency_ms': 36.2},
            'post_int8': {'name': 'Static INT8 Quantized Model', 'mean_accuracy': 0.798, 'std_accuracy': 0.014, 'mean_memory_mb': 114.0, 'mean_latency_ms': 23.5},
            'sparse_gnn': {'name': 'Dynamic Sparsified Surrogate', 'mean_accuracy': 0.819, 'std_accuracy': 0.012, 'mean_memory_mb': 160.0, 'mean_latency_ms': 18.9},
        },
        'meta_analysis': {
            'pooled_effect_size': 0.0618,
            'ci_95_lower': 0.053,
            'ci_95_upper': 0.071,
            'z_statistic': 12.68,
            'p_value_z': 0.0001,
            'cochran_q': 0.24,
            'p_value_q': 0.99,
            'tau_squared': 0.0,
            'i_squared_percent': 0.0,
        }
    }

    papers = [
        PaperMetadata(
            doi='10.1109/TNNLS.2024.1234567',
            title='Foundations of Neural Surrogates',
            authors=['Smith, J.'],
            year=2024,
            venue='IEEE TNNLS',
        )
    ]
    dataset = DatasetFinder.discover(metrics_dict['topic'], 'physics_surrogate')
    assembler = CompliantLaTeXAssembler(metrics_dict, papers, author=AuthorProfile(), dataset=dataset)
    latex_text = assembler.generate_latex()

    assert dataset.name in latex_text
    assert f'{dataset.sample_count:,}' in latex_text
    assert dataset.dimension in latex_text
    assert dataset.bibtex_key in latex_text
    assert 'Apple M4' in latex_text
