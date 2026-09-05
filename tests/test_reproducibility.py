"""Unit tests for Reproducibility Manifest and Provenance Graph Verifier."""

import pytest
from backend.reproducibility.manifest_generator import (
    ReproducibilityGenerator,
    ProvenanceGraphVerifier,
    ProvenanceIntegrityError,
)


def test_reproducibility_manifest_generation():
    """Verify manifest generation records git sha, hardware, seeds, and dataset info."""
    contract_data = {
        "contract_id": "contract_abc123",
        "selected_method": "Adaptive Transformer Framework",
    }
    experiment_spec = {"num_seeds": 5, "num_epochs": 40}

    manifest = ReproducibilityGenerator.generate_manifest(
        run_id="run_test_001",
        contract_data=contract_data,
        experiment_spec=experiment_spec,
        dataset_name="GLUE Benchmark",
        random_seeds=[42, 137, 2024],
        provenance_data={"nodes": {"q1": {"parent_ids": []}}},
    )

    assert manifest.run_id == "run_test_001"
    assert len(manifest.git_sha) >= 7
    assert manifest.hardware_cpu != ""
    assert manifest.random_seeds == [42, 137, 2024]
    assert manifest.dataset_identifier == "GLUE Benchmark"
    assert manifest.contract_id == "contract_abc123"


def test_provenance_dag_verifier():
    """Verify DAG verifier validates closure and catches dangling references."""
    # Valid DAG
    valid_dag = {
        "nodes": {
            "root_question": {"parent_ids": []},
            "plan": {"parent_ids": ["root_question"]},
            "experiment": {"parent_ids": ["plan"]},
            "result": {"parent_ids": ["experiment"]},
        }
    }
    assert ProvenanceGraphVerifier.verify_dag(valid_dag) is True

    # Dangling reference (parent does not exist)
    dangling_dag = {
        "nodes": {
            "root_question": {"parent_ids": []},
            "plan": {"parent_ids": ["non_existent_parent"]},
        }
    }
    with pytest.raises(ProvenanceIntegrityError):
        ProvenanceGraphVerifier.verify_dag(dangling_dag)
