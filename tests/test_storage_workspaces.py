"""Unit and integration tests for Persistent Workspaces and ArtifactStore."""

import pytest
import os
import tempfile
from pathlib import Path

from backend.storage.artifact_store import ArtifactStore, ArtifactType, ArtifactIntegrityError
from backend.storage.workspace_manager import WorkspaceManager, RunStatus


def test_artifact_store_read_write_integrity(tmp_path):
    """Verify artifact write, SHA-256 computation, and tamper-detection."""
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    
    payload = "Sample LaTeX manuscript content for research testing."
    art = store.store_text(
        project_id="proj_test_01",
        run_id="run_test_01",
        artifact_type=ArtifactType.MANUSCRIPT,
        filename="manuscript.tex",
        content=payload,
    )
    
    assert art.artifact_id.startswith("art_")
    assert len(art.sha256) == 64
    assert store.read_text(art.artifact_id) == payload

    # Tamper with file to verify fail-closed integrity check
    with open(art.location, "a") as f:
        f.write("\nTampered unauthorized content")

    with pytest.raises(ArtifactIntegrityError):
        store.read_text(art.artifact_id)


def test_workspace_manager_project_and_run_lifecycle(tmp_path):
    """Verify project creation, run queueing, status updates, and run comparison."""
    ws = WorkspaceManager(base_dir=tmp_path / "workspaces")

    # 1. Create Project
    proj = ws.create_project(
        name="Graph Neural Network Research",
        description="Investigation into topological graph embeddings",
        user_id="researcher_42",
    )
    assert proj.name == "Graph Neural Network Research"
    assert proj.user_id == "researcher_42"
    assert proj.project_id.startswith("proj_")

    # 2. Create Runs
    run1 = ws.create_run(proj.project_id, topic="Graph Fraud Detection with Imbalanced Nodes")
    run2 = ws.create_run(proj.project_id, topic="Graph Traffic Forecasting with Spatial Lag")

    assert run1.status == RunStatus.QUEUED
    assert run2.status == RunStatus.QUEUED
    assert len(ws.list_runs(proj.project_id)) == 2

    # 3. Update Run Status
    ws.update_run(
        run1.run_id,
        status=RunStatus.COMPLETED,
        page_count=8,
        contract_data={"contract_id": "c1", "selected_dataset": "METR-LA"},
    )
    retrieved_run = ws.get_run(run1.run_id)
    assert retrieved_run.status == RunStatus.COMPLETED
    assert retrieved_run.page_count == 8

    # 4. Compare Runs
    comp = ws.compare_runs([run1.run_id, run2.run_id])
    assert comp["runs_evaluated"] == 2
    assert run1.run_id in comp["runs"]
    assert run2.run_id in comp["runs"]
