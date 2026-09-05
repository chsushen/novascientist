"""Integration tests for FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.api.server import create_app
from backend.storage.artifact_store import ArtifactStore
from backend.storage.workspace_manager import WorkspaceManager
from backend.jobs.job_manager import JobManager


@pytest.fixture
def client(tmp_path):
    store = ArtifactStore(base_dir=tmp_path / "artifacts")
    ws = WorkspaceManager(base_dir=tmp_path / "workspaces", artifact_store=store)
    job_mgr = JobManager(workspace_manager=ws, artifact_store=store, max_concurrent_jobs=2)
    app = create_app(workspace_manager=ws, artifact_store=store, job_manager=job_mgr)
    return TestClient(app)


def test_health_and_diagnostics_endpoints(client):
    """Verify /health and /diagnostics return system status and git sha."""
    res_h = client.get("/health")
    assert res_h.status_code == 200
    data_h = res_h.json()
    assert data_h["status"] == "healthy"
    assert "app_version" in data_h

    res_d = client.get("/diagnostics")
    assert res_d.status_code == 200
    data_d = res_d.json()
    assert "git_sha" in data_d
    assert "active_jobs_count" in data_d


def test_project_and_run_api_lifecycle(client):
    """Verify full CRUD cycle for projects and run submissions via REST API."""
    # 1. Create Project
    res_proj = client.post("/api/v1/projects", json={
        "name": "API Test Project",
        "description": "Integration testing project",
        "user_id": "test_user_01"
    })
    assert res_proj.status_code == 201
    proj_data = res_proj.json()
    project_id = proj_data["project_id"]
    assert project_id.startswith("proj_")

    # 2. List Projects
    res_list = client.get("/api/v1/projects")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1

    # 3. Get Project by ID
    res_get = client.get(f"/api/v1/projects/{project_id}")
    assert res_get.status_code == 200
    assert res_get.json()["name"] == "API Test Project"

    # 4. Submit Run
    res_run = client.post(f"/api/v1/projects/{project_id}/runs", json={
        "topic": "Evaluating Parameter-Efficient Adaptation on Domain Classification",
        "target_length": "4_page_conference",
        "execution_mode": "fast_microbenchmark",
        "num_seeds": 2,
        "num_epochs": 2,
    })
    assert res_run.status_code == 202
    run_info = res_run.json()
    assert run_info["status"] == "QUEUED"
    run_id = run_info["run_id"]

    # 5. Check Run Status
    res_status = client.get(f"/api/v1/runs/{run_id}/status")
    assert res_status.status_code == 200
    assert "state" in res_status.json()
