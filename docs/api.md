# NovaScientist Production REST API Reference

The NovaScientist REST API provides asynchronous research project management, background job execution, and verified artifact retrieval.

## Base URL
```
http://localhost:8000
```

---

## 1. System Diagnostics

### `GET /health`
Liveness and health check probe.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "app_version": "2.3.0",
  "environment": "production",
  "timestamp": 1788540000.0
}
```

### `GET /diagnostics`
Telemetry on active jobs, queue depth, Git SHA, and storage status.
- **Response `200 OK`**:
```json
{
  "application": "NovaScientist v2.3",
  "git_sha": "42a335e",
  "version": "2.3.0",
  "active_jobs_count": 0,
  "queued_jobs_count": 0,
  "total_projects": 3,
  "total_runs": 8,
  "data_directory": "/app/.novascientist_data",
  "demo_mode": false
}
```

---

## 2. Research Projects

### `POST /api/v1/projects`
Create a new research workspace project.
- **Request Body**:
```json
{
  "name": "Spatiotemporal Graph Modeling",
  "description": "Topological graph neural network investigations",
  "user_id": "dr_smith"
}
```
- **Response `201 Created`**:
```json
{
  "project_id": "proj_a1b2c3d4e5",
  "user_id": "dr_smith",
  "name": "Spatiotemporal Graph Modeling",
  "description": "Topological graph neural network investigations",
  "created_at": 1788540100.0,
  "updated_at": 1788540100.0,
  "runs": []
}
```

### `GET /api/v1/projects`
List all projects. Optional query parameter: `?user_id=...`.

### `GET /api/v1/projects/{project_id}`
Retrieve project metadata and constituent research runs.

---

## 3. Asynchronous Research Runs

### `POST /api/v1/projects/{project_id}/runs`
Submit and queue an asynchronous research run.
- **Request Body**:
```json
{
  "topic": "Can retrieval-augmented generation improve factual consistency in domain-specific question answering?",
  "author_name": "Elena Rostova",
  "author_affiliation": "Max Planck Institute for Informatics",
  "author_email": "erostova@mpi-inf.mpg.de",
  "target_length": "8_12_pages_journal",
  "execution_mode": "fast_microbenchmark",
  "num_seeds": 5,
  "num_epochs": 5
}
```
- **Response `202 Accepted`**:
```json
{
  "status": "QUEUED",
  "job_id": "job_run_9876543210",
  "run_id": "run_9876543210",
  "project_id": "proj_a1b2c3d4e5",
  "topic": "Can retrieval-augmented generation improve factual consistency in domain-specific question answering?"
}
```

### `GET /api/v1/runs/{run_id}/status`
Poll live execution progress.
- **Response `200 OK`**:
```json
{
  "job_id": "job_run_9876543210",
  "run_id": "run_9876543210",
  "state": "RUNNING",
  "progress_percent": 68.0,
  "current_stage": "Executing Multi-Seed Training",
  "stage_message": "Evaluating deterministic seed 3/5..."
}
```

### `POST /api/v1/runs/{run_id}/cancel`
Cancel an active or queued research run.

---

## 4. Evidence, Contracts & Artifacts

### `GET /api/v1/runs/{run_id}/contract`
Retrieve the frozen `ScientificResearchContract` governing all downstream execution.

### `GET /api/v1/runs/{run_id}/evidence`
Retrieve verified CrossRef/OpenAlex literature evidence and extracted claims.

### `GET /api/v1/runs/{run_id}/results`
Retrieve empirical multi-seed telemetry and metrics.

### `GET /api/v1/runs/{run_id}/statistics`
Retrieve the statistical critique report and hypothesis validation metrics.

### `GET /api/v1/runs/{run_id}/provenance`
Retrieve the full provenance DAG (nodes and edges).

### `GET /api/v1/runs/{run_id}/paper`
Retrieve the compiled LaTeX source, physical page count, and artifact identifiers.

### `GET /api/v1/runs/{run_id}/artifacts/{artifact_id}/download`
Download verified immutable artifact payload (PDF, LaTeX, Overleaf ZIP, figures, metrics).
