"""Tests for NovaScientistOrchestrator."""

import pytest
from backend.core.conversational_agent import ExecutionMode, TargetPaperLength
from backend.core.latex_assembler import AuthorProfile
from backend.core.orchestrator import NovaScientistOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_execution(tmp_path):
    orchestrator = NovaScientistOrchestrator(output_dir=str(tmp_path / "dist"))
    author = AuthorProfile(name="Test Author", affiliation="Test Org", email="test@org.edu")

    progress_events = []
    def on_progress(msg: str, pct: float):
        progress_events.append((msg, pct))

    result = await orchestrator.execute(
        topic="Low-Rank Dynamic Graph Attention for Smart Disaster Resilience",
        author=author,
        target_length=TargetPaperLength.FULL_JOURNAL,
        execution_mode=ExecutionMode.REAL_PYTORCH_TRAINING,
        num_seeds=2,
        num_epochs=3,
        progress_callback=on_progress,
    )

    assert result.success is True
    assert "Low-Rank" in result.topic or "Disaster" in result.topic
    assert result.page_count >= 1
    assert len(result.figures) >= 1
    assert (tmp_path / "dist" / "workspace" / "main.tex").exists()
    assert (tmp_path / "dist" / "workspace" / "references.bib").exists()
    assert len(progress_events) > 0
