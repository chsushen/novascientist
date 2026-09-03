"""Tests for ConversationalAgent and ExecutionPlan."""

import pytest
from backend.core.conversational_agent import (
    ConversationalAgent,
    ExecutionMode,
    TargetPaperLength,
)
from backend.core.universal_engine import ComputationalDomain


def test_conversational_agent_topic_refinement():
    agent = ConversationalAgent()
    refined = agent.refine_topic("low-rank dynamic graph attention for evacuation forecasting")
    assert "Disaster" in refined or "Low-Rank" in refined or "Evacuation" in refined
    assert agent.context.domain == ComputationalDomain.GRAPH
    assert agent.context.selected_dataset is not None
    assert len(agent.context.target_venues) > 0


def test_conversational_agent_execution_plan():
    agent = ConversationalAgent("Physics-Informed Neural Surrogates under Bounded Memory")
    agent.set_target_length(TargetPaperLength.FULL_JOURNAL)
    agent.set_execution_mode(ExecutionMode.REAL_PYTORCH_TRAINING)
    agent.set_authorship("Dr. Alice", "Stanford University", "alice@stanford.edu", is_anonymous=False)

    plan = agent.generate_execution_plan()
    summary = plan.to_summary_dict()

    assert "Physics" in summary["topic"]
    assert "Physics" in summary["domain"]
    assert "8–12 Pages" in summary["target_length"]
    assert summary["execution_mode"] == "real_pytorch_training"
    assert "Dr. Alice" in summary["authorship"]
    assert len(plan.stages) == 7
