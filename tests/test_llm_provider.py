"""Unit tests for LLM Provider abstraction and cost tracking."""

import pytest
from backend.llm.provider import (
    MockDeterministicProvider,
    LLMCostTracker,
    CostExceededError,
    FallbackProvider,
    get_configured_llm_provider,
)


@pytest.mark.asyncio
async def test_mock_deterministic_provider():
    """Verify MockDeterministicProvider generates responses with valid token accounting."""
    tracker = LLMCostTracker(cost_limit_usd=5.0)
    provider = MockDeterministicProvider(cost_tracker=tracker)

    resp = await provider.generate(
        prompt="Synthesize research evidence for retrieval-augmented generation.",
        expected_json_keys=["summary", "confidence"],
    )

    assert resp.total_tokens > 0
    assert resp.estimated_cost_usd > 0.0
    assert resp.parsed_json is not None
    assert "summary" in resp.parsed_json
    assert "confidence" in resp.parsed_json

    summary = tracker.get_summary()
    assert summary["total_calls"] == 1
    assert summary["total_cost_usd"] > 0


@pytest.mark.asyncio
async def test_cost_tracker_budget_enforcement():
    """Verify CostTracker raises CostExceededError when spending surpasses limit."""
    tracker = LLMCostTracker(cost_limit_usd=0.001)
    provider = MockDeterministicProvider(cost_tracker=tracker)

    # Large prompt to exceed 0.001 limit
    with pytest.raises(CostExceededError):
        for _ in range(50):
            await provider.generate(prompt="Long repeated prompt " * 200)


@pytest.mark.asyncio
async def test_fallback_provider_chain():
    """Verify FallbackProvider succeeds when first provider in chain succeeds."""
    tracker = LLMCostTracker()
    p1 = MockDeterministicProvider(cost_tracker=tracker)
    p2 = MockDeterministicProvider(cost_tracker=tracker)
    fallback = FallbackProvider(primary=p1, fallbacks=[p2])

    resp = await fallback.generate(prompt="Test fallback prompt")
    assert resp.text is not None
