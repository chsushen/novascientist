"""Scholarly Venue Recommender & Acceptance Profile Matcher.

Matches research topics and domain tokens against indexed IEEE, ACM, and Springer
conferences and journals, ranking the top 3 recommended publication targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.core.universal_engine import ComputationalDomain


@dataclass
class VenueProfile:
    """Scholarly publication venue profile and acceptance statistics."""

    name: str
    short_name: str
    publisher: str  # IEEE, ACM, Springer, Elsevier, NeurIPS
    venue_type: str  # Journal, Conference
    impact_factor: float | None
    h5_index: int
    acceptance_rate_pct: float | None
    typical_turnaround_months: float
    topics: list[str]
    template_class: str
    primary_domain: ComputationalDomain


@dataclass
class VenueRecommendation:
    """Ranked venue recommendation with relevance score and rationale."""

    rank: int
    venue: VenueProfile
    relevance_score: float
    match_rationale: str


class VenueMatcher:
    """Indexes top-tier venues and calculates semantic topic-to-venue affinities."""

    VENUES_DATABASE: list[VenueProfile] = [
        # Physics Surrogates / Scientific AI
        VenueProfile(
            name="IEEE Transactions on Neural Networks and Learning Systems",
            short_name="IEEE TNNLS",
            publisher="IEEE",
            venue_type="Journal",
            impact_factor=10.4,
            h5_index=156,
            acceptance_rate_pct=14.5,
            typical_turnaround_months=4.2,
            topics=[
                "neural networks",
                "surrogates",
                "pinn",
                "physics",
                "representation learning",
                "deep learning",
            ],
            template_class="IEEEtran.cls",
            primary_domain=ComputationalDomain.PHYSICS_SURROGATE,
        ),
        VenueProfile(
            name="Journal of Computational Physics",
            short_name="JCP (Elsevier)",
            publisher="Elsevier",
            venue_type="Journal",
            impact_factor=4.6,
            h5_index=118,
            acceptance_rate_pct=22.0,
            typical_turnaround_months=5.5,
            topics=[
                "pde",
                "scientific computing",
                "fluid dynamics",
                "differential equations",
                "numerical methods",
            ],
            template_class="elsarticle.cls",
            primary_domain=ComputationalDomain.PHYSICS_SURROGATE,
        ),
        VenueProfile(
            name="IEEE Transactions on Pattern Analysis and Machine Intelligence",
            short_name="IEEE TPAMI",
            publisher="IEEE",
            venue_type="Journal",
            impact_factor=23.6,
            h5_index=215,
            acceptance_rate_pct=11.2,
            typical_turnaround_months=6.0,
            topics=[
                "pattern analysis",
                "graph neural networks",
                "vision",
                "representation learning",
                "quantization",
            ],
            template_class="IEEEtran.cls",
            primary_domain=ComputationalDomain.GRAPH,
        ),
        VenueProfile(
            name="ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
            short_name="ACM KDD",
            publisher="ACM",
            venue_type="Conference",
            impact_factor=None,
            h5_index=142,
            acceptance_rate_pct=15.1,
            typical_turnaround_months=3.0,
            topics=[
                "graph mining",
                "large-scale learning",
                "quantized memory",
                "algorithms",
                "networks",
            ],
            template_class="acmart.cls",
            primary_domain=ComputationalDomain.GRAPH,
        ),
        VenueProfile(
            name="IEEE Transactions on Computers",
            short_name="IEEE TC",
            publisher="IEEE",
            venue_type="Journal",
            impact_factor=3.7,
            h5_index=84,
            acceptance_rate_pct=18.0,
            typical_turnaround_months=4.5,
            topics=[
                "computer architecture",
                "low-compute",
                "embedded systems",
                "quantized arithmetic",
                "memory bounds",
            ],
            template_class="IEEEtran.cls",
            primary_domain=ComputationalDomain.GRAPH,
        ),
        VenueProfile(
            name="IEEE/CVF Conference on Computer Vision and Pattern Recognition",
            short_name="CVPR",
            publisher="IEEE",
            venue_type="Conference",
            impact_factor=None,
            h5_index=356,
            acceptance_rate_pct=23.6,
            typical_turnaround_months=3.5,
            topics=[
                "computer vision",
                "visual representations",
                "image recognition",
                "transformers",
            ],
            template_class="cvpr.cls",
            primary_domain=ComputationalDomain.VISION,
        ),
        VenueProfile(
            name="Neural Information Processing Systems",
            short_name="NeurIPS",
            publisher="NeurIPS",
            venue_type="Conference",
            impact_factor=None,
            h5_index=285,
            acceptance_rate_pct=25.8,
            typical_turnaround_months=3.0,
            topics=[
                "machine learning",
                "neural surrogates",
                "optimization",
                "deep learning",
                "statistical theory",
            ],
            template_class="neurips.sty",
            primary_domain=ComputationalDomain.PHYSICS_SURROGATE,
        ),
    ]

    @classmethod
    def match_venues(
        cls, topic: str, domain: Any, top_k: int = 3
    ) -> list[VenueRecommendation]:
        """Rank target publication venues based on topic keywords and computational domain."""
        topic_tokens = set(re.findall(r"\w+", topic.lower()))
        domain_str = domain.value if hasattr(domain, "value") else str(domain)

        # Match domain enum if possible
        matched_domain_enum = None
        if isinstance(domain, ComputationalDomain):
            matched_domain_enum = domain
        else:
            for cd in ComputationalDomain:
                if cd.value == domain_str or cd.name.lower() == domain_str.lower():
                    matched_domain_enum = cd
                    break

        scored: list[tuple[float, VenueProfile, str]] = []

        for v in cls.VENUES_DATABASE:
            score = 0.0
            reasons = []

            # Domain congruence bonus
            if (
                v.primary_domain == domain
                or (matched_domain_enum and v.primary_domain == matched_domain_enum)
                or v.primary_domain.value == domain_str
            ):
                score += 4.5
                reasons.append(f"Domain alignment with {domain_str}")

            # Topic keyword matching
            matched_kws = []
            for topic_kw in v.topics:
                kw_tokens = set(topic_kw.lower().split())
                overlap = topic_tokens.intersection(kw_tokens)
                if overlap:
                    matched_kws.append(topic_kw)
                    score += len(overlap) * 2.2

            if matched_kws:
                reasons.append(f"Key topic overlap: {', '.join(matched_kws[:3])}")

            # Higher impact factor / h5 bonus
            if v.impact_factor:
                score += min(2.0, v.impact_factor * 0.1)
            elif v.h5_index:
                score += min(2.0, v.h5_index * 0.01)

            rationale = (
                "; ".join(reasons)
                if reasons
                else "Broad methodology and systems suitability"
            )
            scored.append((score, v, rationale))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        recommendations: list[VenueRecommendation] = []
        for idx, (score, venue, rationale) in enumerate(scored[:top_k], start=1):
            norm_score = round(min(0.99, 0.50 + score * 0.05), 2)
            recommendations.append(
                VenueRecommendation(
                    rank=idx,
                    venue=venue,
                    relevance_score=norm_score,
                    match_rationale=rationale,
                )
            )

        return recommendations
