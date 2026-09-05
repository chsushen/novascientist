"""
NovaScientist Conversational Requirement-Gathering Engine.

Manages conversational state, interactive dialog transitions, and execution plan formulation
for autonomous research paper generation and hardware benchmarking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.core.universal_engine import (
    ComputationalDomain,
    UniversalDomainDispatcher,
    get_physical_hardware_info,
)
from backend.core.dataset_finder import DatasetFinder, DatasetMetadata
from backend.core.venue_matcher import VenueMatcher, VenueRecommendation


class TargetPaperLength(str, Enum):
    """Target manuscript length and format."""
    SHORT_CONFERENCE = "4_pages_conference"
    FULL_JOURNAL = "8_12_pages_journal"


class ExecutionMode(str, Enum):
    """Hardware execution and training mode."""
    REAL_PYTORCH_TRAINING = "real_pytorch_training"
    FAST_MICROBENCHMARK = "fast_microbenchmark"


@dataclass
class ResearchContext:
    """User-specified research context and emphasis points."""
    raw_topic: str
    refined_topic: str = ""
    domain: Optional[ComputationalDomain] = None
    domain_display_name: str = ""
    target_length: TargetPaperLength = TargetPaperLength.FULL_JOURNAL
    execution_mode: ExecutionMode = ExecutionMode.REAL_PYTORCH_TRAINING
    custom_gaps: List[str] = field(default_factory=list)
    baselines_to_compare: List[str] = field(default_factory=list)
    novelty_points: List[str] = field(default_factory=list)
    target_venues: List[VenueRecommendation] = field(default_factory=list)
    selected_dataset: Optional[DatasetMetadata] = None
    num_seeds: int = 5
    author_name: str = "Anonymous Author(s)"
    affiliation: str = "Affiliation Withheld for Double-Blind Review"
    email: str = "anonymous@conference-review.org"
    is_anonymous: bool = True


@dataclass
class ExecutionPlan:
    """Formal approved execution plan ready for pipeline orchestration."""
    context: ResearchContext
    dataset_name: str
    dataset_samples: int
    hardware_summary: str
    target_venue_name: str
    expected_page_count: str
    stages: List[str]
    is_approved: bool = False

    def to_summary_dict(self) -> Dict[str, Any]:
        """Serialize plan for UI rendering."""
        return {
            "topic": self.context.refined_topic or self.context.raw_topic,
            "domain": self.context.domain_display_name,
            "target_length": self.expected_page_count,
            "execution_mode": self.context.execution_mode.value,
            "hardware": self.hardware_summary,
            "dataset": f"{self.dataset_name} ({self.dataset_samples:,} samples)",
            "primary_venue": self.target_venue_name,
            "authorship": f"{self.context.author_name} ({'Double-Blind' if self.context.is_anonymous else self.context.affiliation})",
            "seeds": f"k = {self.context.num_seeds}",
            "novelty_focus": self.context.novelty_points or ["Task-Specific Inductive Architecture", "Multi-Seed Grounded Validation"],
            "stages_count": len(self.stages),
        }


class ConversationalAgent:
    """Interactive conversational agent for requirement gathering and plan formulation."""

    def __init__(self, initial_topic: str = "") -> None:
        self.context = ResearchContext(raw_topic=initial_topic)
        self.hardware_info = get_physical_hardware_info()
        if initial_topic:
            self.refine_topic(initial_topic)

    def refine_topic(self, topic: str) -> str:
        """Analyze, sanitize, and title-case the research topic."""
        self.context.raw_topic = topic.strip()
        from backend.core.latex_assembler import CompliantLaTeXAssembler
        self.context.refined_topic = CompliantLaTeXAssembler.format_academic_title(topic)

        # Domain classification & canonical dataset discovery
        classification = UniversalDomainDispatcher.classify_topic(self.context.refined_topic)
        self.context.domain = classification.domain
        self.context.domain_display_name = classification.domain_display_name

        dataset = DatasetFinder.discover(self.context.refined_topic, classification.domain)
        self.context.selected_dataset = dataset

        # Venue recommendation matching
        venues = VenueMatcher.match_venues(self.context.refined_topic, classification.domain, top_k=3)
        self.context.target_venues = venues

        # Extract automated novelty and research gaps based on topic
        self._derive_automated_gaps_and_novelty()
        return self.context.refined_topic

    def _derive_automated_gaps_and_novelty(self) -> None:
        """Derive dynamic, topic-tailored research gaps, novelty focus points, and baselines."""
        from backend.core.topic_profile import TopicProfileExtractor
        from backend.core.research_contract import ResearchContractBuilder

        dom_val = self.context.domain.value if self.context.domain else None
        profile = TopicProfileExtractor.extract(self.context.refined_topic, domain=dom_val)
        contract = ResearchContractBuilder.build_contract(self.context.refined_topic, profile)

        # Populate context dynamically
        self.context.custom_gaps = [
            f"Lack of robust optimization and evaluation protocols for {profile.subdomain.lower()}",
            f"Severe performance degradation under non-stationary shifts in {profile.task_type.value.replace('_', ' ')}",
            f"High resource overhead and latency bottlenecks in canonical baseline architectures",
        ]
        self.context.novelty_points = [
            f"Proposed {profile.model_full_name_suggestion} tailored to {profile.subdomain}",
            f"Multi-seed deterministic evaluation across canonical {contract.selected_dataset}",
            f"Grounded empirical analysis evaluating {', '.join(contract.primary_metrics)}",
        ]
        self.context.baselines_to_compare = contract.selected_baselines or profile.candidate_baselines

    def set_target_length(self, length: TargetPaperLength) -> None:
        """Set target publication length."""
        self.context.target_length = length

    def set_execution_mode(self, mode: ExecutionMode) -> None:
        """Set hardware execution mode."""
        self.context.execution_mode = mode

    def set_authorship(self, name: str, affiliation: str, email: str, is_anonymous: bool = False) -> None:
        """Configure paper authorship and double-blind settings."""
        self.context.is_anonymous = is_anonymous
        if is_anonymous:
            self.context.author_name = "Anonymous Author(s)"
            self.context.affiliation = "Affiliation Withheld for Double-Blind Review"
            self.context.email = "anonymous@conference-review.org"
        else:
            self.context.author_name = name
            self.context.affiliation = affiliation
            self.context.email = email

    def generate_execution_plan(self) -> ExecutionPlan:
        """Synthesize the complete structured execution plan for user review."""
        if not self.context.refined_topic:
            self.refine_topic(self.context.raw_topic or "Dynamic Representation Learning under Memory Bounds")

        ds = self.context.selected_dataset or DatasetFinder.discover(self.context.refined_topic, self.context.domain or ComputationalDomain.GRAPH)
        primary_venue = self.context.target_venues[0].venue.name if self.context.target_venues else "IEEE Transactions on Pattern Analysis and Machine Intelligence"

        page_str = "8–12 Pages (Full IEEE Transactions Journal)" if self.context.target_length == TargetPaperLength.FULL_JOURNAL else "4–6 Pages (IEEE Conference Paper)"
        hw_str = f"{self.hardware_info['cpu_model']} ({self.hardware_info['cpu_cores']} cores, {self.hardware_info['architecture']}, {self.hardware_info['total_ram_gb']} GB RAM)"

        stages = [
            "1. Scholarly Literature Discovery (25-40 Verified DOIs via CrossRef & OpenAlex)",
            "2. Static AST Dataflow Guard Audit (Zero Train-Val Contamination)",
            "3. Real PyTorch Hardware Training Sandbox (GPU/MPS/CPU with AdamW across 5 seeds)",
            "4. Publication Vector Plotting Suite (5 High-Resolution Vector Figures)",
            "5. Deep IEEE Transactions Journal Assembly (10 Complete Structured Sections)",
            "6. Adversarial Reviewer Swarm Audit (Statistical Power & Rhetoric Linter)",
            "7. Tectonic XeTeX Engine Compilation & Self-Contained Overleaf Packaging",
        ]

        return ExecutionPlan(
            context=self.context,
            dataset_name=ds.name,
            dataset_samples=ds.sample_count,
            hardware_summary=hw_str,
            target_venue_name=primary_venue,
            expected_page_count=page_str,
            stages=stages,
            is_approved=False,
        )
