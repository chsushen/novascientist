"""NovaScientist Topic & Task Research Profiling Engine.

Dynamically analyzes user research questions, abstracts, and domains to derive
a structured, multidimensional TopicResearchProfile consisting of candidate pools
(metrics, baselines, datasets, method families, mathematical objects, figures)
that downstream agents score and select based on evidence and experimental telemetry.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from backend.core.dataset_finder import DatasetFinder


class ResearchParadigm(str, Enum):
    """Core scientific paradigm governing the research investigation."""

    EMPIRICAL_BENCHMARK = "empirical_benchmark"
    THEORETICAL_ALGORITHMIC = "theoretical_algorithmic"
    SYSTEMS_OPTIMIZATION = "systems_optimization"
    METHODOLOGICAL_COMPARISON = "methodological_comparison"
    APPLIED_DOMAIN_STUDY = "applied_domain_study"
    OPTIMIZATION = "systems_optimization"


class TaskType(str, Enum):
    """Specific computational or scientific task category."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    LANGUAGE_MODELING = "language_modeling"
    SEQUENCE_GENERATION = "sequence_generation"
    GENERATION = "generation"
    OBJECT_DETECTION_SEGMENTATION = "object_detection_segmentation"
    TIMESERIES_FORECASTING = "timeseries_forecasting"
    FORECASTING = "timeseries_forecasting"
    GRAPH_REASONING = "graph_reasoning"
    FEDERATED_COORDINATION = "federated_coordination"
    PDE_OPERATOR_LEARNING = "pde_operator_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CLINICAL_RISK_ASSESSMENT = "clinical_risk_assessment"
    TABULAR_PREDICTION = "tabular_prediction"
    QUANTUM_SIMULATION = "quantum_simulation"


class DataModality(str, Enum):
    """Primary data representation and structure."""

    IMAGE_VOLUMETRIC = "image_volumetric"
    IMAGE = "image_volumetric"
    NATURAL_LANGUAGE_TEXT = "natural_language_text"
    TEXT = "natural_language_text"
    MULTIVARIATE_TIME_SERIES = "multivariate_time_series"
    TIME_SERIES = "multivariate_time_series"
    RELATIONAL_GRAPH = "relational_graph"
    GRAPH = "relational_graph"
    TABULAR_HETEROGENEOUS = "tabular_heterogeneous"
    TABULAR = "tabular_heterogeneous"
    SPATIOTEMPORAL_GRID = "spatiotemporal_grid"
    BIOLOGICAL_SEQUENCES = "biological_sequences"
    QUANTUM_CIRCUITS = "quantum_circuits"


@dataclass
class TopicResearchProfile:
    """Structured candidate-driven profile extracted from a research topic and context."""

    research_question: str
    domain: str
    subdomain: str
    task_type: TaskType
    research_paradigm: ResearchParadigm
    data_modality: DataModality
    problem_characteristics: list[str] = field(default_factory=list)
    candidate_metrics: list[str] = field(default_factory=list)
    primary_metric: str = "Accuracy (%)"
    candidate_datasets: list[str] = field(default_factory=list)
    candidate_baselines: list[str] = field(default_factory=list)
    candidate_method_families: list[str] = field(default_factory=list)
    mathematical_objects: list[str] = field(default_factory=list)
    likely_statistical_tests: list[str] = field(default_factory=list)
    figure_candidates: list[str] = field(default_factory=list)
    publication_conventions: list[str] = field(default_factory=list)
    requires_formal_theorem: bool = False
    model_acronym_suggestion: str = ""
    model_full_name_suggestion: str = ""
    inferred_domain_priors: dict[str, Any] = field(default_factory=dict)

    @property
    def topic(self) -> str:
        return self.research_question

    @property
    def profile_id(self) -> str:
        return f"prof_{abs(hash(self.research_question)) % 100000:05d}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile into structured machine-readable dictionary."""
        d = asdict(self)
        d["topic"] = self.topic
        d["profile_id"] = self.profile_id
        d["task_type"] = (
            self.task_type.value
            if isinstance(self.task_type, TaskType)
            else str(self.task_type)
        )
        d["research_paradigm"] = (
            self.research_paradigm.value
            if isinstance(self.research_paradigm, ResearchParadigm)
            else str(self.research_paradigm)
        )
        d["data_modality"] = (
            self.data_modality.value
            if isinstance(self.data_modality, DataModality)
            else str(self.data_modality)
        )
        return d


class TopicProfileExtractor:
    """Analyzes natural language research topics and context to produce TopicResearchProfiles."""

    @classmethod
    def extract(
        cls,
        topic: str,
        domain: str | None = None,
        additional_context: str | None = None,
        target_format: str = "8_12_pages_journal",
    ) -> TopicResearchProfile:
        """Standardized interface for deriving topic research profiles."""
        return cls.profile_topic(
            topic,
            explicit_domain=domain,
            additional_context=additional_context,
            target_format=target_format,
        )

    @classmethod
    def profile_topic(
        cls,
        topic: str,
        explicit_domain: str | None = None,
        additional_context: str | None = None,
        target_format: str = "8_12_pages_journal",
    ) -> TopicResearchProfile:
        """Derive a candidate-rich scientific research profile dynamically from the topic."""
        t_lower = (topic + " " + (additional_context or "")).lower()
        dom_lower = (explicit_domain or "").lower()

        # Evidence-driven domain and task inference
        if any(
            k in t_lower
            for k in [
                "rag",
                "retrieval-augmented",
                "retrieval augmented",
                "factual consistency",
                "factuality",
                "hallucination",
                "question answering",
            ]
        ) or (
            dom_lower == "nlp"
            and any(k in t_lower for k in ["retrieval", "qa", "fact", "rag"])
        ):
            task = TaskType.SEQUENCE_GENERATION
            modality = DataModality.NATURAL_LANGUAGE_TEXT
            domain_name = "nlp"
            subdomain = "Retrieval-Augmented Generation & Factual Consistency"
            paradigm = ResearchParadigm.EMPIRICAL_BENCHMARK
            metrics = [
                "Factual Consistency Score (%)",
                "Exact Match (EM %)",
                "Token F1 Score (%)",
                "Hallucination Rate (%)",
            ]
            primary_m = "Factual Consistency Score (%)"
            baselines = [
                "Dense Passage Retrieval (DPR) + LLM",
                "BM25 Keyword Retrieval + Cross-Encoder",
                "Closed-Book Parametric LLM",
                "Standard RAG Baseline",
            ]
            method_families = [
                "Iterative Context Reranking",
                "Faithfulness-Guided Generation",
                "Dense Multi-Hop Retrieval",
            ]
            math_objs = [
                "Retrieval Probability P(z|x) = softmax(E_q(x)^T E_d(z))",
                "Marginal Sequence Likelihood P(y|x) = \\sum_z P(z|x)P(y|x,z)",
                "Faithfulness Attribution Score",
                "Context Relevance Loss",
            ]
            stat_tests = [
                "Paired Student's t-test on Factuality Scores",
                "Bootstrap Resampling Significance Test",
                "DerSimonian-Laird Random-Effects Meta-Analysis",
            ]
            figs = [
                "Factual Consistency vs Retrieval Depth (Top-k)",
                "Hallucination Rate across Context Densities",
                "Exact Match vs Context Relevance Frontier",
                "Ablation over Reranker Architectures",
            ]
            requires_theorem = False

        elif dom_lower in ["signal_processing", "industrial_iot", "vibration"] or any(
            k in t_lower
            for k in [
                "vibration",
                "machinery",
                "rotating",
                "bearing",
                "fault detection",
                "sensor anomaly",
                "acoustic",
                "accelerometer",
                "frequency spectrum",
                "fft",
                "spectral",
                "condition monitoring",
            ]
        ):
            task = TaskType.CLASSIFICATION
            modality = DataModality.MULTIVARIATE_TIME_SERIES
            domain_name = "signal_processing"
            subdomain = "Industrial Machinery Diagnostics & Vibration Anomaly Detection"
            paradigm = ResearchParadigm.EMPIRICAL_BENCHMARK
            metrics = [
                "Fault Detection F1-Score (%)",
                "Area Under ROC Curve (AUROC %)",
                "Early Anomaly Lead Time (hours)",
                "False Alarm Rate (%)",
            ]
            primary_m = "Fault Detection F1-Score (%)"
            baselines = [
                "FFT Spectral Energy Baseline",
                "1D Convolutional Vibration Net",
                "Wavelet Packet Random Forest",
                "Temporal Transformer Baseline",
            ]
            method_families = [
                "Continuous Wavelet Transform + CNN",
                "Adaptive Multi-Scale Fourier Filtering",
                "Phase-Space Trajectory Embeddings",
            ]
            math_objs = [
                "Short-Time Fourier Transform STFT(t, \\omega)",
                "Continuous Wavelet Transform CWT(a, b)",
                "Spectral Kurtosis SK(f)",
                "Envelope Hilbert Transform H[x(t)]",
            ]
            stat_tests = [
                "Paired Student's t-test across Machine Operating Regimes",
                "Wilcoxon Signed-Rank Test",
                "DerSimonian-Laird Meta-Analysis",
            ]
            figs = [
                "Time-Frequency Spectrogram & Fault Feature Activation",
                "AUROC Anomaly Detection Frontier",
                "Early Anomaly Lead Time vs False Alarm Rate",
                "Ablation over Wavelet Decomposition Scales",
            ]
            requires_theorem = False

        elif dom_lower == "graph" or any(
            k in t_lower
            for k in [
                "graph",
                "gnn",
                "node",
                "edge",
                "topology",
                "relational",
                "fraud",
                "imbalance",
            ]
        ):
            task = TaskType.GRAPH_REASONING
            modality = DataModality.RELATIONAL_GRAPH
            domain_name = "graph"
            subdomain = (
                "Topological Graph Representation & Imbalanced Node Classification"
            )
            paradigm = (
                ResearchParadigm.THEORETICAL_ALGORITHMIC
                if "expressive" in t_lower or "bound" in t_lower
                else ResearchParadigm.EMPIRICAL_BENCHMARK
            )
            metrics = [
                "Area Under Precision-Recall Curve (AUPRC)",
                "Minority-Class F1 Score",
                "Node Classification Top-1 Accuracy (%)",
                "Neighborhood Aggregation Latency (ms)",
            ]
            primary_m = "Area Under Precision-Recall Curve (AUPRC)"
            baselines = [
                "Graph Convolutional Network (GCN)",
                "Graph Attention Network (GATv2)",
                "GraphSAGE Neighbor Sampling",
                "Standard Message Passing NN",
            ]
            method_families = [
                "Hierarchical Subgraph Pooling",
                "Spectral Graph Filtering",
                "Relational Memory-Bounded Transformers",
            ]
            math_objs = [
                "Normalized Graph Laplacian L = I - D^{-1/2}AD^{-1/2}",
                "Weisfeiler-Lehman 1-WL Color Refinement",
                "Dirichlet Energy E(X)",
                "Spectral Filter Eigenvalues λ_k",
            ]
            stat_tests = [
                "Paired t-test over K-Fold Cross Validation",
                "Wilcoxon Signed-Rank Test",
                "DerSimonian-Laird Meta-Analysis",
            ]
            figs = [
                "Precision-Recall Frontier under Topological Imbalance",
                "Graph Neighborhood Aggregation Depth Response",
                "Graph Neighborhood Representation t-SNE Embedding",
            ]
            requires_theorem = False

        elif dom_lower == "federated" or any(
            k in t_lower for k in ["federat", "client drift", "fl", "decentralized"]
        ):
            task = TaskType.FEDERATED_COORDINATION
            modality = (
                DataModality.IMAGE_VOLUMETRIC
                if any(
                    k in t_lower for k in ["image", "vision", "medical", "mri", "ct"]
                )
                else DataModality.TABULAR_HETEROGENEOUS
            )
            domain_name = "federated"
            subdomain = "Decentralized Optimization & Privacy-Preserving ML"
            paradigm = (
                ResearchParadigm.SYSTEMS_OPTIMIZATION
                if "optimization" in t_lower
                or "communication" in t_lower
                or "latency" in t_lower
                else ResearchParadigm.THEORETICAL_ALGORITHMIC
            )
            metrics = [
                "Global Model Accuracy (%)",
                "Client Drift Divergence (L2)",
                "Communication Rounds to Convergence",
                "Privacy Budget Epsilon (ε)",
            ]
            primary_m = "Global Model Accuracy (%)"
            baselines = [
                "FedAvg (McMahan et al.)",
                "FedProx (Li et al.)",
                "SCAFFOLD (Karimireddy et al.)",
                "Local Independent Training",
            ]
            method_families = [
                "Stochastic Controlled Averaging",
                "Adaptive Client Clustering",
                "Proximal Regularized Aggregation",
            ]
            math_objs = [
                "Client Loss Functions f_i(w)",
                "Stochastic Gradient Variance σ_k^2",
                "Consensus Error ||w_t - w*||^2",
                "Privacy Budget (ε, δ)",
            ]
            stat_tests = [
                "Paired Student's t-test across Seeds",
                "Wilcoxon Signed-Rank Test",
                "DerSimonian-Laird Meta-Analysis",
            ]
            figs = [
                "Convergence Trajectories across Communication Rounds",
                "Client Heterogeneity Drift Heatmap",
                "Communication vs Accuracy Pareto Frontier",
                "Ablation over Non-IID Dirichlet Alpha",
            ]
            requires_theorem = True

        elif dom_lower == "vision" or any(
            k in t_lower
            for k in [
                "medical image",
                "segmentation",
                "diffusion prior",
                "vision transformer",
                "vit",
                "resnet",
                "unet",
                "pixel",
                "convolution",
                "spatial attention",
            ]
        ):
            task = (
                TaskType.OBJECT_DETECTION_SEGMENTATION
                if any(
                    k in t_lower
                    for k in ["segment", "lesion", "tumor", "organ", "image", "vision"]
                )
                else TaskType.CLASSIFICATION
            )
            modality = DataModality.IMAGE_VOLUMETRIC
            domain_name = "vision"
            subdomain = "Diagnostic Segmentation & Spatial Representation"
            paradigm = ResearchParadigm.APPLIED_DOMAIN_STUDY
            metrics = [
                "Dice Similarity Coefficient (DSC %)",
                "Intersection over Union (IoU %)",
                "Area Under ROC Curve (AUROC %)",
                "Expected Calibration Error (ECE %)",
            ]
            primary_m = "Dice Similarity Coefficient (DSC %)"
            baselines = [
                "3D nnU-Net Standard Pipeline",
                "Vision Transformer (ViT)",
                "ResNet-50 Feature Backbone",
                "Dense V-Net Baseline",
            ]
            method_families = [
                "Uncertainty-Guided Attention Refinement",
                "Boundary-Preserving Topological Loss",
                "Cross-Domain Contrastive Calibration",
            ]
            math_objs = [
                "Generalized Dice Loss L_Dice",
                "Hausdorff Distance Sup_{x} Inf_{y} ||x - y||",
                "Calibration Reliability Curve ECE",
                "Inter-Observer Fleiss' Kappa",
            ]
            stat_tests = [
                "Non-Parametric Wilcoxon Signed-Rank Test",
                "Delong Test for AUROC Differences",
                "DerSimonian-Laird Random-Effects Meta-Analysis",
            ]
            figs = [
                "Qualitative Multi-Slice Segmentation Overlays",
                "ROC and Precision-Recall Curves",
                "Calibration Reliability Diagrams",
                "Inter-Site Robustness Box Plots",
            ]
            requires_theorem = False

        elif (
            dom_lower == "time_series"
            or any(
                k in t_lower
                for k in [
                    "forecast",
                    "time series",
                    "timeseries",
                    "temporal",
                    "autoregress",
                    "trend",
                    "seasonality",
                    "traffic",
                    "sensor",
                ]
            )
        ) and not any(
            k in t_lower for k in ["graph", "gnn", "node", "edge", "text", "language"]
        ):
            task = TaskType.TIMESERIES_FORECASTING
            modality = DataModality.MULTIVARIATE_TIME_SERIES
            domain_name = "time_series"
            subdomain = "Multivariate Autoregressive & State-Space Modeling"
            paradigm = (
                ResearchParadigm.EMPIRICAL_BENCHMARK
                if "benchmark" in t_lower
                else ResearchParadigm.METHODOLOGICAL_COMPARISON
            )
            metrics = [
                "Mean Absolute Error (MAE)",
                "Root Mean Squared Error (RMSE)",
                "Continuous Ranked Probability Score (CRPS)",
                "Mean Absolute Percentage Error (MAPE %)",
            ]
            primary_m = "Mean Absolute Error (MAE)"
            baselines = [
                "AutoARIMA / Classical VAR",
                "DeepAR Recurrent Network",
                "PatchTST Channel-Independent Transformer",
                "DLinear Decomposition Baseline",
            ]
            method_families = [
                "Multi-Scale Patch Tokenization",
                "Frequency-Domain Fourier Operators",
                "Dynamic Lag-Attention Networks",
            ]
            math_objs = [
                "Temporal Auto-Covariance Matrix Γ(τ)",
                "Multivariate State Transition Matrix A",
                "Spectral Density S(ω)",
                "Forecast Horizon Forecast Error e_{t+h}",
            ]
            stat_tests = [
                "Diebold-Mariano Forecast Accuracy Test",
                "Paired t-test across Horizon Steps",
                "DerSimonian-Laird Random-Effects Meta-Analysis",
            ]
            figs = [
                "Multi-Horizon Forecast Trajectories with 95% Confidence Intervals",
                "Residual Error Distribution Across Timesteps",
                "Horizon-Wise MAE Degradation Curves",
                "Patch Length Sensitivity Heatmap",
            ]
            requires_theorem = False

        elif dom_lower == "nlp" or any(
            k in t_lower
            for k in [
                "nlp",
                "language",
                "transformer",
                "llm",
                "attention",
                "token",
                "prompt",
                "peft",
                "lora",
                "retrieval",
                "rag",
            ]
        ):
            task = (
                TaskType.LANGUAGE_MODELING
                if any(
                    k in t_lower
                    for k in ["generat", "causal", "gpt", "pretrain", "causal language"]
                )
                else TaskType.CLASSIFICATION
            )
            modality = DataModality.NATURAL_LANGUAGE_TEXT
            domain_name = "nlp"
            subdomain = "Efficient Parameter Tuning & Sequence Modeling"
            paradigm = ResearchParadigm.EMPIRICAL_BENCHMARK
            metrics = [
                "Macro F1 Score (%)",
                "Top-1 Accuracy (%)",
                "Active Parameter Count & Memory Footprint (MB)",
            ]
            primary_m = "Macro F1 Score (%)"
            baselines = [
                "Full Fine-Tuning Baseline",
                "Low-Rank Adaptation (LoRA)",
                "Prefix-Tuning / Prompt-Tuning",
                "Standard INT8 Static Quantization",
            ]
            method_families = [
                "Dynamic Low-Rank Projections",
                "Sub-Quadratic Linear Attention",
                "Layer-Wise Sparse Adaptation",
            ]
            math_objs = [
                "Key-Query Attention Matrix softmax(QK^T / sqrt(d))",
                "Low-Rank Decomposition W_0 + B*A",
                "Entropy Residual H(p, q)",
                "Gradient Orthogonality Bound",
            ]
            stat_tests = [
                "Bootstrap Resampling Significance Test",
                "Paired t-test on Macro F1",
                "DerSimonian-Laird Meta-Analysis",
            ]
            figs = [
                "Accuracy vs Active Parameter Footprint Trade-off",
                "Per-Layer Attention Weight Entropy Heatmap",
                "Token-Wise Loss Convergence Curves",
                "Rank Dimension (r) Sensitivity Plot",
            ]
            requires_theorem = False

        elif dom_lower == "physics_surrogate" or any(
            k in t_lower
            for k in [
                "physics",
                "pde",
                "pinn",
                "operator",
                "navier",
                "fluid",
                "mechanics",
                "differential",
                "helmholtz",
                "darcy",
            ]
        ):
            task = TaskType.PDE_OPERATOR_LEARNING
            modality = DataModality.SPATIOTEMPORAL_GRID
            domain_name = "physics_surrogate"
            subdomain = "Nonlinear Partial Differential Equation Surrogates"
            paradigm = ResearchParadigm.THEORETICAL_ALGORITHMIC
            metrics = [
                "Relative L2 Spectral Error (%)",
                "Conservation Law Residual Loss",
                "Max Absolute Pointwise Error",
                "Inference Speedup Factor over Solver (x)",
            ]
            primary_m = "Relative L2 Spectral Error (%)"
            baselines = [
                "Fourier Neural Operator (FNO-2D)",
                "DeepONet Dual-Branch Operator",
                "Physics-Informed Neural Network (PINN)",
                "Classical High-Order RK4 Solver",
            ]
            method_families = [
                "Hamiltonian-Conserving Neural Operators",
                "Adaptive Wavelet Discretization",
                "Physics-Constrained Residual Networks",
            ]
            math_objs = [
                "Differential Operator Residual N[u] - f = 0",
                "Energy Conservation Invariant H(q, p)",
                "Sobolev Norm ||u||_{H^s}",
                "Spectral Truncation Error Bound",
            ]
            stat_tests = [
                "Paired Spectral Norm Difference Test",
                "DerSimonian-Laird Random-Effects Meta-Analysis",
            ]
            figs = [
                "2D Solution Field Contour & Pointwise Error Maps",
                "Conservation Invariant Trajectory over Time",
                "Spectral Energy Spectrum vs Wavenumber",
                "PDE Residual Convergence Profile",
            ]
            requires_theorem = True

        else:
            task = (
                TaskType.CLASSIFICATION if "classif" in t_lower else TaskType.REGRESSION
            )
            modality = (
                DataModality.TABULAR_HETEROGENEOUS
                if any(
                    k in t_lower for k in ["tabular", "credit", "financial", "clinical"]
                )
                else DataModality.IMAGE_VOLUMETRIC
            )
            domain_name = "Applied Machine Learning & Neural Computation"
            subdomain = "Adaptive Learning & Optimization Architecture"
            paradigm = ResearchParadigm.METHODOLOGICAL_COMPARISON
            metrics = [
                "Classification Accuracy / Task Score (%)",
                "F1 Macro Score (%)",
                "Model Memory Footprint (MB)",
                "Inference Latency (ms)",
            ]
            primary_m = "Classification Accuracy / Task Score (%)"
            baselines = [
                "Standard Full-Precision Baseline",
                "Regularized Competitive Model",
                "Lightweight Pruned Architecture",
            ]
            method_families = [
                "Adaptive Representation Learning",
                "Regularized Stochastic Gradient Formulation",
                "Dynamic Feature Alignment",
            ]
            math_objs = [
                "Empirical Risk Minimization R(θ)",
                "Generalization Bound E[R(θ) - R_emp(θ)]",
                "Loss Lipschitz Constant L",
                "Hessian Eigenvalue Distribution",
            ]
            stat_tests = [
                "Paired Student's t-test",
                "DerSimonian-Laird Random-Effects Meta-Analysis",
            ]
            figs = [
                "Empirical Accuracy & Generalization Comparison",
                "Loss Convergence Curves Across Iterations",
                "Latency vs Performance Trade-off",
                "Hyperparameter Sensitivity Analysis",
            ]
            requires_theorem = (
                "bound" in t_lower or "theory" in t_lower or "convergence" in t_lower
            )

        # Clean, domain-appropriate method acronym and full name generation
        question_stop_words = {
            "and",
            "of",
            "the",
            "for",
            "in",
            "with",
            "under",
            "using",
            "on",
            "a",
            "an",
            "to",
            "via",
            "can",
            "how",
            "what",
            "why",
            "which",
            "does",
            "do",
            "is",
            "are",
            "could",
            "would",
            "should",
            "improve",
            "enhance",
            "optimize",
            "remain",
            "accurate",
            "achieve",
            "investigating",
            "evaluating",
        }
        words = [
            w
            for w in re.findall(r"[A-Za-z]+", topic)
            if w.lower() not in question_stop_words
        ]

        # Domain-aware deterministic naming
        if any(
            k in t_lower
            for k in [
                "rag",
                "retrieval",
                "factual",
                "factuality",
                "hallucination",
                "question answering",
            ]
        ):
            acronym = "Ada-RAG"
            full_name = "Adaptive Faithful Retrieval-Augmented Generation Model"
        elif any(
            k in t_lower for k in ["peft", "parameter-efficient", "adapter", "lora"]
        ):
            acronym = "Ada-PEFT"
            full_name = "Adaptive Parameter-Efficient Language Adapter"
        elif any(
            k in t_lower for k in ["forecast", "time series", "timeseries", "temporal"]
        ):
            acronym = "TempShift-Net"
            full_name = "Adaptive Temporal Shift Forecaster"
        elif any(k in t_lower for k in ["vibration", "machinery", "bearing", "fault"]):
            acronym = "VibroDiag-Net"
            full_name = "Multi-Scale Vibration Anomaly Classifier"
        elif any(k in t_lower for k in ["graph", "gnn", "topology", "imbalance"]):
            acronym = "TopoGNN"
            full_name = "Topological Imbalanced Graph Neural Network"
        elif any(k in t_lower for k in ["federat", "client drift", "fl"]):
            acronym = "FedAdapt-Net"
            full_name = "Adaptive Decentralized Consensus Architecture"
        elif any(k in t_lower for k in ["pinn", "physics", "wavelet", "surrogate"]):
            acronym = "Wavelet-PINN"
            full_name = "Physics-Informed Wavelet Surrogate Network"
        else:
            if len(words) >= 3:
                acronym = "".join([w[0].upper() for w in words[:3]]) + "-Net"
            elif len(words) == 2:
                acronym = words[0][:3].upper() + "-" + words[1][:3].capitalize()
            else:
                acronym = "Nova-" + (words[0][:4].capitalize() if words else "Model")
            full_name = (
                f"Adaptive {' '.join(w.capitalize() for w in words[:3])} Architecture"
                if words
                else "Adaptive Representation Framework"
            )

        problem_characteristics = [
            f"Primary Task: {task.value.replace('_', ' ').title()}",
            f"Input Modality: {modality.value.replace('_', ' ').title()}",
            f"Scientific Focus: {subdomain}",
            f"Primary Evaluation Metric: {primary_m}",
        ]

        pub_conventions = [
            "IEEE Transactions Standard Double-Blind Layout",
            f"Target Manuscript: {target_format.replace('_', ' ').title()}",
            "Deterministic Multi-Seed (k>=5) Statistical Validation",
            "DerSimonian-Laird Random-Effects Meta-Analysis for Effect Sizes",
        ]

        return TopicResearchProfile(
            research_question=topic,
            domain=domain_name,
            subdomain=subdomain,
            task_type=task,
            research_paradigm=paradigm,
            data_modality=modality,
            problem_characteristics=problem_characteristics,
            candidate_metrics=metrics,
            primary_metric=primary_m,
            candidate_datasets=[
                d.name
                for d in DatasetFinder.discover_candidates(topic, domain_name, limit=3)
            ]
            or ["Canonical Benchmark Dataset"],
            candidate_baselines=baselines,
            candidate_method_families=method_families,
            mathematical_objects=math_objs,
            likely_statistical_tests=stat_tests,
            figure_candidates=figs,
            publication_conventions=pub_conventions,
            requires_formal_theorem=requires_theorem,
            model_acronym_suggestion=acronym,
            model_full_name_suggestion=full_name,
            inferred_domain_priors={"domain": domain_name, "subdomain": subdomain},
        )
