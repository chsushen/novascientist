"""NovaScientist Dynamic Baseline Selection & Experiment Method Engine.

Derives task-appropriate, defensible baseline comparison sets and structured
ExperimentMethodSpecs dynamically from research profiles and literature evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from backend.core.literature_advisor import LiteratureSynthesisReport
from backend.core.topic_profile import DataModality, TaskType, TopicResearchProfile


@dataclass
class ExperimentMethodSpec:
    """Rigorous execution specification for the proposed scientific method."""

    method_id: str
    method_name: str
    task_type: TaskType
    architecture_definition: str
    input_modality: DataModality
    training_procedure: str
    inference_procedure: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    loss_objective: str = ""
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    telemetry_schema: list[str] = field(
        default_factory=lambda: [
            "accuracy",
            "memory_mb",
            "latency_ms",
            "throughput",
            "compression_ratio",
        ]
    )
    compatible_dataset_schema: dict[str, Any] = field(default_factory=dict)
    execution_key: str = "proposed_mb_qgt"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_type"] = (
            self.task_type.value
            if isinstance(self.task_type, TaskType)
            else str(self.task_type)
        )
        d["input_modality"] = (
            self.input_modality.value
            if isinstance(self.input_modality, DataModality)
            else str(self.input_modality)
        )
        return d


@dataclass
class BaselineSpecification:
    """Detailed specification for a single comparative baseline."""

    baseline_id: str
    name: str
    display_name: str
    category: str  # 'uncompressed_canonical', 'state_of_the_art', 'lightweight_pruned', 'classical_reference'
    source_citation: str
    selection_rationale: str
    expected_tradeoff: str
    execution_key: str  # Key mapping to empirical metrics dict
    is_corpus_grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineComparisonSuite:
    """Complete structured comparative baseline suite for experimental execution."""

    proposed_method_spec: ExperimentMethodSpec
    baselines: list[BaselineSpecification] = field(default_factory=list)
    suite_rationale: str = ""

    @property
    def proposed_method_name(self) -> str:
        return self.proposed_method_spec.method_name

    @property
    def proposed_execution_key(self) -> str:
        return self.proposed_method_spec.execution_key

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["proposed_method_spec"] = self.proposed_method_spec.to_dict()
        d["baselines"] = [b.to_dict() for b in self.baselines]
        return d


class DynamicBaselineSelector:
    """Selects and parameterizes defensible comparison baselines and execution specs dynamically."""

    def __init__(self) -> None:
        pass

    @classmethod
    def create_method_spec(
        cls,
        profile: TopicResearchProfile,
    ) -> ExperimentMethodSpec:
        """Construct a structured, task-grounded experiment specification for the proposed method."""
        m_name = (
            profile.model_full_name_suggestion
            or f"Proposed {profile.model_acronym_suggestion}"
        )
        task = profile.task_type
        modality = profile.data_modality

        if task in (
            TaskType.LANGUAGE_MODELING,
            TaskType.SEQUENCE_GENERATION,
            TaskType.GENERATION,
        ):
            arch_def = f"Autoregressive Transformer with {profile.model_acronym_suggestion} Low-Rank Projection Adaptation"
            train_proc = "AdamW optimizer (lr=2e-4, weight_decay=0.01) with linear warmup and cosine decay over causal token batches."
            infer_proc = "Autoregressive generation with KV-cache optimization and parameterized low-rank weight projections."
            loss_obj = r"\mathcal{L} = -\sum_t \log P(x_t \mid x_{<t}; \theta_0 + \Delta\theta)"
            hparams = {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "batch_size": 32,
                "max_seq_len": 512,
            }
        elif task == TaskType.TIMESERIES_FORECASTING:
            arch_def = f"Channel-Independent Multi-Horizon Temporal Forecaster with {profile.model_acronym_suggestion} Lag Modules"
            train_proc = "Lookback windowed mini-batch optimization (MSE loss) across rolling validation partitions."
            infer_proc = "Direct multi-step forecast rollout with residual autoregressive correction."
            loss_obj = r"\mathcal{L} = \frac{1}{H} \sum_{h=1}^H \| X_{t+h} - \hat{X}_{t+h} \|_2^2"
            hparams = {
                "lookback_window": 96,
                "forecast_horizon": 24,
                "patch_len": 16,
                "stride": 8,
            }
        elif task == TaskType.GRAPH_REASONING:
            arch_def = f"Relational Graph Neural Architecture with {profile.model_acronym_suggestion} Spatial Message Passing"
            train_proc = "Neighborhood-sampled mini-batch cross-entropy optimization across inductive split folds."
            infer_proc = (
                "Layer-wise cached message passing over graph adjacency buffers."
            )
            loss_obj = r"\mathcal{L} = -\sum_{v \in \mathcal{V}_{\text{train}}} y_v \log \hat{y}_v + \lambda \|\Theta\|_F^2"
            hparams = {
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.2,
                "neighbor_samples": [15, 10, 5],
            }
        elif task == TaskType.FEDERATED_COORDINATION:
            arch_def = f"Decentralized Federated Optimization Model with {profile.model_acronym_suggestion} Consensus Operator"
            train_proc = "Local client stochastic gradient steps (E=5 epochs) with server-side proximal aggregation."
            infer_proc = "Client-side evaluation over private non-IID test partitions."
            loss_obj = (
                r"\min_w \frac{1}{K} \sum_{k=1}^K f_k(w) + \frac{\mu}{2} \|w - w^t\|^2"
            )
            hparams = {
                "num_clients": 100,
                "client_sample_rate": 0.1,
                "local_epochs": 5,
                "mu_prox": 0.01,
            }
        elif task == TaskType.PDE_OPERATOR_LEARNING:
            arch_def = f"Continuous Physics Operator with {profile.model_acronym_suggestion} Residual Discretization"
            train_proc = "Collocation point physics residual minimization combined with data-driven boundary matching."
            infer_proc = "Mesh-independent continuous forward evaluation across domain grid points."
            loss_obj = r"\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}}"
            hparams = {"modes": 16, "width": 64, "collocation_points": 10000}
        else:
            arch_def = f"Adaptive Neural Classifier with {profile.model_acronym_suggestion} Representation Discretization"
            train_proc = (
                "Cross-entropy loss optimization with multi-seed data fold isolation."
            )
            infer_proc = "Batched forward pass over scaled evaluation partitions."
            loss_obj = r"\mathcal{L} = -\sum_i y_i \log \hat{y}_i"
            hparams = {"hidden_dim": 128, "batch_size": 64, "learning_rate": 1e-3}

        return ExperimentMethodSpec(
            method_id=f"method_{profile.model_acronym_suggestion.lower().replace('-', '_')}",
            method_name=m_name,
            task_type=task,
            architecture_definition=arch_def,
            input_modality=modality,
            training_procedure=train_proc,
            inference_procedure=infer_proc,
            hyperparameters=hparams,
            loss_objective=loss_obj,
            resource_requirements={
                "max_memory_mb": 128.0,
                "device_target": "commodity_workstation",
            },
            compatible_dataset_schema={"modality": modality.value, "task": task.value},
            execution_key="proposed_mb_qgt",
        )

    @classmethod
    def select_baselines(
        cls,
        profile: TopicResearchProfile,
        literature_report: LiteratureSynthesisReport | None = None,
    ) -> BaselineComparisonSuite:
        """Select a domain- and task-appropriate baseline suite and proposed method spec."""
        task = profile.task_type
        proposed_spec = cls.create_method_spec(profile)

        baselines: list[BaselineSpecification] = []

        if (
            task
            in (
                TaskType.LANGUAGE_MODELING,
                TaskType.SEQUENCE_GENERATION,
                TaskType.GENERATION,
            )
            or "nlp" in profile.domain.lower()
        ):
            baselines = [
                BaselineSpecification(
                    baseline_id="base_01",
                    name="Full Fine-Tuning Baseline",
                    display_name="Full Fine-Tuning (All Parameters Active)",
                    category="uncompressed_canonical",
                    source_citation="Devlin et al., NAACL 2019",
                    selection_rationale="Serves as the unconstrained upper-bound representation capacity benchmark.",
                    expected_tradeoff="High task accuracy with maximal parameter storage and gradient memory.",
                    execution_key="dense_baseline",
                ),
                BaselineSpecification(
                    baseline_id="base_02",
                    name="Static INT8 Quantization",
                    display_name="Post-Training INT8 Quantization",
                    category="state_of_the_art",
                    source_citation="Dettmers et al., NeurIPS 2022",
                    selection_rationale="Standard industry post-training quantization baseline for transformer layers.",
                    expected_tradeoff="Moderate memory reduction with potential outlier feature degradation.",
                    execution_key="post_int8",
                ),
                BaselineSpecification(
                    baseline_id="base_03",
                    name="Low-Rank Parameter-Efficient Adaptation",
                    display_name="Low-Rank Adapter (LoRA r=8)",
                    category="lightweight_pruned",
                    source_citation="Hu et al., ICLR 2022",
                    selection_rationale="Evaluates whether low-rank parameter subspace matches proposed architectural efficiency.",
                    expected_tradeoff="Compact trainable footprint with fixed projection rank constraints.",
                    execution_key="sparse_gnn",
                ),
            ]
        elif (
            task == TaskType.TIMESERIES_FORECASTING or "time" in profile.domain.lower()
        ):
            baselines = [
                BaselineSpecification(
                    baseline_id="base_01",
                    name="Full Multichannel Autoregressive Baseline",
                    display_name="Dense VAR / Autoregressive Transformer",
                    category="uncompressed_canonical",
                    source_citation="Nie et al., ICLR 2023",
                    selection_rationale="Canonical full-rank autoregressive model benchmark across all forecast horizons.",
                    expected_tradeoff="Captures cross-channel dynamics but incurs quadratic computational complexity.",
                    execution_key="dense_baseline",
                ),
                BaselineSpecification(
                    baseline_id="base_02",
                    name="Static Decoupled Linear Model (DLinear)",
                    display_name="DLinear Trend-Seasonal Decomposition",
                    category="state_of_the_art",
                    source_citation="Zeng et al., AAAI 2023",
                    selection_rationale="High-efficiency linear decomposition baseline known for robust time-series forecasting.",
                    expected_tradeoff="Extremely fast inference with potential expressive limits on non-stationary shifts.",
                    execution_key="post_int8",
                ),
                BaselineSpecification(
                    baseline_id="base_03",
                    name="Decimated Temporal Lag Network",
                    display_name="Sparse Multi-Lag Recurrent Baseline",
                    category="lightweight_pruned",
                    source_citation="Salinas et al., IJF 2020",
                    selection_rationale="Standard autoregressive recurrent baseline with sampled lag connections.",
                    expected_tradeoff="Lower memory consumption but higher variance across multi-step horizons.",
                    execution_key="sparse_gnn",
                ),
            ]
        elif (
            task == TaskType.FEDERATED_COORDINATION
            or "federat" in profile.domain.lower()
        ):
            baselines = [
                BaselineSpecification(
                    baseline_id="base_01",
                    name="FedAvg Standard Aggregation",
                    display_name="Federated Averaging (McMahan et al.)",
                    category="uncompressed_canonical",
                    source_citation="McMahan et al., AISTATS 2017",
                    selection_rationale="The foundational baseline for all distributed federated optimization studies.",
                    expected_tradeoff="Simple parameter averaging but highly vulnerable to client drift under non-IID data.",
                    execution_key="dense_baseline",
                ),
                BaselineSpecification(
                    baseline_id="base_02",
                    name="FedProx Regularized Aggregation",
                    display_name="FedProx Proximal Aggregator",
                    category="state_of_the_art",
                    source_citation="Li et al., MLSys 2020",
                    selection_rationale="Standard proximal regularized benchmark for addressing heterogeneous client drift.",
                    expected_tradeoff="Improved stability at the cost of additional hyperparameter tuning.",
                    execution_key="post_int8",
                ),
                BaselineSpecification(
                    baseline_id="base_03",
                    name="SCAFFOLD Controlled Averaging",
                    display_name="SCAFFOLD Variance Reduction",
                    category="lightweight_pruned",
                    source_citation="Karimireddy et al., ICML 2020",
                    selection_rationale="Evaluates control-variate gradient correction to prevent local client drift.",
                    expected_tradeoff="Fast consensus convergence with doubled communication payload per round.",
                    execution_key="sparse_gnn",
                ),
            ]
        else:
            baselines = [
                BaselineSpecification(
                    baseline_id="base_01",
                    name="Dense Full-Precision Baseline",
                    display_name="Uncompressed FP32 Baseline",
                    category="uncompressed_canonical",
                    source_citation="Standard Deep Learning Baseline",
                    selection_rationale="Establishes uncompressed reference capacity and upper-bound task performance.",
                    expected_tradeoff="Maximum parameter capacity with high memory and latency footprint.",
                    execution_key="dense_baseline",
                ),
                BaselineSpecification(
                    baseline_id="base_02",
                    name="Static Post-Training Discretization",
                    display_name="Uniform INT8 Discretization",
                    category="state_of_the_art",
                    source_citation="Jacob et al., CVPR 2018",
                    selection_rationale="Standard post-training quantization baseline for evaluating compression degradation.",
                    expected_tradeoff="4x weight reduction with potential quantization noise in high-frequency modes.",
                    execution_key="post_int8",
                ),
                BaselineSpecification(
                    baseline_id="base_03",
                    name="Dynamic Sparsified Architecture",
                    display_name="Magnitude-Pruned Sparsified Baseline",
                    category="lightweight_pruned",
                    source_citation="Han et al., NeurIPS 2015",
                    selection_rationale="Evaluates structured parameter pruning as an alternative efficiency strategy.",
                    expected_tradeoff="Sparse memory footprint with unstructured indexing overhead.",
                    execution_key="sparse_gnn",
                ),
            ]

        suite_rationale = (
            f"The comparative suite for {profile.domain} evaluates {proposed_spec.method_name} "
            f"against canonical full-precision ({baselines[0].name}), industry discretization ({baselines[1].name}), "
            f"and structured parameter efficiency ({baselines[2].name}) across identical multi-seed data partitions."
        )

        return BaselineComparisonSuite(
            proposed_method_spec=proposed_spec,
            baselines=baselines,
            suite_rationale=suite_rationale,
        )
