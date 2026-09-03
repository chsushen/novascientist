# NovaScientist v2.0 🔬

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://novascientist-cqhrr8wptwmrzjksbr8pyw.streamlit.app/)
[![GitHub stars](https://img.shields.io/github/stars/chsushen/novascientist?style=social)](https://github.com/chsushen/novascientist)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20MPS%20%7C%20CPU-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/pytest-31%20passed%20(100%25)-brightgreen.svg)](tests/)
[![Format](https://img.shields.io/badge/Format-8--12%20Page%20IEEE%20Transactions-8B5CF6.svg)]()
[![Figures](https://img.shields.io/badge/Figures-5%20Publication%20Vectors%20(PDF%2FPNG)-059669.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Interactive Conversational Research Agent & Real PyTorch Hardware Benchmarking Suite for IEEE Publications**

NovaScientist v2.0 is an interactive, multi-agent AI research-to-publication engine that turns any scientific research query or problem statement into an authentic, empirically evaluated, and rigorously typeset 8–12 page IEEE Transactions journal paper.

Running on local hardware with automatic acceleration (NVIDIA CUDA, Apple Silicon MPS, or Multi-Core CPU), NovaScientist converses with the researcher to clarify intent, executes genuine PyTorch neural optimization runs across deterministic seeds ($k=5$), verifies mathematical invariants with formal proofs, generates a 5-figure vector plotting suite, and compiles an Overleaf-ready LaTeX package and publication PDF.

---

## 🏛️ System Architecture

```
                    [ Researcher Input / Question / Hypothesis ]
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │    1. Conversational Requirement Gathering    │
                 │   • Topic Refinement & Hypothesis Alignment   │
                 │   • Target Length (8–12p Journal vs 4p Conf)  │
                 │   • Hardware Execution Mode (Real vs Fast)    │
                 │   • Interactive Execution Plan Preview Card   │
                 └───────────────────────┬───────────────────────┘
                                         │  [User Approval]
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │       2. Scholarly Literature Agent           │
                 │   • Live CrossRef & OpenAlex Retrieval        │
                 │   • 100% Active, Verified DOIs (Zero Hal.)    │
                 │   • Structured Related-Work Taxonomy Table    │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │          3. AST Static Dataflow Gate          │
                 │   • Code Integrity & Leakage Verification     │
                 │   • Strict Pre-Split Scaler Auditing          │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │    4. Real PyTorch Hardware Training Sandbox  │
                 │   • Auto-Detect CUDA / Apple Silicon MPS / CPU│
                 │   • Train 4 Candidate Neural Architectures    │
                 │   • Save PyTorch Model Checkpoints (.pt)      │
                 │   • Log Empirical Losses, Latency, & RSS      │
                 │   • DerSimonian-Laird Random-Effects Analysis │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │       5. Scientific Vector Plotting Suite     │
                 │   • Fig 1: Architecture Block Diagram (Vector)│
                 │   • Fig 2: Convergence Loss & Accuracy Curves │
                 │   • Fig 3: Multi-Objective Pareto Frontier    │
                 │   • Fig 4: Component Module Ablation Bars     │
                 │   • Fig 5: 2D Hyperparameter Sensitivity Map  │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │    6. Deep Multi-Agent Journal Synthesis      │
                 │   • 10 Structured IEEE Transactions Sections  │
                 │   • Formal Proofs (Lemma 1, Theorems 1 & 2)   │
                 │   • Tables 1, 2, 3 & Algorithm Environment    │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │      7. Adversarial Reviewer Swarm Audit      │
                 │   • Statistical Power & Heterogeneity Check   │
                 │   • Scientific Rhetoric & Overclaim Linter    │
                 └───────────────────────┬───────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
       [ 8. Tectonic XeTeX Engine ]              [ 9. Publication Packager ]
        Compiles 8–12 Page IEEE PDF               Overleaf ZIP Archive + Checkpoints
```

---

## 🌟 Key Capabilities in v2.0

1. **Interactive Conversational Requirement Gathering**:
   - Step-by-step guided dialogue via Streamlit GUI and CLI chat mode.
   - Refines topic titles, identifies domain affinity, recommends canonical datasets, configures seed budgets, and generates an interactive execution plan for user review.

2. **Real PyTorch Hardware Training Sandbox**:
   - Auto-detects local acceleration: **NVIDIA CUDA**, **Apple Silicon Neural Engine (MPS)**, or **Multi-Core CPU**.
   - Trains candidate architectures (`ProposedMBQGT`, `DenseFP32Baseline`, `StaticINT8Baseline`, `SparseGNNBaseline`) using authentic loss functions, AdamW optimization, and learning rate schedulers.
   - Saves `.pt` model weights to `dist/experiments/checkpoints/` and detailed epoch histories to `dist/experiments/logs/`.

3. **8–12 Page Full IEEE Transactions Journal Synthesis**:
   - Generates an exhaustive 10-section manuscript formatted in double-column `IEEEtran.cls`.
   - **Section 1**: Introduction & Grand Computational Challenges.
   - **Section 2**: Related Work & Structured DOI Taxonomy Table.
   - **Section 3**: Theoretical Formulation & Mathematical Foundations (Lemma 1 [Dynamic Block Bounds], Theorem 1 [Bounded Discretization Variance], Theorem 2 [Stochastic Gradient Convergence], Proposition 1 [Cache Line Miss Bounds] with complete proofs).
   - **Section 4**: System Architecture & Algorithmic Pseudocode (`algorithm` / `algorithmicx`).
   - **Section 5**: Experimental Setup & Physical Hardware Telemetry.
   - **Section 6**: Empirical Benchmark Results & DerSimonian-Laird Random-Effects Meta-Analysis ($Q, \tau^2, I^2, Z$).
   - **Section 7**: Component Ablation Study (Table 2) & Hyperparameter Sensitivity (Table 3).
   - **Section 8**: In-Depth Technical Discussion (Memory Wall, Cache Locality, SIMD Vectorization, Failure Modes).
   - **Section 9**: Ethical Statement & AI-Assistance Acknowledgment (IEEE/ACM 2024+ Standards).
   - **Section 10**: Conclusion & Future Trajectories.

4. **Rich 5-Figure Publication Vector Suite**:
   - Generates dual publication-grade vector `.pdf` and high-DPI `.png` files:
     * **Fig 1 (`fig1_system_architecture`)**: System Dataflow & Quantization Engine Diagram.
     * **Fig 2 (`fig2_convergence_curves`)**: Dual-Panel Loss Decay & Validation Accuracy Saturation.
     * **Fig 3 (`fig3_pareto_frontier`)**: Multi-Objective Pareto Frontier (Latency vs. Peak RAM vs. Accuracy).
     * **Fig 4 (`fig4_ablation_study`)**: Component Module Contribution Ablation Bar Chart.
     * **Fig 5 (`fig5_sensitivity_heatmap`)**: 2D Hyperparameter Sensitivity Matrix across Quantization Depths and Cache Sizes.

5. **Target Publication Venue Matcher**:
   - Indexes and ranks top-tier venues (IEEE TPAMI, IEEE TNNLS, IEEE Access, ACM TODS, NeurIPS) by impact factor, h5-index, and typical review turnaround.

---

## 🌐 Supported Research Domains & Canonical Benchmarks

| Computational Domain | Representative Research Scope | Canonical Evaluation Datasets |
| :--- | :--- | :--- |
| **Physics Surrogates & PINNs** | Neural operators, Darcy flow, Navier-Stokes, shock fronts | *Darcy Flow Benchmark, Burgers Shock Dynamics, Allen-Cahn* |
| **Graph Neural Networks (GNNs)** | Relational topologies, dynamic attention, spatial transport | *METR-LA Sensor Network, PeMS-BAY Traffic, OGB-MolHIV, Cora* |
| **Low-Compute Computer Vision** | Patch attention, edge CNNs, robust classification | *ImageNet-1K, CIFAR-100-C Robustness, ADE20K Semantic* |
| **Sub-Linear NLP & LLMs** | Low-rank projection, memory-bounded KV caching, attention | *GLUE Benchmark Suite, WikiText-103, C4 Multi-Domain* |
| **Time-Series Forecasting** | Multivariate sensor dynamics, spatio-temporal lag models | *Electricity (ECL), Weather (MPI-BGC), Traffic PeMS* |
| **Tabular & Heterogeneous ML** | Gradient-boosted embeddings, risk stratification | *Higgs Boson ML, Adult Census, California Housing* |

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/chsushen/novascientist.git
cd novascientist

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Web GUI (Streamlit)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) to interact with the conversational research assistant, review execution plans, monitor live PyTorch training progress, explore the 5-figure carousel, and download the compiled IEEE PDF and `.pt` model weights.

### 3. Run via Interactive CLI Chat
```bash
python cli.py chat
# Or:
python cli.py --interactive
```

### 4. Run via Direct CLI Command
```bash
python cli.py run \
  --topic "Low-Rank Dynamic Graph Attention for Smart Disaster Resilience and Evacuation Forecasting" \
  --pages 8_12_pages_journal \
  --mode real \
  --seeds 5 \
  --output ~/Desktop/my_research_paper.pdf
```

### 5. Reproduce Benchmarks Standalone
```bash
python reproduce_benchmarks.py --mode real --seeds 5 --figures
```

### 6. Run Test Suite
```bash
PYTHONPATH=. python -m pytest tests/ -v
```

---

## 📦 Output Artifacts Structure

Upon execution, NovaScientist generates the following structured bundle in `dist/`:

```
dist/
├── novascientist_<topic>_v2.zip    # Overleaf-ready package (1-click upload)
├── experiments/
│   ├── checkpoints/
│   │   └── proposed_mb_qgt_weights.pt  # Trained PyTorch model weights
│   └── logs/
│       └── training_log.json           # Raw per-epoch training & validation history
└── workspace/
    ├── main.pdf                         # Compiled 8–12 page IEEE Transactions PDF
    ├── main.tex                         # Complete LaTeX source with 10 sections & proofs
    ├── references.bib                   # BibTeX with active CrossRef DOIs & dataset entries
    ├── IEEEtran.cls                     # Official IEEE Transactions LaTeX document class
    ├── artifacts/
    │   └── metrics.json                 # Multi-seed empirical metrics & meta-analysis
    └── figures/
        ├── fig1_system_architecture.pdf / .png
        ├── fig2_convergence_curves.pdf / .png
        ├── fig3_pareto_frontier.pdf / .png
        ├── fig4_ablation_study.pdf / .png
        └── fig5_sensitivity_heatmap.pdf / .png
```

---

## ⚖️ Academic Integrity & AI-Assistance Compliance

NovaScientist complies with IEEE and ACM 2024+ authorship guidelines:
- **Zero Hallucinated Citations**: All literature DOIs are queried directly against live CrossRef and OpenAlex indexes.
- **AST Dataflow Integrity**: Code pipelines are statically audited to guarantee zero test data leakage during pre-processing.
- **Explicit Disclosure**: Assembled manuscripts include a formal Section 9 *Ethical Statement and AI-Assistance Acknowledgment* identifying NovaScientist as the experimental orchestration and typesetting framework while attributing scientific intent to human researchers.

---

## 📄 License

NovaScientist is open-sourced under the [Apache License 2.0](LICENSE).
