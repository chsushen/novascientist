# NovaScientist 🔬
### *Autonomous Research-to-Publication Agent & Local Hardware Benchmarking Framework*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-24%20passed%20(100%25)-brightgreen.svg)](tests/)
[![Hardware Invariant](https://img.shields.io/badge/Hardware-CPU%20%2F%20Apple%20Silicon%20Invariant-orange.svg)]()
[![Export](https://img.shields.io/badge/Export-Overleaf%20ZIP%20%2B%20PDF-blueviolet.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**NovaScientist** is an autonomous AI agent framework designed to turn **any scientific research query or problem statement** into a complete, benchmarked, and typeset IEEE Transactions research manuscript with publication-ready figures and citations.

Running entirely on local commodity hardware (standard x86 CPUs or Apple Silicon ARM64), NovaScientist orchestrates an end-to-end scholarly pipeline: it retrieves active peer-reviewed literature with verified DOIs, audits AST dataflow integrity, executes multi-seed hardware micro-benchmarks, synthesizes empirical results via random-effects meta-analysis, and compiles an Overleaf-ready LaTeX package and PDF in seconds.

---

## 🏛️ System Architecture

```
                          [ Any Research Topic or Hypothesis ]
                                           │
          ┌────────────────────────────────┴────────────────────────────────┐
          ▼                                                                 ▼
[ 1. Literature Discovery ]                                     [ 2. Domain Dispatcher ]
Live CrossRef / OpenAlex APIs                                   6 Computational Domains
100% Verified DOIs & BibTeX                                     Canonical Dataset Matcher
          │                                                                 │
          └────────────────────────────────┬────────────────────────────────┘
                                           ▼
                               [ 3. AST Guard & Audit ]
                              Static Dataflow Verification
                               (Zero Train-Val Leakage)
                                           │
                                           ▼
                           [ 4. Local Hardware Benchmarks ]
                            Deterministic k=5 CPU Passes
                           DerSimonian-Laird Meta-Analysis
                                           │
          ┌────────────────────────────────┴────────────────────────────────┐
          ▼                                                                 ▼
[ 5. Vector Plotting ]                                          [ 6. IEEEtran Assembler ]
Convergence & Pareto Dynamics                                   5–8 Page IEEE Transactions
Vector Figures (PDF & PNG)                                      Domain Math & Real Hardware Specs
          │                                                                 │
          └────────────────────────────────┬────────────────────────────────┘
                                           ▼
                           [ 7. Reviewer Swarm Audit ]
                            Statistical Power Assertion
                             Scientific Rhetoric Linter
                                           │
          ┌────────────────────────────────┴────────────────────────────────┐
          ▼                                                                 ▼
[ 8. Tectonic Compiler ]                                        [ 9. Overleaf Packager ]
XeTeX Engine Compilation                                        Self-Contained ZIP Bundle
Produces Publication PDF                                        Ready for 1-Click Import
```

---

## 🌐 Supported Research Domains

NovaScientist dynamically classifies incoming research topics into one of 6 computational domains and automatically pairs them with canonical evaluation benchmark datasets:

| Domain | Example Research Topics | Canonical Benchmarks |
| :--- | :--- | :--- |
| **Physics Surrogates & PINNs** | Neural operators, Darcy flow, Navier-Stokes, Burgers shock | *Darcy Flow, Burgers Shock, Allen-Cahn Phase Field* |
| **Graph Neural Networks** | Relational topologies, message-passing, spatial graphs | *METR-LA, PeMS-BAY, OGB-MolHIV, Cora Citation* |
| **Computer Vision** | Low-compute transformers, spatial patch attention, edge vision | *ImageNet-1K, CIFAR-100-C, ADE20K Semantic* |
| **NLP & Sequence Models** | Sub-linear attention, memory-bounded LLM caching, sequence models | *GLUE Benchmark, WikiText-103, C4 Multi-Domain* |
| **Time-Series Forecasting** | Multivariate sensor dynamics, temporal lag forecasting, spectral models | *Electricity (ECL), Weather (MPI-BGC), Exchange-Rate* |
| **Tabular & Heterogeneous** | Gradient-boosted feature embeddings, categorical risk analysis | *Higgs Boson ML, Adult Census, California Housing* |

---

## 💻 How to Run on Your PC

Follow these steps to set up and run NovaScientist locally on macOS, Linux, or Windows (WSL2):

### Step 1: Clone the Repository
```bash
git clone https://github.com/chsushen/novascientist.git
cd novascientist
```

### Step 2: Set Up Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: (Optional) Configure Environment Variables
NovaScientist runs **100% locally with zero required external API keys** using deterministic multi-seed micro-benchmarks and open scholarly APIs.

If you wish to configure optional settings:
```bash
cp .env.example .env
```
Optional settings in `.env`:
- `GEMINI_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY` *(for custom external LLM agent extensions)*
- `CROSSREF_MAILTO` *(your email for polite CrossRef API rate limits)*

---

### Step 4: Run via Command-Line Interface (CLI)

Generate a complete research paper for any topic with one command:

```bash
# Example 1: Generate an anonymous double-blind conference submission
python cli.py \
  --topic "Low-Rank Dynamic Graph Attention for Smart Disaster Resilience and Evacuation Forecasting" \
  --anonymous \
  --output ./paper.pdf
```

```bash
# Example 2: Generate with custom human authorship
python cli.py \
  --topic "Physics-Informed Dynamic Neural Surrogates under Bounded Memory" \
  --author "Dr. Jane Doe" \
  --affiliation "Department of Electrical Engineering and Computer Science, MIT" \
  --email "janedoe@mit.edu" \
  --output ~/Desktop/physics_paper.pdf
```

#### Available CLI Arguments:
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--topic` | `str` | *Required* | Research title or topic prompt |
| `--anonymous` | `flag` | `False` | Enforces standard IEEE double-blind author masking |
| `--author` | `str` | `Anonymous Author(s)` | Primary author name |
| `--affiliation` | `str` | `Affiliation Withheld...` | Institutional affiliation |
| `--email` | `str` | `anonymous@...` | Corresponding author email |
| `--seeds` | `int` | `5` | Number of deterministic evaluation seeds ($k \ge 5$) |
| `--output`, `-o` | `str` | `None` | Destination path for the compiled `.pdf` |
| `--output-dir` | `str` | `./dist` | Target directory for Overleaf `.zip` packages |

---

### Step 5: Run via Interactive Streamlit Web GUI

Launch the web interface for visual stage tracking, vector figure inspectors, and instant Overleaf ZIP downloads:

```bash
streamlit run app.py
```
Or launch via the convenience script:
```bash
./run.sh
```
Open **http://localhost:8501** in your browser.

---

### Step 6: Standalone Reproducibility Benchmark Runner

To independently execute the hardware micro-benchmarking engine, log memory/latency, and verify the DerSimonian-Laird statistical meta-analysis without generating a full paper:

```bash
python reproduce_benchmarks.py
```

---

## 🎯 Target Publication Venue Matching

NovaScientist includes an automated **Venue Matcher** that evaluates domain alignment, acceptance rates, and impact factors to recommend top-tier IEEE/ACM publication venues:

1. **IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI)** — IF: $23.6$, Acceptance: $11.2\%$
2. **ACM SIGKDD Conference on Knowledge Discovery and Data Mining (ACM KDD)** — $h5$: $142$, Acceptance: $15.1\%$
3. **IEEE Transactions on Neural Networks and Learning Systems (IEEE TNNLS)** — IF: $10.4$, Acceptance: $13.5\%$
4. **IEEE Transactions on Computers (IEEE TC)** — IF: $3.7$, Acceptance: $18.0\%$

---

## 📦 Importing into Overleaf

1. Locate the generated bundle in `dist/novascientist_<topic_slug>.zip`.
2. Open [Overleaf](https://www.overleaf.com) and click **New Project** $\rightarrow$ **Upload Project**.
3. Upload the `.zip` archive.
4. Set the compiler to **pdfLaTeX** or **XeLaTeX** in Project Settings and click **Recompile**.

---

## 🧪 Running Tests

Run the full unit and integration test suite:
```bash
PYTHONPATH=. pytest tests/ -v
```

---

## ⚖️ License

This project is open-source and licensed under the [Apache 2.0 License](LICENSE).
