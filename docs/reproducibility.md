# NovaScientist v2.0 — Reproducibility & Verification Guide

This document specifies the deterministic seed controls, test fixtures, hardware isolation invariants, and verification instructions for NovaScientist v2.0.

---

## 1. Environment Requirements & Installation

NovaScientist v2.0 runs on commodity hardware (macOS Apple Silicon, Linux CUDA, or Windows/Linux Multi-Core CPU).

```bash
# Clone the repository
git clone https://github.com/chsushen/novascientist.git
cd novascientist

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variable Configurations
- `SCHOLARLY_CONTACT_EMAIL`: Email contact string passed to scholarly APIs (default: `novascientist@research.org`).
- `SCHOLARLY_API_TIMEOUT`: Timeout in seconds for CrossRef/OpenAlex HTTP queries (default: `8.0`).

---

## 2. Deterministic Seed Controls

All multi-seed empirical evaluations and surrogate physics engines use deterministic pseudo-random number generator (PRNG) initialization:

- **Default Seed Progression**:
  $$s_i = 42 + 137 \times i \quad \text{for } i \in \{0, 1, 2, 3, 4\}$$
  $$(42, 179, 316, 453, 590)$$
- **PyTorch Invariance**:
  - `torch.manual_seed(seed)`
  - `np.random.seed(seed)`
  - `np.random.default_rng(seed)`
- **AST Dataflow Guard**:
  - `ASTGuard.enforce(code)` audits code AST to certify that feature transformations (e.g. `StandardScaler`) are strictly fitted on the training split before transforming the test split, guaranteeing zero test-set leakage.

---

## 3. Test Fixture Isolation & Verification Invariants

### 3.1 Test Fixture Isolation
- All canonical test fixture papers are housed in [`tests/fixtures/literature/fixtures.py`](../tests/fixtures/literature/fixtures.py).
- Stamped with `source_origin = "test_fixture"`, `text_origin = "test_fixture"`, and `TEST_FIXTURE = True`.
- Production literature services are physically isolated from test fixtures.

### 3.2 Evidence Grounding Invariants
- **Zero Fabricated Papers**: If CrossRef/OpenAlex APIs fail or return no results, the literature service returns `sources = []` rather than injecting synthetic templates.
- **Zero Metadata-Only Claims**: Claims are extracted strictly when verbatim source text (abstract or full text) is available. Sources with `text_origin = 'none'` generate zero claims.

### 3.3 DOI Verification Invariants
- HTTP 200 alone **never** produces `DOIVerificationStatus.VERIFIED`.
- `VERIFIED` requires standard ISO 26324 syntax, active resolution (HTTP 2xx), extracted bibliographic metadata, and title/year fuzzy agreement ($\pm 1$ year tolerance, token Jaccard $\ge 0.50$).

### 3.4 Statistical Critic Invariants
- If $n=1$ seed is evaluated, the statistical critic flags `insufficient_repeated_seed_evidence` and marks `passed = False` with `std = 0.0`, `se = 0.0`, `ci_95 = None`. Zero synthetic p-values or variance are fabricated.

---

## 4. Running the Test Suite

```bash
# Run the complete test suite across all 24 modules
pytest tests/ -v

# Run with coverage report
pytest --cov=backend tests/ -v
```

Expected output: **129 passed (100% pass rate)** across all test modules.

---

## 5. Artifact Inspection & Verification

Every execution writes auditable artifacts to the build workspace:
- `dist/workspace/main.tex`: Full 8–12 page IEEE Transactions LaTeX source.
- `dist/workspace/references.bib`: Verified BibTeX entries with active DOIs.
- `dist/workspace/figures/`: 5-figure vector suite (`fig1` through `fig5` in PDF/PNG).
- `dist/workspace/artifacts/metrics.json`: Multi-seed empirical telemetry and DerSimonian-Laird meta-analysis parameters.
- `dist/experiments/checkpoints/`: PyTorch weights (`proposed_mb_qgt_weights.pt`).
- `artifacts/research_memory.json`: Persistent cross-session knowledge base.
- `artifacts/benchmark_results/agentic_benchmark_summary.json`: Automated benchmark results.
