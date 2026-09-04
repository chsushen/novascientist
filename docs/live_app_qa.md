# NovaScientist v2.0 — Live Streamlit Application Public QA & Verification Protocol

**Public Application Endpoint**: [`https://novascientist-cqhrr8wptwmrzjksbr8pyw.streamlit.app/`](https://novascientist-cqhrr8wptwmrzjksbr8pyw.streamlit.app/)  
**GitHub Repository**: [`https://github.com/chsushen/novascientist`](https://github.com/chsushen/novascientist)  
**Target Release Baseline**: `v2.0-release` (Commit `285cf32`, 134/134 automated tests passing)  

---

## 1. Executive Purpose & Scope
This testing protocol provides a rigorous manual Quality Assurance (QA) verification checklist for evaluating the live, deployed Streamlit application of **NovaScientist v2.0**. It verifies that external public users experience a fully autonomous, scientific research lifecycle without encountering hardcoded stubs, synthetic placeholders, local filesystem dependencies, or silent failure modes.

---

## 2. Public Live Application Lifecycle Checklist

### Stage 1: Research Topic Formulation & Compute Configuration
* [ ] **Topic Input Field**: Allows free-form textual input of arbitrary scientific/computational queries.
* [ ] **Input Validation & Guard**: Submitting a blank, empty, or whitespace-only query displays an immediate warning banner (`"Please enter a research topic to proceed."`) and halts execution.
* [ ] **Target Paper Format & Length**: Dropdown selection correctly toggles between standard 4-page conference short papers and 8–12 page comprehensive IEEE/ACM journal formats.
* [ ] **Hardware & Compute Detection**: Sidebar displays dynamically inspected physical CPU processor model, active core count, total RAM, and PyTorch execution backend (`cpu`, `mps`, or `cuda`).
* [ ] **Multi-Seed Configuration**: Seed selector allows selecting evaluation folds ($k \in [1, 10]$, default $k=5$).
* [ ] **Epoch & Budget Selection**: Epoch controls adjust training duration per seed fold.
* [ ] **Pre-split AST Isolation Guard**: Verifies that AST validation runs and certifies zero test-set data leakage prior to forward computation.

### Stage 2: Research Planning & Methodology Approval Gate
* [ ] **Domain Classification**: Dispatcher correctly classifies query into one of 8 computational domains (`physics_surrogate`, `graph`, `vision`, `nlp`, `timeseries`, `tabular`, `bioinformatics`, `quantum`).
* [ ] **Structured Plan Generation**: Displays structured `ResearchPlan` containing formal hypothesis, specific research questions, target datasets, primary metrics, and baseline models.
* [ ] **Human-in-the-Loop Approval Gate**: Displays interactive approval controls allowing the user to review the generated plan, adjust parameters if desired, and approve progression to empirical execution.

### Literature & Evidence Grounding Subsystem
* [ ] **Scholarly Source Retrieval**: Queries live CrossRef and OpenAlex APIs for authentic peer-reviewed literature matching the topic domain.
* [ ] **Evidence Passage Extraction**: Extracted claims cite verbatim or near-verbatim supporting text passages from retrieved papers.
* [ ] **Metadata-Only Source Handling**: Papers with missing or truncated abstracts are marked with `is_metadata_only = True` and disallowed from generating ungrounded empirical assertions.
* [ ] **Source Provenance**: Every cited paper is listed with title, authors, year, publication venue, DOI, and BibTeX key.

### Digital Object Identifier (DOI) Verification
* [ ] **CrossRef Resolution**: Queries CrossRef REST API (`https://api.crossref.org/works/{doi}`) to verify DOI existence.
* [ ] **Bi-directional Metadata Matching**: Compares CrossRef metadata against local bibliographic citation via Jaccard title token overlap ($\ge 0.60$) and publication year matching.
* [ ] **Verification State Distinction**: Sources are categorized into explicit states:
  * `VERIFIED`: DOI resolves via HTTP 200 and bibliographic metadata matches.
  * `UNVERIFIED_MISMATCH`: DOI resolves but title/year does not match.
  * `UNVERIFIED_NOT_FOUND`: DOI returns HTTP 404 or fails resolution.
* [ ] **Non-Fabrication**: HTTP 200 alone does not mark a DOI as verified if title tokens diverge.

### Dual-Engine Experimentation & Telemetry
* [ ] **Execution Dispatch**: Runs either high-precision surrogate microbenchmarking or real PyTorch neural network training.
* [ ] **Multi-Seed Execution**: Executes all configured seeds ($k=5$) deterministically with independent random seeds.
* [ ] **Monotonic High-Precision Timing**: Runtimes measured via `time.perf_counter()` rather than estimated clock multipliers.
* [ ] **Telemetry Capture**: Records start/end ISO timestamps, peak RSS RAM usage, inference latency, throughput (samples/sec), and compression ratios.
* [ ] **Model Checkpoint Management**: Checkpoints (`.pt` files) are saved to persistent paths and referenced in deliverables.

### Data-Driven Statistical Critic
* [ ] **Deterministic Metrics Estimation**: Calculates sample mean, standard deviation, and 95% confidence intervals from actual seed outputs.
* [ ] **Effect Size Calculation**: Computes Cohen's $d$ and Hedges' $g$ using exact pooled standard deviation equations.
* [ ] **Hypothesis Testing**: Computes two-tailed paired Student's $t$-test or Wilcoxon signed-rank test $p$-values.
* [ ] **DerSimonian-Laird Random-Effects Meta-Analysis**: Computes Cochran's $Q$, between-study variance $\tau^2$, and Higgins' $I^2$ inconsistency metric.
* [ ] **Insufficient-Seed Failure Gate**: Running single-seed ($k=1$) or underpowered ($k=2$) experiments triggers critical warnings and critic rejection.
* [ ] **Zero-Variance Detection**: Flags identical/unvarying seed metrics as potential anomalies.

### Scientific Reviewer & Bounded Revision Loop
* [ ] **7-Pillar Peer Review**: Evaluates manuscript across Novelty, Methodology, Evidence, Experiments, Statistical Power, Reproducibility, and Limitations.
* [ ] **Structured Finding Objects**: Generates findings with explicit severity ratings (`critical`, `major`, `minor`) and concrete recommendations.
* [ ] **Bounded Self-Correction Loop**: Refactors manuscript to address reviewer findings with a hard upper bound of 3 iterations ($\le 3$).
* [ ] **Convergence & Verdict**: Concludes with an objective peer review verdict (`accept`, `minor_revision`, `major_revision`, or `reject`).

### Persistent Research Memory & Knowledge Graph
* [ ] **Atomic Disk Storage**: Serializes research tasks, evidence bundles, metrics, and review verdicts to disk via atomic file replacement (`os.replace`).
* [ ] **Historical Context Retrieval**: Automatically retrieves past domain-relevant research tasks upon initializing a new topic.
* [ ] **Corrupted File Recovery**: Gracefully handles malformed JSON memory files by recovering to a clean memory state without crashing.
* [ ] **Provenance Graph**: Constructs complete Directed Acyclic Graph (DAG) tracing lineage from Question $\rightarrow$ Plan $\rightarrow$ Sources $\rightarrow$ DOIs $\rightarrow$ Claims $\rightarrow$ Experiments $\rightarrow$ Telemetry $\rightarrow$ Review $\rightarrow$ Deliverables.

### Research Integrity Live Panel
* [ ] **Verified DOI Rate Metric**: Displays true percentage of verified DOIs (e.g., `100.0%`).
* [ ] **Unsupported Claim Rate Metric**: Displays true percentage of unsupported claims (e.g., `0.0%`, highlighted green).
* [ ] **Statistical Power Audit Status**: Displays `✓ PASSED` or `⚠ FLAGGED` based on sample size and variance checks.
* [ ] **Peer Review Verdict Badge**: Displays live reviewer consensus.

### Final Publication Deliverables
* [ ] **PDF Manuscript**: Displays embedded Base64 PDF preview and provides direct download button for `main.pdf`.
* [ ] **LaTeX Source**: Provides downloadable `main.tex` and `references.bib`.
* [ ] **Scientific Vector Figures Suite**: Renders 5-figure interactive carousel (Architecture, Convergence, Pareto Frontier, Ablations, Sensitivity).
* [ ] **Overleaf ZIP Bundle**: Packages all `.tex`, `.bib`, `.cls`, and figure assets into a ready-to-import Overleaf ZIP archive.
* [ ] **Model Checkpoint**: Provides path/download for PyTorch weights (`proposed_mb_qgt_weights.pt`).
* [ ] **Run Summary & Provenance**: Exposes complete `run_summary.json` and `provenance_graph.json` in state inspector.

---

## 3. What Counts as a PASS During Manual Public Testing?

| Subsystem / Stage | PASS Criteria | FAIL Criteria |
|---|---|---|
| **Deployment & Launch** | App loads within 5s with clean UI, zero debug traces, and active hardware sidebar. | Blank white screen, unhandled Streamlit traceback, or missing environment error. |
| **Stage 1 Validation** | Empty topic prompts warning; valid topic activates Domain Dispatcher and stage transition. | App crashes on empty string; hangs without user feedback. |
| **Stage 2 Approval** | Research Plan renders cleanly with objectives, hypotheses, and clickable Proceed button. | Plan contains blank fields; Proceed button is unresponsive or skips approval. |
| **Evidence & DOI** | Verified DOI rate reflects actual CrossRef match rate; claims cite exact sources. | 100% DOI rate reported when fake DOIs are used; ungrounded claims generated. |
| **Experiment Telemetry** | Monotonic timing (`time.perf_counter()`), actual per-seed metrics, non-zero memory profile. | Runtime reported as 0.0s; all seeds have identical unvarying loss arrays. |
| **Statistical Critic** | Exact Cohen's $d$, $p$-value, and $I^2$ computed; single-seed flagged as insufficient. | Single seed claims statistical significance; $p$-values hardcoded to constant. |
| **Scientific Review** | Seven pillars reviewed; revision loop runs $\le 3$ iterations; verdict aligns with findings. | Review loop runs indefinitely ($>3$); critical flaws ignored with "Accept" verdict. |
| **Manuscript & PDF** | Compiles valid 2-column IEEE/ACM PDF; downloads `.tex`, `.bib`, and Overleaf ZIP cleanly. | PDF generation crashes silently without fallback download; broken figure links. |
| **Privacy & Security** | Zero local paths (`/Users/...`), zero API keys, zero private hostnames exposed in UI. | Local developer filesystem paths or raw API keys visible in client UI. |

---

## 4. What Must Be Captured in Screenshots?

When preparing visual documentation or release walkthroughs, capture screenshots of these **genuine UI states**:

1. **Dashboard Home & Configuration**:
   * Initial Stage 1 interface showing topic input box, domain dispatcher tags, and sidebar hardware inspection telemetry.
2. **Interactive Research Plan & Approval Gate**:
   * Stage 2 interface displaying generated hypotheses, baseline configurations, and the approval button.
3. **Research Integrity Banner**:
   * Top-level banner displaying Verified DOI Rate, Unsupported Claim Rate, Statistical Power Audit, and Reviewer Verdict.
4. **5-Figure Scientific Vector Suite Carousel**:
   * Figures 1–5 tabs showing Architecture, Optimization Convergence, Pareto Frontier, Ablations, and 2D Sensitivity Heatmaps.
5. **Matched Publication Venues**:
   * Top 3 target venue cards with impact factors, acceptance rates, and estimated review turnaround times.
6. **PDF Manuscript Viewer & Export Panel**:
   * Interactive PDF preview window with download buttons for PDF, LaTeX source, Overleaf ZIP, and model weights.
7. **Autonomous Multi-Agent State & Provenance Inspector**:
   * Expanded JSON inspector showing `provenance_graph.json` and `run_summary.json`.

---

## 5. Critical Scientific-Integrity Checks

To guarantee absolute scientific honesty during live testing:

* **HTTP 200 vs. Verified DOI**: Verify that a source returning HTTP 200 with mismatched title metadata is categorized as `UNVERIFIED_MISMATCH` and NOT counted as a verified DOI.
* **Missing Supporting Evidence**: Verify that when scholarly search returns empty or metadata-only sources, the Evidence Agent refuses to generate unsupported empirical claims.
* **Missing Experiment Data**: Verify that if an experiment fails or tensor dimensions diverge, the system flags the run as `status = "failed"` rather than fabricating fallback numbers.
* **Single-Seed vs. Multi-Seed**: Verify that configuring $k=1$ seed causes the Statistical Critic to fail the run with an explicit underpowered sample warning ($n < 3$).
* **Mathematical Authenticity**: Verify that Cochran's $Q$, $\tau^2$, and $I^2$ are computed dynamically via the DerSimonian-Laird estimator rather than mocked.
* **Reviewer Independence**: Verify that severe evidence or statistical flaws result in a `major_revision` or `reject` verdict rather than unconditional acceptance.
* **Measured vs. Estimated Runtime**: Verify that reported runtimes reflect monotonic clock measurements (`time.perf_counter()`) of actual tensor forward passes.

---

## 6. Expected Public Failure & Edge-Case Behavior

The application must demonstrate graceful degradation across all common failure modes:

| Failure Scenario | Root Cause | Expected Application Behavior |
|---|---|---|
| **Scholarly API Timeout** | CrossRef or OpenAlex latency exceeds 5.0s. | Catches timeout gracefully, falls back to local cached references or metadata-only mode with informational warning; no crash. |
| **DOI Resolution Failure** | DOI does not exist (HTTP 404) or server error (HTTP 500). | Marks source as `UNVERIFIED_NOT_FOUND`; updates Verified DOI rate accordingly; proceeds safely. |
| **Missing Abstract / Metadata** | Scholarly database provides only title and author list. | Marks paper as `is_metadata_only = True`; includes citation in BibTeX but restricts claim generation. |
| **Underpowered Sample Size** | User configures $k=1$ or $k=2$ random seeds. | Statistical Critic flags study as underpowered; Scientific Reviewer issues critical finding; requires multi-seed ($k \ge 3$) for acceptance. |
| **Corrupted Memory JSON** | Memory file on disk is truncated or contains invalid JSON. | Research Memory automatically logs warning, salvages corrupt file to backup, and initializes clean memory bank without crashing. |
| **LaTeX Engine Unavailable** | Cloud container lacks installed Tectonic / pdflatex binary. | Renders notification banner and provides direct one-click download for raw `main.tex`, `references.bib`, and Overleaf ZIP archive. |
| **Empty Topic Submission** | User clicks submit with blank text field. | Displays warning message requesting valid research query; retains current UI state. |

---

## 7. Verification Summary & Next Steps
Upon successful completion of all checklist items on the live deployment:
1. Record timestamp and execution device of manual verification.
2. Store verified run artifacts in `artifacts/demo/`.
3. Proceed to public announcement on GitHub, LinkedIn, and project portfolio.
