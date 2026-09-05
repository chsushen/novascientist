# NovaScientist — Technical Interview Story & Architectural Deep Dive

This document provides honest, detailed, engineering-focused answers to common technical interview questions regarding NovaScientist.

---

### Q1: What did you actually build?
**Answer:**
I built **NovaScientist**, an evidence-first AI research orchestration platform and research infrastructure prototype. It takes a high-level research question (e.g., *"Can Retrieval-Augmented Generation improve factual consistency in domain QA?"*), decomposes it into a formal investigation plan, grounds it in verified scholarly literature via CrossRef/OpenAlex APIs, freezes an immutable **Scientific Research Contract**, executes multi-seed empirical experiments, runs formal statistical tests using SciPy, constructs an end-to-end provenance graph, and generates publication-ready IEEE Transactions LaTeX drafts and compiled PDFs with dynamic vector figures.

It is not an ungrounded chatbot or a speculative wrapper; it is a modular, deterministic pipeline that coordinates LLM planning with rigorous scientific computing and typesetting tools.

---

### Q2: What problem does it solve?
**Answer:**
Automated research systems and LLM-assisted science commonly suffer from four critical integrity failures:
1. **Hallucinated Citations & Fabricated Findings**: Large language models easily generate fake DOIs or invent benchmark results when telemetry is missing.
2. **Template Drift & Metric Recycling**: Systems often hardcode default metrics or reuse the same static benchmark across completely unrelated domains (e.g., applying hardware compression metrics to text summarization).
3. **Missing Statistical Significance**: Single-seed results are often presented as definitive without reporting sample standard deviations, paired hypothesis tests, or effect size distributions.
4. **Untraceable Lineage**: Once a manuscript or summary is generated, it is impossible to determine which seed, dataset, or literature source produced a specific number or claim.

NovaScientist solves these challenges through strict structural constraints: frozen contracts, zero-fallback telemetry rules, formal SciPy statistical testing, and cryptographic DAG provenance tracking.

---

### Q3: Why did you use a Research Contract?
**Answer:**
In multi-agent systems, "drift" is a major point of failure. If Stage 1 decides to investigate PubMedQA for Medical NLP, downstream agents (the experiment runner, the figure generator, the LaTeX assembler) might inadvertently revert to hardcoded defaults or general-purpose metrics.

To prevent this, we introduced `ScientificResearchContract`. At the end of the scoping phase, the contract is locked via `.freeze()`. It explicitly records:
- The target dataset and baseline candidate suite
- The proposed architectural variant
- Contracted primary and secondary metrics
- Required mathematical loss formulations
- Required statistical tests and significance thresholds
- Contracted figure blueprints and manuscript section structure

Post-freeze, the contract is immutable (raising `ScientificContractViolationError` on attempted modifications). Downstream validators audit all generated artifacts against the frozen contract. If a metric or figure does not match the contract, the pipeline fails closed.

---

### Q4: Why Provenance?
**Answer:**
Scientific credibility requires auditable lineage. In NovaScientist, every entity (question, literature source, DOI verification, claim, methodology spec, seed execution run, telemetry array, hypothesis evaluation, peer review finding, and compiled artifact) is modeled as a node in a NetworkX Directed Acyclic Graph (DAG).

Edges define explicit cryptographic relationships (`grounded_in`, `tested_by`, `evaluated_from`, `supports`). This allows a researcher to click on any number or chart in the compiled manuscript and trace its exact lineage back through the statistical test, the specific random seed logs, and the originating research question. Furthermore, DAG validation routines verify graph closure, catching orphaned claims or dangling experiments.

---

### Q5: Where are LLMs used vs. what is deterministic?
**Answer:**
We strictly separated LLM generative reasoning from deterministic software execution:

- **LLM Reasoning Is Used For**:
  - Semantic parsing and research question decomposition.
  - Formulating qualitative theoretical descriptions and hypothesis prose.
  - Multi-perspective peer review critiques (evaluating novelty, readability, and clarity).
  - Drafting narrative section text under structured constraints.

- **Deterministic Software Is Used For**:
  - Live bibliographic API querying (CrossRef/OpenAlex) and DOI format verification.
  - Contract state management, schema enforcement, and freeze locks.
  - Multi-seed PyTorch neural training and microbenchmark execution.
  - Wall-clock runtime profiling (`time.perf_counter`) and memory tracking.
  - All statistical calculations (paired Student\x27s $t$-tests, Chi-Square variance tests, Cohen\x27s $d$, DerSimonian-Laird meta-analysis) via `scipy.stats` and `numpy`.
  - Matplotlib vector figure rendering from telemetry logs.
  - Tectonic LaTeX compilation and physical PDF page validation via `pypdf`.
  - Cryptographic SHA-256 hashing and provenance graph construction.

---

### Q6: What are the known limitations?
**Answer:**
NovaScientist is an experimental research-automation prototype with clear boundaries:
1. **Computational Only**: It operates in digital/computational domains (ML, NLP, Time-Series, GNNs, PINNs) and cannot conduct physical wet-lab or clinical experiments.
2. **Statistical Grounding Contingency**: Statistical outputs reflect the empirical reality of the supplied experiments. If experiments run on simplified surrogates or small datasets, the statistics are valid only for those specific experimental conditions.
3. **Scaffold Draft Status**: Generated manuscripts are high-quality scaffolds and drafts that require critical human domain expert review, verification, and editing before scholarly submission.
4. **Scholarly Coverage**: Bibliographic retrieval depends on public scholarly metadata APIs; proprietary or non-indexed literature may not be retrieved.

---

### Q7: What did you learn from testing and building this system?
**Answer:**
1. **The Danger of Fallback Placeholders**: Early versions had fallback defaults (e.g., `metrics.get("p_value", 0.001)`). In practice, this creates subtle integrity bugs where missing telemetry still results in a "passed" scientific hypothesis. We eliminated every fallback constant so that missing telemetry explicitly yields `NOT_EVALUATED` and halts the release gate.
2. **Fail-Closed Validation Is Essential**: In AI pipelines, silent degradation is worse than explicit crashes. By enforcing fail-closed contract gates, broken dataflows are caught immediately during execution.
3. **Comprehensive Unit & Integration Testing**: Maintaining a 209-test suite with adversarial test cases (e.g., injecting uncontracted metrics, corrupting random seeds, testing AST boundary leakage) was crucial for ensuring architectural stability and preventing regression.
