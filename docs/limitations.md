# NovaScientist Known Limitations & Operational Boundaries

NovaScientist is an advanced **experimental research-automation and research-infrastructure prototype**. To maintain scientific integrity and prevent misinterpretation, users and evaluators should understand the following operational and methodological boundaries:

---

## 1. Human Oversight & Scientific Responsibility
- **Not a Replacement for Human Scientists**: NovaScientist is an orchestration tool, not an autonomous replacement for domain experts. All generated problem formulations, theoretical hypotheses, mathematical expressions, empirical findings, and manuscript drafts require thorough human verification before scholarly submission or production deployment.
- **Human Authorship Accountability**: In compliance with IEEE, ACM, and international scholarly publishing standards (2024+), AI systems cannot be listed as authors. Human researchers remain solely accountable for the factual accuracy, intellectual originality, and ethical compliance of any published work.

---

## 2. Experimental & Telemetry Boundaries
- **Computational Scope**: NovaScientist operates strictly within digital and computational domains (empirical ML, NLP/RAG, PEFT, spatiotemporal forecasting, graph neural networks, and physics-informed neural surrogates). It cannot execute physical wet-lab chemistry, biology experiments, hardware synthesis, or clinical trials.
- **Statistical Validity Contingency**: Hypothesis decisions ($p$-values, effect sizes, variance bounds) strictly reflect the empirical data collected during execution. If underlying mock surrogates or real training routines run on small sample sizes or synthetic distributions, the statistical conclusions are valid only within the context of those specific runs.
- **Hardware Constraints**: Local multi-seed neural training ($k \ge 5$) requires adequate local GPU (CUDA/MPS) or multi-core CPU capacity. Large foundation model pre-training requires external distributed cluster orchestration.

---

## 3. Literature Retrieval & API Boundaries
- **Upstream Coverage**: Live bibliographic retrieval depends on public scholarly metadata APIs (CrossRef, OpenAlex). Works not indexed by these providers or behind proprietary paywalls may not be discovered.
- **API Availability & Rate Limits**: Upstream network latency or service degradation from scholarly APIs may impact literature synthesis. A deterministic local cache is provided for stability.
- **Passage Extraction**: While DOIs and titles are programmatically verified, domain researchers must confirm that contextual interpretation of cited literature accurately represents original author intent.

---

## 4. Manuscript & Publication Limits
- **Draft Status**: All generated IEEE Transactions LaTeX and compiled PDF documents are publication **drafts**. They serve as high-quality starting scaffolds with pre-formatted vector plots, verified citations, and empirical tables, but they require expert editing, prose polishing, and domain-specific narrative refinement.
- **Automated Peer Review Scope**: The 7-pillar automated reviewer provides structured heuristics and statistical critiques to flag potential weaknesses, but it does not replace independent, double-blind human peer review.

