# NovaScientist — Resume-Accurate Project Description

## Headline
> Built an evidence-first AI research orchestration platform that transforms research questions into structured research plans, literature/evidence workflows, experiment pipelines, provenance graphs, statistical analyses, and reproducible IEEE/Overleaf publication artifacts.

---

## Technical Bullet Points (Interview & Resume Ready)

- **Frozen Research Contract Architecture**: Designed a fail-closed orchestration engine in Python that binds problem scopes, datasets, baselines, and evaluation protocols into an immutable contract (`ScientificResearchContract`), eliminating downstream template drift and benchmark leakage.
- **Empirical Multi-Seed Execution & Statistical Verification**: Built deterministic multi-seed experiment runners ($k \ge 5$) integrated with SciPy statistical routines (`paired_t_test`, `chi2` variance bounds, Cohen's $d$, and DerSimonian-Laird random-effects meta-analysis) with zero fallback placeholders.
- **Cryptographic Provenance DAG**: Developed a NetworkX-based Directed Acyclic Graph tracker establishing complete end-to-end lineage across research questions, scholarly citations (CrossRef/OpenAlex DOI verified), empirical seed telemetry, statistical critiques, and publication deliverables.
- **Automated Publication & Vector Figure Pipeline**: Engineered a deep journal typesetting engine compiling multi-page IEEE Transactions LaTeX manuscripts and standalone PDFs via Tectonic, complete with dynamic multi-panel Matplotlib vector figures and Overleaf export packages.
- **Production API & Sandboxed Job Infrastructure**: Implemented an asynchronous FastAPI service layer featuring decoupled background task workers, job status tracking, memory-bounded security sandboxing, AST dataflow guards, and comprehensive diagnostic endpoints.
- **Rigorous Test Suite & Verification**: Authored a 209-test automated test suite achieving 100% pass rate across contract invariants, API contracts, provenance completeness, and statistical inference routines.
