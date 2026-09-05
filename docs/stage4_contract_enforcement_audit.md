# Downstream Scientific Contract Enforcement & Stage 4 UI Audit

## 1. Executive Summary

In NovaScientist v2.3, the **`ScientificResearchContract`** was established as the **Single Authoritative Source of Truth** across the entire autonomous research lifecycle:
$$\text{Research Question} \longrightarrow \text{Scientific Contract} \longrightarrow \text{Execution \& Telemetry} \longrightarrow \text{Manuscript \& Figures} \longrightarrow \text{Stage 4 UI}$$

This audit verifies that all downstream stages (Stage 2 approval $\rightarrow$ Stage 3 execution $\rightarrow$ LaTeX Assemblers $\rightarrow$ Vector Figures $\rightarrow$ Stage 4 Streamlit UI) dynamically and strictly adhere to the approved contract with zero uncontracted template injection, zero hardware metrics leakage on non-hardware questions, and zero UI indexing exceptions.

---

## 2. Identified Root Causes & Resolved Defects

| Component | Defect / Vulnerability | Resolution |
| :--- | :--- | :--- |
| **`app.py` Stage 4** | `NameError: fig_tabs[0]` caused by static `tab1..tab5 = st.tabs(...)` followed by `with fig_tabs[0]:`. | Replaced with dynamic discovery: `fig_tabs = st.tabs(...)` iterating over `discovered_figs`. |
| **`app.py` Stage 4 Metrics** | Hardcoded 4-column metrics (Peak RAM, Latency, Meta-Analysis summary $I^2$) even on NLP / RAG topics. | Dynamically checks `contract_has_hardware` and `contract.statistical_requirement`. Falls back to Cross-Seed Variance and Hypothesis test $p$-value for non-hardware runs. |
| **`app.py` Stage 2 $\rightarrow$ 3** | User contract approval did not call `contract.freeze()` or store in `st.session_state.contract`. | Stage 2 explicitly invokes `contract.freeze()`, persists to session state, and Stage 3 forwards `contract` to `orchestrator.execute(...)`. |
| **`research_contract.py`** | Mutation of frozen contracts was possible for unlisted fields. | `__setattr__` now intercepts **all** field mutations once `is_frozen = True`, raising fail-closed `ScientificContractViolationError`. |
| **`latex_assembler.py`** | Defaulted to hardware table with RAM/Latency columns even for NLP/forecasting questions when contract was active. | Dynamically builds `tab1_block` with or without hardware columns, matches `contract.figure_requirements`, and adapts complexity discussions. |
| **`deep_journal_assembler.py`** | Hardcoded fallback inserted 5 legacy figures if `figures_latex_block` was empty. | Figure generation and LaTeX insertion now strictly adhere to `contract.figure_requirements`. |
| **`orchestrator.py`** | Unconditionally recorded `meta_analysis` provenance node even when contract specified standard hypothesis testing. | Dynamically branches provenance recording between `meta_analysis` and `statistical_analysis`. Writes `contract_consistency_report.json` to workspace. |

---

## 3. Canonical Acceptance Test Verification Matrix

Three canonical scientific questions across distinct computational paradigms were verified through end-to-end autonomous execution:

```
======================================================================
[TEST 1] Question: Can Retrieval-Augmented Generation Improve Factual Consistency in Domain-Specific Question Answering?
Contract ID: contract_9797e1771b
Task: question_answering | Paradigm: EMPIRICAL_BENCHMARK
Dataset: PubMedQA Biomedical Question Answering
Primary Metrics: ['Factual Consistency Score (%)', 'Exact Match (EM %)']
Math Requirement: optimization_objective
Stat Requirement: paired_t_test
Figure Requirements (2): ['Factual Consistency vs Retrieval Depth', 'Hallucination Rate vs Context Density']
Execution Status: PASS | Physical PDF: 7 Pages (Target: 6–8) | Figures: 2
Contract Consistency Report: PASS
Uncontracted Hardware Terms Found: 0
======================================================================
[TEST 2] Question: Can parameter-efficient adaptation improve domain-specific text classification while reducing trainable parameters?
Contract ID: contract_c1f4880fd5
Task: text_classification | Paradigm: METHODOLOGICAL_COMPARISON
Dataset: GLUE General Language Understanding Benchmark
Primary Metrics: ['Macro F1 Score (%)', 'Top-1 Accuracy (%)']
Math Requirement: optimization_objective
Stat Requirement: paired_t_test
Figure Requirements (3): ['System Architecture Diagram', 'Macro-F1 vs Trainable Parameter Footprint Pareto Frontier', 'Adapter Module Component Ablation Bar Chart']
Execution Status: PASS | Physical PDF: 7 Pages (Target: 6–8) | Figures: 3
Contract Consistency Report: PASS
Uncontracted Hardware Terms Found: 0
======================================================================
[TEST 3] Question: How can long-horizon multivariate forecasting remain accurate under temporal distribution shift?
Contract ID: contract_92b3af0fc0
Task: timeseries_forecasting | Paradigm: METHODOLOGICAL_COMPARISON
Dataset: Exchange-Rate Multi-Horizon Forecasting
Primary Metrics: ['Mean Absolute Error (MAE)', 'Root Mean Squared Error (RMSE)']
Math Requirement: derivation_only
Stat Requirement: paired_t_test
Figure Requirements (2): ['Forecast Trajectories vs Ground Truth', 'Horizon-Wise Error Degradation Curve']
Execution Status: PASS | Physical PDF: 7 Pages (Target: 6–8) | Figures: 2
Contract Consistency Report: PASS
Uncontracted Hardware Terms Found: 0
======================================================================
ALL 3 CANONICAL ACCEPTANCE RUNS PASSED CERTIFICATION!
======================================================================
```

---

## 4. Statistical Procedure Selector Audit

The statistical selection engine was audited to ensure methods are derived strictly from experimental design invariants rather than keyword heuristics:

| Experimental Design Invariants | Derived Statistical Procedure | Verification Status |
| :--- | :--- | :--- |
| Paired evaluation, Gaussian / Normal distributions, $k \ge 3$ seeds | `StatisticalRequirement.PAIRED_T_TEST` | **PASS** |
| Paired evaluation, Non-Normal / Ordinal / Heavy-tailed | `StatisticalRequirement.WILCOXON_SIGNED_RANK` | **PASS** |
| Small sample regime ($k = 2$ seeds) | `StatisticalRequirement.BOOTSTRAP_CONFIDENCE_INTERVAL` | **PASS** |
| Single-run execution ($N = 1$ deterministic run) | `StatisticalRequirement.NONE` (Descriptive only) | **PASS** |
| Multi-group independent comparisons ($>2$ groups, unpaired) | `StatisticalRequirement.ONE_WAY_ANOVA` | **PASS** |
| Multicenter / Multi-site heterogeneous variance datasets | `StatisticalRequirement.RANDOM_EFFECTS_META_ANALYSIS` | **PASS** |

---

## 5. Physical PDF Page Count & Compilation Audit

- **Standard Conference Target**: 6–8 Physical Pages $\longrightarrow$ **Actual Output: 7 Pages** (`within_target = True`)
- **Extended Journal Target**: 8–12 Physical Pages $\longrightarrow$ **Actual Output: 10 Pages** (`within_target = True`)
- **Verification Engine**: `pypdf.PdfReader` counting physical rasterized / vector pages with IEEE-compatible formatting and zero uncontracted filler.

---

## 6. Test Suite Summary

- **Total Test Cases**: 194
- **Passed**: 194 (100%)
- **Failed**: 0
- **Regression Suite**: `tests/test_stage4_contract_enforcement.py` added with 5 unit/integration test cases covering contract freezing immutability, assembler compliance, mathematical treatment enforcement, and statistical selector design derivations.
