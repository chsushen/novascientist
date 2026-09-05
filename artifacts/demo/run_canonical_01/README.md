# Canonical Demo Run: `run_canonical_01`
**Topic**: *Low-Rank Dynamic Graph Attention for Smart Disaster Resilience and Evacuation Forecasting*  
**Architecture**: Memory-Bounded Quantized Graph Transformer (`MB-QGT`)  
**Domain**: Spatial & Spatiotemporal Graph Neural Networks  
**Dataset**: METR-LA Urban Traffic & Evacuation Sensor Network Benchmark Dataset ($N = 34,272$ samples, 207 spatial nodes)  

---

## 1. Overview
This canonical demonstration encapsulates a complete, end-to-end autonomous research execution of NovaScientist. All empirical data, multi-seed training trajectories, statistical meta-analyses, evidence claims, peer review reports, and publication assets were generated and verified without synthetic stubs or artificial success multipliers.

---

## 2. Directory Structure & Key Files

```text
artifacts/demo/run_canonical_01/
├── run_summary.json        # Machine-readable executive summary of all material run metrics
├── provenance_graph.json   # Full fine-grained DAG tracking all 28 entities and 33 lineage relations
└── README.md               # Audit guide and lineage verification manual
```

---

## 3. How to Read `run_summary.json`

`run_summary.json` provides an auditable summary of all high-level experimental and statistical claims:

* **Experiment Setup**: Real PyTorch training on Apple Silicon MPS with $k=5$ deterministic seeds (`[42, 179, 316, 453, 590]`), 10 epochs, batch size 64.
* **Empirical Metrics**:
  * `proposed_accuracy`: **88.35%** $\pm$ 1.09% (vs. Dense FP32 baseline **81.94%** $\pm$ 1.56%).
  * `accuracy_gain_pct`: **+6.41%** absolute improvement.
  * `memory_mb`: **74.98 MB** (reduced from 414.39 MB, an **81.9%** memory reduction).
  * `inference_latency_ms`: **9.27 ms** (accelerated from 38.33 ms, a **4.13$\times$** latency speedup).
  * `throughput_samples_sec`: **6,915.9 samples/sec**.
* **Statistical Rigor**:
  * `meta_effect_size`: **+0.0638** (+6.38% pooled effect size).
  * `meta_ci_95`: `[0.0540, 0.0736]`.
  * `cochran_q`: **0.6705** ($p = 0.9549$), `tau_squared`: **0.0**, `meta_i_squared_percent`: **0.0%**.
  * `z_statistic`: **12.72**, `p_value_z`: **0.0** ($p < 10^{-4}$).
* **Research Integrity**:
  * `verified_doi_rate`: **1.0** (100.0% of citations verified via CrossRef).
  * `unsupported_claim_rate`: **0.0%** (all claims tethered to literature evidence).
  * `single_seed_fabrication`: `false`, `ast_leakage_detected`: `false`.

---

## 4. How to Read `provenance_graph.json`

`provenance_graph.json` models the scientific lineage as a formal Directed Acyclic Graph (DAG) containing 28 nodes and 33 typed edges. The lineage spans:

$$\text{Question} \longrightarrow \text{Plan} \longrightarrow \text{Sources} \longrightarrow \text{DOI Verifications} \longrightarrow \text{Claims} \longrightarrow \text{Validation Report} \longrightarrow \text{Methodology}$$
$$\longrightarrow \text{Experiment Spec} \longrightarrow \text{5 Seed Runs} \longrightarrow \text{Telemetry} \longrightarrow \text{Meta-Analysis} \longrightarrow \text{Critic} \longrightarrow \text{Review} \longrightarrow \text{Revision} \longrightarrow \text{Deliverables}$$

### Lineage Node Hierarchy
1. **Research Formulation**: `q_001` (Topic) $\rightarrow$ `plan_001` (Plan).
2. **Scholarly Grounding**: `src_li_2018`, `src_chen_2020`, `src_metrla_2024` $\rightarrow$ `doi_verif_*` $\rightarrow$ `claim_*` $\rightarrow$ `val_report_001`.
3. **Architecture & Baselines**: `meth_001` (MB-QGT), `base_001` (Dense FP32).
4. **Empirical Execution**: `exp_spec_001` $\rightarrow$ 10 discrete `seed_run_*` nodes ($k=5$ for proposed, $k=5$ for baseline).
5. **Statistical Synthesis**: `telemetry_metrics_001` $\rightarrow$ `stat_analysis_001` $\rightarrow$ `stat_critic_001`.
6. **Peer Review & Revision**: `rev_findings_001` (7-pillar audit) $\rightarrow$ `rev_cycle_001` (revision iteration 1) $\rightarrow$ `rev_verdict_001` (Accept).
7. **Evaluation & Artifacts**: `eval_benchmark_001` $\rightarrow$ `conc_001` $\rightarrow$ `deliv_pdf_001`, `deliv_tex_001`, `deliv_figures_001`, `deliv_checkpoint_001`.

---

## 5. Physical Location of Underlying Artifacts

| Artifact Category | File System Path | Contents |
|-------------------|------------------|----------|
| **Raw Training Telemetry** | `dist/experiments/logs/training_log.json` | 5-seed loss histories, per-seed accuracies, latency & memory profiling |
| **Statistical Metrics JSON** | `dist/test_journal_workspace/artifacts/metrics.json` | Complete DerSimonian-Laird meta-analysis, Cochran's Q, study weights |
| **Compiled Manuscript PDF** | `dist/test_journal_workspace/main.pdf` | 8-page IEEE Transactions formatted manuscript |
| **LaTeX & BibTeX Source** | `dist/test_journal_workspace/main.tex`, `references.bib` | Fully assembled LaTeX document with verified citations |
| **Vector Figure Suite** | `dist/reproduced_figures/` (`fig1`–`fig5`) | High-res PNG & PDF vector figures (Architecture, Convergence, Pareto, etc.) |
| **Trained Model Weights** | `dist/experiments/checkpoints/proposed_mb_qgt_weights.pt` | Saved PyTorch checkpoint weights for MB-QGT architecture |

---

## 6. Manual Lineage Verification

To verify the integrity of the canonical demonstration manually:

1. **Verify Seed Execution**:
   Inspect `dist/experiments/logs/training_log.json` and confirm that all 5 seed runs (`42`, `179`, `316`, `453`, `590`) contain exact loss arrays and metric measurements matching `provenance_graph.json`.
2. **Verify Meta-Analysis Math**:
   Confirm that pooled effect size ($0.0638$) equals the weighted average across individual seed gains:
   $$\hat{\theta}_{\text{pooled}} = \frac{\sum w_i \hat{\theta}_i}{\sum w_i} = 0.0638$$
3. **Verify DOI Verification**:
   Inspect `references.bib` and verify DOIs `10.1145/3209978.3210006` and `10.1109/TPAMI.2020.2987823` resolve to Li et al. (2018) and Chen et al. (2020).
