# NovaScientist Known Limitations & Operational Boundaries

## 1. Domain Coverage Boundaries
- **Supported Paradigms**: Empirical Machine Learning, Natural Language Processing, Computer Vision, Spatiotemporal Time Series, Graph Neural Networks, and Physics-Informed Operator Surrogates.
- **Unsupported Paradigms**: Wet-lab biological experiments, proprietary clinical trials, non-computational social science, and ungrounded speculative reasoning.

---

## 2. Resource & Execution Constraints
- **Hardware Profile**: Default microbenchmarks and surrogate training runs are optimized for multi-core POSIX CPU and CUDA environments. Distributed multi-node training requires external cluster orchestration.
- **Literature Rate Limits**: Live querying of CrossRef and OpenAlex APIs is subject to external upstream rate limits. A built-in literature cache is maintained for stability.

---

## 3. Human Oversight Requirement
- **No Automated Publishing**: NovaScientist requires human scientific oversight before any generated manuscript is submitted to academic peer review or commercial venues.
- **Authorship Compliance**: In accordance with IEEE and ACM 2024+ publishing policies, human researcher credentials must be provided as primary authors.
