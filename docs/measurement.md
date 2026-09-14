# Measurement protocol

Performance numbers are useful only with enough context to reproduce them.
Store benchmark JSON under `benchmark-results/` and record:

- Git commit and whether the working tree was clean.
- Dataset name, filters, split strategy, and row counts.
- Random seed and model configuration.
- CPU, GPU, memory, operating system, Python, PyTorch, FAISS, and Go versions.
- Warm-up policy, concurrency, duration, and request distribution.

For retrieval, report recall@K against a brute-force reference and query latency
separately. For ranking, report ROC AUC and log loss on a time-based holdout. For
serving, include p50, p95, and p99 rather than a single average. For pacing,
report end-of-window budget error, the full spend curve, and value-weighted click
yield under the same auction stream as the baseline.

The benchmark tools print JSON to standard output. Redirect it to a named file
only after recording the environment above.
