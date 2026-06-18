# Ads Ranking, Retrieval & Budget Pacing Engine

An end-to-end learning project for the core systems behind an ad marketplace:
candidate retrieval, click/conversion ranking, second-price auctions, and
closed-loop budget pacing. The code is split at the same boundary commonly used
in serving systems: Python owns data preparation, model training, evaluation,
and offline simulation; a small Go service handles latency-sensitive ranking and
Redis-backed response caching.

The repository ships with a deterministic synthetic event generator so the full
pipeline runs without access to private ad logs. It can also read flat event
files shaped like the public Alibaba UserBehavior data.

> **Practice-history note:** this repository's commit timestamps are intentionally
> distributed across a simulated three-month development window for Git practice.
> They do not record the dates when this implementation was originally assembled.

## Architecture

```text
event rows -> feature encoding -> two-tower training -> vector index
                                      |
                                      v
                              DCNv2 + ESMM ranker
                                      |
query -> candidate retrieval -> value score -> PID pacing -> second-price auction
                                      |
                                      v
                               Go JSON API + Redis
```

## Included components

- PyTorch two-tower model with normalized user/ad embeddings.
- FAISS inner-product retrieval, with a NumPy fallback for local development.
- DCNv2-style cross layers and an ESMM objective for CTR and post-click CVR.
- Rank-based AUC, recall@K, and binary log-loss implementations.
- PID pacing controller with anti-windup and configurable multiplier bounds.
- Deterministic second-price auction with reserve prices and tie-breaking.
- Go `/v1/rank` API, bounded request validation, and optional Redis caching.
- Synthetic workload, offline auction simulation, and concurrent HTTP load tool.
- Python and Go tests plus a GitHub Actions workflow.

## Quick start

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Train both models on a deterministic synthetic sample:

```bash
ads-train --events 100000 --epochs 3 --output artifacts
```

Run retrieval and auction measurements:

```bash
python benchmarks/retrieval_benchmark.py --users 2000 --ads 20000
python benchmarks/auction_sim.py --auctions 50000 --seed 7
```

Run the serving stack with Docker:

```bash
docker compose up --build
curl http://localhost:8080/healthz
```

See [docs/architecture.md](docs/architecture.md) for model and systems details
and [docs/measurement.md](docs/measurement.md) for the benchmark protocol.

## Performance reporting

Benchmark output depends on the dataset, hardware, software versions, and random
seed. This repository does not bundle claimed AUC, recall, throughput, or pacing
results. The included commands emit machine-readable measurements so results can
be reported alongside their exact environment and commit.

## License

MIT
