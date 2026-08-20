from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

from ads_engine.retrieval import VectorIndex, recall_at_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure vector retrieval recall and latency")
    parser.add_argument("--users", type=int, default=2_000)
    parser.add_argument("--ads", type=int, default=20_000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.users > args.ads:
        raise SystemExit("--users cannot exceed --ads for this paired workload")
    rng = np.random.default_rng(args.seed)
    ads = rng.normal(size=(args.ads, args.dimensions)).astype(np.float32)
    relevant = rng.choice(args.ads, size=args.users, replace=False)
    queries = ads[relevant] + rng.normal(0.0, 0.2, size=(args.users, args.dimensions))

    build_start = time.perf_counter()
    index = VectorIndex(ads)
    build_seconds = time.perf_counter() - build_start
    search_start = time.perf_counter()
    result = index.search(queries, args.k)
    search_seconds = time.perf_counter() - search_start

    report = {
        "backend": index.backend,
        "users": args.users,
        "ads": args.ads,
        "dimensions": args.dimensions,
        "k": args.k,
        "seed": args.seed,
        "recall_at_k": recall_at_k(result.ids, relevant),
        "build_seconds": build_seconds,
        "query_seconds": search_seconds,
        "queries_per_second": args.users / search_seconds,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
