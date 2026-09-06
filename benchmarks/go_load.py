from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import time
import urllib.request


def request_once(url: str, payload: bytes) -> float:
    started = time.perf_counter()
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}")
        response.read()
    return (time.perf_counter() - started) * 1_000


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * quantile), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Concurrent load test for the Go rank endpoint")
    parser.add_argument("--url", default="http://localhost:8080/v1/rank")
    parser.add_argument("--requests", type=int, default=2_000)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    rng = random.Random(7)
    query = [rng.uniform(-1, 1) for _ in range(32)]
    candidates = [
        {
            "ad_id": f"ad-{index}",
            "embedding": [rng.uniform(-1, 1) for _ in range(32)],
            "pctr": rng.uniform(0.01, 0.3),
            "bid_micros": rng.randint(20_000, 2_000_000),
            "pacing_multiplier": rng.uniform(0.5, 1.5),
        }
        for index in range(200)
    ]
    payload = json.dumps(
        {"request_id": "load-test", "query_embedding": query, "candidates": candidates, "limit": 20}
    ).encode()

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        latencies = list(executor.map(lambda _: request_once(args.url, payload), range(args.requests)))
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "throughput_requests_per_second": args.requests / elapsed,
                "latency_ms": {
                    "mean": statistics.fmean(latencies),
                    "p50": percentile(latencies, 0.50),
                    "p95": percentile(latencies, 0.95),
                    "p99": percentile(latencies, 0.99),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
