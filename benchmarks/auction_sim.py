from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from ads_engine.auction import Candidate, run_second_price
from ads_engine.pacing import PIDPacer


@dataclass
class Outcome:
    spend: float = 0.0
    expected_value: float = 0.0
    wins: int = 0


def run_stream(
    qualities: np.ndarray,
    values: np.ndarray,
    competitor_bids: np.ndarray,
    budget: float,
    paced: bool,
) -> Outcome:
    outcome = Outcome()
    pacer = PIDPacer(budget=budget, kp=1.0, ki=0.2, kd=0.01, maximum=1.8)
    count = len(qualities)

    for index, (quality, value, competitor) in enumerate(
        zip(qualities, values, competitor_bids, strict=True)
    ):
        if paced and index % 100 == 0:
            pacer.update(index / count, outcome.spend)
        multiplier = pacer.multiplier if paced else 1.0
        base_bid = float(value)
        auction = run_second_price(
            [
                Candidate("campaign", base_bid, float(quality), multiplier),
                Candidate("market", float(competitor), 1.0),
            ],
            reserve=0.01,
        )
        if auction.winner_id == "campaign" and outcome.spend + auction.clearing_price <= budget:
            outcome.spend += auction.clearing_price
            outcome.expected_value += float(quality * value)
            outcome.wins += 1
    return outcome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a synthetic second-price auction stream")
    parser.add_argument("--auctions", type=int, default=50_000)
    parser.add_argument("--budget", type=float, default=5_000.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    qualities = rng.beta(2.0, 18.0, args.auctions)
    values = rng.lognormal(0.6, 0.5, args.auctions)
    competitor_bids = rng.lognormal(-1.0, 0.7, args.auctions)
    baseline = run_stream(qualities, values, competitor_bids, args.budget, paced=False)
    paced = run_stream(qualities, values, competitor_bids, args.budget, paced=True)

    def summarize(outcome: Outcome) -> dict[str, float | int]:
        return {
            "spend": outcome.spend,
            "budget_error_fraction": (outcome.spend - args.budget) / args.budget,
            "expected_value": outcome.expected_value,
            "wins": outcome.wins,
        }

    print(
        json.dumps(
            {
                "auctions": args.auctions,
                "budget": args.budget,
                "seed": args.seed,
                "constant_multiplier": summarize(baseline),
                "pid_pacing": summarize(paced),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
