from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    ad_id: str
    base_bid: float
    quality: float
    pacing_multiplier: float = 1.0

    @property
    def effective_bid(self) -> float:
        if self.base_bid < 0 or self.quality < 0 or self.pacing_multiplier < 0:
            raise ValueError("bid, quality, and pacing multiplier must be non-negative")
        return self.base_bid * self.quality * self.pacing_multiplier


@dataclass(frozen=True)
class AuctionResult:
    winner_id: str | None
    clearing_price: float
    winning_bid: float
    eligible_count: int


def run_second_price(candidates: list[Candidate], reserve: float = 0.0) -> AuctionResult:
    """Run a deterministic quality-adjusted second-price auction."""
    if reserve < 0:
        raise ValueError("reserve cannot be negative")

    eligible = [candidate for candidate in candidates if candidate.effective_bid >= reserve]
    eligible.sort(key=lambda candidate: (-candidate.effective_bid, candidate.ad_id))
    if not eligible:
        return AuctionResult(None, 0.0, 0.0, 0)

    winner = eligible[0]
    runner_up = eligible[1].effective_bid if len(eligible) > 1 else reserve
    clearing_price = min(winner.effective_bid, max(reserve, runner_up))
    return AuctionResult(winner.ad_id, clearing_price, winner.effective_bid, len(eligible))
