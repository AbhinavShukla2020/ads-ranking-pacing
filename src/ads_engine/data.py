from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EventTable:
    user_features: NDArray[np.float32]
    ad_features: NDArray[np.float32]
    context_features: NDArray[np.float32]
    clicked: NDArray[np.float32]
    converted: NDArray[np.float32]
    bid: NDArray[np.float32]
    market_price: NDArray[np.float32]

    def __len__(self) -> int:
        return len(self.clicked)

    @property
    def ranking_features(self) -> NDArray[np.float32]:
        return np.concatenate(
            (self.user_features, self.ad_features, self.context_features), axis=1
        )

    def take(self, indices: NDArray[np.integer]) -> EventTable:
        return EventTable(
            self.user_features[indices],
            self.ad_features[indices],
            self.context_features[indices],
            self.clicked[indices],
            self.converted[indices],
            self.bid[indices],
            self.market_price[indices],
        )


def _sigmoid(values: NDArray[np.floating]) -> NDArray[np.float32]:
    clipped = np.clip(values, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def synthetic_events(
    count: int,
    *,
    user_dim: int = 16,
    ad_dim: int = 16,
    context_dim: int = 8,
    seed: int = 7,
) -> EventTable:
    """Generate an imperfect but learnable impression stream."""
    if count < 2:
        raise ValueError("count must be at least two")
    rng = np.random.default_rng(seed)
    users = rng.normal(size=(count, user_dim)).astype(np.float32)
    ads = rng.normal(size=(count, ad_dim)).astype(np.float32)
    context = rng.normal(size=(count, context_dim)).astype(np.float32)

    latent_dim = min(user_dim, ad_dim, 8)
    affinity = np.sum(users[:, :latent_dim] * ads[:, :latent_dim], axis=1) / np.sqrt(
        latent_dim
    )
    click_logit = -2.3 + 0.7 * affinity + 0.25 * context[:, 0] - 0.15 * context[:, 1]
    click_probability = _sigmoid(click_logit)
    clicked = rng.binomial(1, click_probability).astype(np.float32)

    conversion_logit = -1.8 + 0.45 * affinity + 0.3 * ads[:, 0] + 0.2 * context[:, 2]
    post_click_probability = _sigmoid(conversion_logit)
    converted = (clicked * rng.binomial(1, post_click_probability)).astype(np.float32)

    bid = rng.lognormal(mean=-0.2, sigma=0.55, size=count).astype(np.float32)
    market_price = rng.lognormal(mean=-0.35, sigma=0.5, size=count).astype(np.float32)
    return EventTable(users, ads, context, clicked, converted, bid, market_price)


def save_npz(events: EventTable, destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        user_features=events.user_features,
        ad_features=events.ad_features,
        context_features=events.context_features,
        clicked=events.clicked,
        converted=events.converted,
        bid=events.bid,
        market_price=events.market_price,
    )


def load_npz(source: str | Path) -> EventTable:
    with np.load(source) as values:
        return EventTable(
            values["user_features"].astype(np.float32),
            values["ad_features"].astype(np.float32),
            values["context_features"].astype(np.float32),
            values["clicked"].astype(np.float32),
            values["converted"].astype(np.float32),
            values["bid"].astype(np.float32),
            values["market_price"].astype(np.float32),
        )
