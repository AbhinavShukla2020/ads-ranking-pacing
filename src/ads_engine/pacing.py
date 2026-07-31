from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PIDPacer:
    """Adjust bid multipliers to follow a cumulative budget curve.

    Positive error means the campaign is behind its target spend, so the
    multiplier increases. ``update`` expects monotonically increasing elapsed
    fractions and cumulative spend.
    """

    budget: float
    kp: float = 1.2
    ki: float = 0.15
    kd: float = 0.05
    minimum: float = 0.05
    maximum: float = 2.0
    integral_limit: float = 1.5
    multiplier: float = 1.0
    _integral: float = field(default=0.0, init=False, repr=False)
    _last_error: float | None = field(default=None, init=False, repr=False)
    _last_fraction: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if not 0 < self.minimum <= self.maximum:
            raise ValueError("multiplier bounds must satisfy 0 < minimum <= maximum")
        self.multiplier = min(max(self.multiplier, self.minimum), self.maximum)

    def update(self, elapsed_fraction: float, cumulative_spend: float) -> float:
        if not self._last_fraction <= elapsed_fraction <= 1.0:
            raise ValueError("elapsed_fraction must be monotonic and within [0, 1]")
        if cumulative_spend < 0:
            raise ValueError("cumulative_spend cannot be negative")

        error = elapsed_fraction - cumulative_spend / self.budget
        delta = elapsed_fraction - self._last_fraction
        if delta > 0:
            self._integral += error * delta
            self._integral = min(max(self._integral, -self.integral_limit), self.integral_limit)
            derivative = 0.0 if self._last_error is None else (error - self._last_error) / delta
        else:
            derivative = 0.0

        correction = self.kp * error + self.ki * self._integral + self.kd * derivative
        self.multiplier = min(max(1.0 + correction, self.minimum), self.maximum)
        self._last_error = error
        self._last_fraction = elapsed_fraction
        return self.multiplier

    def paced_bid(self, base_bid: float) -> float:
        if base_bid < 0:
            raise ValueError("base_bid cannot be negative")
        return base_bid * self.multiplier
