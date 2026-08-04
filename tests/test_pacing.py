import pytest

from ads_engine.pacing import PIDPacer


def test_multiplier_increases_when_campaign_is_behind() -> None:
    pacer = PIDPacer(budget=100.0)
    assert pacer.update(0.5, 20.0) > 1.0


def test_multiplier_decreases_when_campaign_is_ahead() -> None:
    pacer = PIDPacer(budget=100.0)
    assert pacer.update(0.5, 80.0) < 1.0


def test_multiplier_stays_inside_bounds() -> None:
    pacer = PIDPacer(budget=100.0, kp=100.0, minimum=0.2, maximum=1.5)
    assert pacer.update(0.9, 0.0) == 1.5


def test_elapsed_fraction_must_be_monotonic() -> None:
    pacer = PIDPacer(budget=100.0)
    pacer.update(0.5, 40.0)
    with pytest.raises(ValueError):
        pacer.update(0.4, 45.0)
