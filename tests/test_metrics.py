import math

import pytest

from ads_engine.metrics import binary_log_loss, roc_auc


def test_auc_orders_positive_above_negative() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_assigns_average_rank_to_ties() -> None:
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_log_loss_matches_balanced_coin() -> None:
    assert binary_log_loss([0, 1], [0.5, 0.5]) == pytest.approx(math.log(2))
