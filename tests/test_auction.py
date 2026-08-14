from ads_engine.auction import Candidate, run_second_price


def test_highest_effective_bid_wins_and_pays_second_price() -> None:
    result = run_second_price(
        [Candidate("a", 3.0, 0.5), Candidate("b", 1.0, 1.0), Candidate("c", 2.0, 0.6)]
    )
    assert result.winner_id == "a"
    assert result.winning_bid == 1.5
    assert result.clearing_price == 1.2


def test_reserve_filters_candidates() -> None:
    result = run_second_price([Candidate("a", 0.2, 0.5)], reserve=0.5)
    assert result.winner_id is None
    assert result.clearing_price == 0.0


def test_ties_use_stable_ad_id() -> None:
    result = run_second_price([Candidate("z", 1.0, 1.0), Candidate("a", 2.0, 0.5)])
    assert result.winner_id == "a"
