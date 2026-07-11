import pytest

torch = pytest.importorskip("torch")

from ads_engine.models import DCNv2ESMM, TwoTower


def test_two_tower_outputs_normalized_embeddings() -> None:
    model = TwoTower(user_dim=4, ad_dim=6, embedding_dim=8)
    users, ads = model(torch.randn(3, 4), torch.randn(3, 6))
    assert users.shape == (3, 8)
    assert ads.shape == (3, 8)
    assert torch.linalg.vector_norm(users, dim=1).tolist() == pytest.approx([1.0] * 3)


def test_esmm_joint_probability_cannot_exceed_ctr() -> None:
    model = DCNv2ESMM(input_dim=10)
    pctr, pcvr, pctcvr = model(torch.randn(5, 10))
    assert torch.all(pctcvr <= pctr)
    assert torch.all((pcvr >= 0) & (pcvr <= 1))
