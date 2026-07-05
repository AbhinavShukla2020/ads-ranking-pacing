from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.normalize(self.layers(features), dim=-1)


class TwoTower(nn.Module):
    def __init__(self, user_dim: int, ad_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.user_tower = Tower(user_dim, embedding_dim=embedding_dim)
        self.ad_tower = Tower(ad_dim, embedding_dim=embedding_dim)
        self.log_temperature = nn.Parameter(torch.tensor(2.3))

    def forward(self, user_features: Tensor, ad_features: Tensor) -> tuple[Tensor, Tensor]:
        return self.user_tower(user_features), self.ad_tower(ad_features)

    def in_batch_loss(self, user_features: Tensor, ad_features: Tensor) -> Tensor:
        users, ads = self(user_features, ad_features)
        temperature = self.log_temperature.exp().clamp(1.0, 100.0)
        logits = users @ ads.T * temperature
        targets = torch.arange(len(users), device=users.device)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


class CrossLayer(nn.Module):
    """Full-rank DCNv2 cross layer: x_(l+1) = x0 * (W xl + b) + xl."""

    def __init__(self, dimensions: int):
        super().__init__()
        self.projection = nn.Linear(dimensions, dimensions)

    def forward(self, x0: Tensor, current: Tensor) -> Tensor:
        return x0 * self.projection(current) + current


class DCNv2ESMM(nn.Module):
    def __init__(self, input_dim: int, cross_layers: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.cross = nn.ModuleList(CrossLayer(input_dim) for _ in range(cross_layers))
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        combined_dim = input_dim + hidden_dim // 2
        self.ctr_head = nn.Linear(combined_dim, 1)
        self.cvr_head = nn.Linear(combined_dim, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        crossed = features
        for layer in self.cross:
            crossed = layer(features, crossed)
        representation = torch.cat((crossed, self.deep(features)), dim=-1)
        ctr_logit = self.ctr_head(representation).squeeze(-1)
        cvr_logit = self.cvr_head(representation).squeeze(-1)
        pctr = torch.sigmoid(ctr_logit)
        pcvr_given_click = torch.sigmoid(cvr_logit)
        return pctr, pcvr_given_click, pctr * pcvr_given_click

    def loss(self, features: Tensor, clicked: Tensor, converted: Tensor) -> Tensor:
        pctr, _, pctcvr = self(features)
        ctr_loss = F.binary_cross_entropy(pctr, clicked.float())
        conversion_loss = F.binary_cross_entropy(pctcvr, converted.float())
        return ctr_loss + conversion_loss


@dataclass(frozen=True)
class ModelDimensions:
    user: int = 16
    ad: int = 16
    context: int = 8

    @property
    def ranking(self) -> int:
        return self.user + self.ad + self.context
