from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ads_engine.data import EventTable, load_npz, synthetic_events
from ads_engine.metrics import binary_log_loss, roc_auc
from ads_engine.models import DCNv2ESMM, ModelDimensions, TwoTower


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split(events: EventTable, validation_fraction: float = 0.2) -> tuple[EventTable, EventTable]:
    boundary = int(len(events) * (1.0 - validation_fraction))
    train_indices = np.arange(boundary)
    validation_indices = np.arange(boundary, len(events))
    return events.take(train_indices), events.take(validation_indices)


def _loader(*arrays: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = tuple(torch.from_numpy(array) for array in arrays)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)


def train_retrieval(
    events: EventTable, dimensions: ModelDimensions, epochs: int, device: torch.device
) -> TwoTower:
    positives = events.clicked.astype(bool)
    user_features = events.user_features[positives]
    ad_features = events.ad_features[positives]
    if len(user_features) < 2:
        raise ValueError("retrieval training requires at least two clicked impressions")

    model = TwoTower(dimensions.user, dimensions.ad).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    batches = _loader(user_features, ad_features, batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for users, ads in batches:
            loss = model.in_batch_loss(users.to(device), ads.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def train_ranker(
    events: EventTable, dimensions: ModelDimensions, epochs: int, device: torch.device
) -> DCNv2ESMM:
    model = DCNv2ESMM(dimensions.ranking).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    batches = _loader(
        events.ranking_features,
        events.clicked,
        events.converted,
        batch_size=512,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for features, clicked, converted in batches:
            loss = model.loss(features.to(device), clicked.to(device), converted.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def evaluate_ranker(model: DCNv2ESMM, events: EventTable, device: torch.device) -> dict[str, float]:
    model.eval()
    features = torch.from_numpy(events.ranking_features).to(device)
    pctr, _, pctcvr = model(features)
    click_scores = pctr.cpu().numpy().tolist()
    conversion_scores = pctcvr.cpu().numpy().tolist()
    click_labels = events.clicked.astype(int).tolist()
    conversion_labels = events.converted.astype(int).tolist()
    metrics = {
        "click_auc": roc_auc(click_labels, click_scores),
        "click_log_loss": binary_log_loss(click_labels, click_scores),
        "conversion_log_loss": binary_log_loss(conversion_labels, conversion_scores),
    }
    if 0 < sum(conversion_labels) < len(conversion_labels):
        metrics["conversion_auc"] = roc_auc(conversion_labels, conversion_scores)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retrieval and ranking models")
    parser.add_argument("--input", type=Path, help="optional NPZ event table")
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    dimensions = ModelDimensions()
    events = load_npz(args.input) if args.input else synthetic_events(args.events, seed=args.seed)
    training, validation = split(events)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    retrieval = train_retrieval(training, dimensions, args.epochs, device)
    ranker = train_ranker(training, dimensions, args.epochs, device)
    metrics = evaluate_ranker(ranker, validation, device)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(retrieval.state_dict(), args.output / "two_tower.pt")
    torch.save(ranker.state_dict(), args.output / "ranker.pt")
    summary = {
        "device": str(device),
        "events": len(events),
        "train_events": len(training),
        "validation_events": len(validation),
        "seed": args.seed,
        "epochs": args.epochs,
        "metrics": metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
