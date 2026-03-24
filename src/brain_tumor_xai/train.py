from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .evaluation import collect_predictions, compute_binary_classification_metrics
from .model import save_checkpoint
from .utils import ensure_dir, save_json


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def evaluate_loss(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / max(total_examples, 1)


def fit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    target = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(Path(target) / "checkpoints")
    history_path = Path(target) / "history.json"
    checkpoint_path = checkpoints_dir / "best.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    history: list[dict[str, Any]] = []
    best_f1 = float("-inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        labels, probabilities, _ = collect_predictions(model, val_loader, device)
        metrics = compute_binary_classification_metrics(labels, probabilities)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }
        history.append(record)

        if record["f1"] > best_f1:
            best_f1 = record["f1"]
            save_checkpoint(
                model,
                str(checkpoint_path),
                extra={
                    "epoch": epoch,
                    "metrics": metrics,
                },
            )

    save_json({"history": history}, history_path)
    return {
        "history": history,
        "best_checkpoint": str(checkpoint_path),
    }
