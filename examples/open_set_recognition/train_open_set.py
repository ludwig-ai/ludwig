#!/usr/bin/env python3
"""Reproduce Dhamija et al. NeurIPS 2018 "Reducing Network Agnostophobia".

Trains three classifiers on a synthetic Gaussian dataset:
  - CE Baseline (SoftmaxCrossEntropy)
  - Entropic Open-Set Loss
  - Objectosphere Loss

then reports the mean max-probability on held-out background/unknown
samples.  Lower max-prob = better open-set recognition.

Usage:
    python train_open_set.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-10


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def make_dataset(seed: int = 42):
    """Four known Gaussian clusters + two unknown clusters in 2D."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    known_centers = torch.tensor([[3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]])
    unknown_centers = torch.tensor([[0.0, 0.0], [6.0, 0.0]])
    std = 0.8

    n_known = 200  # per class
    n_unknown = 100  # per class (background)

    known_x, known_y = [], []
    for cls_idx, center in enumerate(known_centers):
        pts = center + std * torch.randn(n_known, 2, generator=rng)
        known_x.append(pts)
        known_y.append(torch.full((n_known,), cls_idx, dtype=torch.long))

    unknown_x = []
    for center in unknown_centers:
        pts = center + std * torch.randn(n_unknown, 2, generator=rng)
        unknown_x.append(pts)

    # Background class index = 4
    BG = 4
    unknown_y_bg = torch.full((n_unknown * 2,), BG, dtype=torch.long)

    X_known = torch.cat(known_x)
    y_known = torch.cat(known_y)
    X_unknown = torch.cat(unknown_x)

    # Training set: all known + all unknown (for agnostophobia models)
    X_train = torch.cat([X_known, X_unknown])
    y_train = torch.cat([y_known, unknown_y_bg])

    # Test set: fresh samples from known and unknown distributions
    test_known_x, test_known_y = [], []
    for cls_idx, center in enumerate(known_centers):
        pts = center + std * torch.randn(50, 2, generator=rng)
        test_known_x.append(pts)
        test_known_y.append(torch.full((50,), cls_idx, dtype=torch.long))

    test_unknown_x = []
    for center in unknown_centers:
        pts = center + std * torch.randn(50, 2, generator=rng)
        test_unknown_x.append(pts)

    X_test_known = torch.cat(test_known_x)
    y_test_known = torch.cat(test_known_y)
    X_test_unknown = torch.cat(test_unknown_x)

    return X_train, y_train, X_test_known, y_test_known, X_test_unknown, BG


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden: int = 64, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss functions  (mirrors the Ludwig module implementations)
# ---------------------------------------------------------------------------


class EntropicOpenSetLoss(nn.Module):
    def __init__(self, background_class: int):
        super().__init__()
        self.bg = background_class

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        known_mask = target != self.bg
        unknown_mask = ~known_mask
        loss = logits.new_tensor(0.0)
        if known_mask.any():
            loss = loss + F.cross_entropy(logits[known_mask], target[known_mask])
        if unknown_mask.any():
            probs = torch.softmax(logits[unknown_mask], dim=-1)
            loss = loss + (probs * torch.log(probs + EPSILON)).sum(dim=-1).mean()
        return loss


class ObjectosphereLoss(nn.Module):
    def __init__(self, background_class: int, xi: float = 10.0, zeta: float = 0.1):
        super().__init__()
        self.bg = background_class
        self.xi = xi
        self.zeta = zeta

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        known_mask = target != self.bg
        unknown_mask = ~known_mask
        loss = logits.new_tensor(0.0)
        if known_mask.any():
            kl = logits[known_mask]
            ce = F.cross_entropy(kl, target[known_mask])
            hinge = torch.clamp(self.xi - kl.norm(dim=-1), min=0.0).pow(2).mean()
            loss = loss + ce + hinge
        if unknown_mask.any():
            ul = logits[unknown_mask]
            probs = torch.softmax(ul, dim=-1)
            neg_entropy = (probs * torch.log(probs + EPSILON)).sum(dim=-1).mean()
            loss = loss + neg_entropy + self.zeta * ul.norm(dim=-1).pow(2).mean()
        return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module, loss_fn: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 200, lr: float = 1e-3
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def mean_max_prob(model: nn.Module, X: torch.Tensor) -> float:
    model.eval()
    logits = model(X)
    return torch.softmax(logits, dim=-1).max(dim=-1).values.mean().item()


@torch.no_grad()
def mean_norm(model: nn.Module, X: torch.Tensor) -> float:
    model.eval()
    return model(X).norm(dim=-1).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(0)
    X_train, y_train, X_test_known, y_test_known, X_test_unknown, BG = make_dataset()

    # Baseline: train only on known samples with standard CE
    X_known_train = X_train[y_train != BG]
    y_known_train = y_train[y_train != BG]

    configs = [
        (
            "CE Baseline",
            MLP(n_classes=4),  # no background class in output
            nn.CrossEntropyLoss(),
            X_known_train,
            y_known_train,
        ),
        (
            "Entropic Open-Set",
            MLP(n_classes=5),  # includes background class output node
            EntropicOpenSetLoss(background_class=BG),
            X_train,
            y_train,
        ),
        (
            "Objectosphere",
            MLP(n_classes=5),
            ObjectosphereLoss(background_class=BG, xi=10.0, zeta=0.1),
            X_train,
            y_train,
        ),
    ]

    results = []
    for name, model, loss_fn, X, y in configs:
        torch.manual_seed(0)
        train(model, loss_fn, X, y, epochs=300)
        mmp_known = mean_max_prob(model, X_test_known)
        mmp_unknown = mean_max_prob(model, X_test_unknown)
        norm_known = mean_norm(model, X_test_known)
        norm_unknown = mean_norm(model, X_test_unknown)
        results.append((name, mmp_known, mmp_unknown, norm_known, norm_unknown))

    cols = ("Model", "Max-prob (known)", "Max-prob (unknown)", "Norm known", "Norm unknown")
    header = f"{cols[0]:<22} | {cols[1]:>16} | {cols[2]:>18} | {cols[3]:>10} | {cols[4]:>12}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, mpk, mpu, nk, nu in results:
        print(f"{name:<22} | {mpk:>16.3f} | {mpu:>18.3f} | {nk:>10.3f} | {nu:>12.3f}")
    print(sep)
    print()
    print("Expected behaviour from the paper:")
    print("  - CE Baseline:        high max-prob on unknowns (≈0.7-0.9)")
    print("  - Entropic Open-Set:  lower max-prob on unknowns (≈0.2-0.3, near uniform)")
    print("  - Objectosphere:      similar to entropic, plus norm(known) >> norm(unknown)")

    # --- Assertions so this script can double as a smoke test ---
    ce_unknown = results[0][2]
    eos_unknown = results[1][2]
    obj_unknown = results[2][2]

    assert (
        eos_unknown < ce_unknown
    ), f"Entropic loss should reduce unknown confidence: {eos_unknown:.3f} < {ce_unknown:.3f}"
    assert (
        obj_unknown < ce_unknown
    ), f"Objectosphere loss should reduce unknown confidence: {obj_unknown:.3f} < {ce_unknown:.3f}"

    obj_norm_known = results[2][3]
    obj_norm_unknown = results[2][4]
    assert (
        obj_norm_known > obj_norm_unknown * 1.5
    ), f"Objectosphere should create norm gap: known={obj_norm_known:.3f} unknown={obj_norm_unknown:.3f}"

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
