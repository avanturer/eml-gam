"""Neural beam search for EML snap configurations.

This module implements a small proof-of-concept supervised model that
takes a sample of `(x, y)` pairs drawn from an unknown depth-d EML
tree and predicts the slot choices of that tree. The result is a
beam-search routine that on depth-3 targets recovers the ground-truth
snap at a non-trivial rate despite random-initialisation gradient
descent reaching 0 per cent on the same set.

Architecture
------------
* Input: a fixed-size tensor of ``n_samples`` sorted ``(x, y)`` pairs.
  Sorting by ``x`` removes the permutation ambiguity that would
  otherwise double the effective training-set size.
* Encoder: two-layer MLP over the flattened sample.
* Heads: one independent linear head per slot in the target tree,
  producing a softmax distribution over its option space.

Loss is plain cross-entropy summed across slots, which treats the
slot predictions as conditionally independent given the sample. This
is a deliberate simplification — an autoregressive decoder would be
more expressive but also considerably more expensive to train. A
beam-search over the product of per-slot distributions recovers the
joint configurations at evaluation time.

This is intentionally a minimal proof of concept. The goal is to
demonstrate that a learnable prior reduces the effective search space
by orders of magnitude relative to random initialisation, not to
provide a production-grade symbolic-regression solver.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .eml_tree import EMLTree


@dataclass
class SnapSchema:
    """Shape of the snap search space for a fixed depth / n_inputs tree."""

    depth: int
    n_inputs: int

    def slot_option_counts(self) -> list[int]:
        """Return the number of options per slot in level-major order."""
        counts: list[int] = []
        for level in range(self.depth):
            is_bottom = level == self.depth - 1
            n_opts = (1 + self.n_inputs) if is_bottom else (2 + self.n_inputs)
            n_slots = 2 ** (level + 1)
            counts.extend([n_opts] * n_slots)
        return counts

    @property
    def n_slots(self) -> int:
        return sum(2 ** (level + 1) for level in range(self.depth))

    def flatten_snap(self, snap: dict[int, torch.Tensor]) -> list[int]:
        out: list[int] = []
        for level in range(self.depth):
            out.extend(snap[level].tolist())
        return out

    def unflatten_snap(self, flat: list[int]) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        cursor = 0
        for level in range(self.depth):
            n_slots = 2 ** (level + 1)
            out[level] = torch.tensor(
                flat[cursor : cursor + n_slots], dtype=torch.long
            )
            cursor += n_slots
        return out


def sample_random_snap(
    schema: SnapSchema, rng: random.Random
) -> list[int]:
    """Uniform sample from the product of slot option sets."""
    return [rng.randrange(n) for n in schema.slot_option_counts()]


def tree_from_flat(schema: SnapSchema, flat: list[int]) -> EMLTree:
    tree = EMLTree(
        depth=schema.depth,
        n_inputs=schema.n_inputs,
        use_input_affine=False,
    )
    tree.set_snap_config(schema.unflatten_snap(flat))
    return tree


def make_dataset(
    schema: SnapSchema,
    n_configs: int,
    n_samples_per_config: int = 32,
    x_range: Tuple[float, float] = (0.3, 2.5),
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a labelled dataset of ``(sample_tensor, snap_labels)`` pairs.

    ``sample_tensor`` has shape ``(n_configs, 2 * n_samples_per_config)``
    and contains the flattened `(x, y)` points sorted by ``x``. The raw
    ``y`` is normalised per-config to zero mean and unit variance to
    remove the dependence on multiplicative and additive scale.
    ``snap_labels`` has shape ``(n_configs, n_slots)`` with dtype long.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    slot_counts = schema.slot_option_counts()
    features_per = 2 * n_samples_per_config
    X = torch.zeros(n_configs, features_per, dtype=torch.float32)
    Y = torch.zeros(n_configs, len(slot_counts), dtype=torch.long)

    for i in range(n_configs):
        flat = sample_random_snap(schema, rng)
        tree = tree_from_flat(schema, flat)
        x_vals = torch.empty(
            n_samples_per_config, schema.n_inputs, dtype=torch.float64
        ).uniform_(*x_range)
        with torch.no_grad():
            y_vals = tree(x_vals)
        # Normalise y.
        y_mean = y_vals.mean()
        y_std = y_vals.std().clamp(min=1e-6)
        y_norm = (y_vals - y_mean) / y_std

        # Sort by x_0, concatenate (x_0, y_norm).
        order = torch.argsort(x_vals[:, 0])
        x_sorted = x_vals[order, 0]
        y_sorted = y_norm[order]
        X[i, :n_samples_per_config] = x_sorted.float()
        X[i, n_samples_per_config:] = y_sorted.float()
        Y[i] = torch.tensor(flat, dtype=torch.long)
    return X, Y


class SlotPredictor(nn.Module):
    """Feed-forward predictor from a sorted sample to independent slot
    distributions."""

    def __init__(
        self,
        input_dim: int,
        slot_counts: list[int],
        hidden: int = 256,
    ):
        super().__init__()
        self.slot_counts = slot_counts
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden, k) for k in slot_counts]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = self.encoder(x)
        return [head(h) for head in self.heads]


def train_predictor(
    schema: SnapSchema,
    n_train: int = 50_000,
    n_val: int = 2_000,
    epochs: int = 12,
    batch_size: int = 256,
    lr: float = 2e-3,
    n_samples_per_config: int = 32,
    hidden: int = 256,
    device: str = "cpu",
    seed: int = 0,
) -> Tuple[SlotPredictor, dict]:
    """Build a training dataset, fit the predictor, and report metrics."""
    print(f"building train set: n_configs={n_train} n_samples={n_samples_per_config}")
    X_tr, Y_tr = make_dataset(
        schema, n_train, n_samples_per_config=n_samples_per_config, seed=seed
    )
    X_val, Y_val = make_dataset(
        schema,
        n_val,
        n_samples_per_config=n_samples_per_config,
        seed=seed + 1,
    )

    model = SlotPredictor(
        input_dim=X_tr.shape[1],
        slot_counts=schema.slot_option_counts(),
        hidden=hidden,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []
    X_tr = X_tr.to(device)
    Y_tr = Y_tr.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    for epoch in range(epochs):
        perm = torch.randperm(X_tr.shape[0], device=device)
        total_loss = 0.0
        for start in range(0, X_tr.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            logits = model(X_tr[idx])
            loss = sum(
                F.cross_entropy(lgt, Y_tr[idx, s])
                for s, lgt in enumerate(logits)
            ) / len(logits)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * idx.shape[0]
        total_loss /= X_tr.shape[0]

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            correct_per_slot = [
                (lgt.argmax(dim=1) == Y_val[:, s]).float().mean().item()
                for s, lgt in enumerate(val_logits)
            ]
            all_correct = torch.stack(
                [
                    (lgt.argmax(dim=1) == Y_val[:, s]).long()
                    for s, lgt in enumerate(val_logits)
                ],
                dim=1,
            ).all(dim=1).float().mean().item()
        model.train()

        mean_slot_acc = sum(correct_per_slot) / len(correct_per_slot)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss,
                "val_mean_slot_acc": mean_slot_acc,
                "val_joint_acc": all_correct,
            }
        )
        print(
            f"  epoch {epoch:2d} | loss={total_loss:.3f} | "
            f"val_slot_acc={mean_slot_acc:.3f} | val_joint={all_correct:.3f}"
        )
    model.eval()
    return model, {"history": history}


def beam_search_snaps(
    model: SlotPredictor,
    schema: SnapSchema,
    sample_tensor: torch.Tensor,
    beam_width: int = 16,
) -> list[Tuple[list[int], float]]:
    """Return the ``beam_width`` joint snap configurations ranked by
    product of per-slot log-probabilities."""
    with torch.no_grad():
        logits = model(sample_tensor.unsqueeze(0))
    log_probs = [F.log_softmax(lgt[0], dim=-1) for lgt in logits]
    beams: list[Tuple[list[int], float]] = [([], 0.0)]
    for slot, lp in enumerate(log_probs):
        k = lp.shape[0]
        expanded: list[Tuple[list[int], float]] = []
        for partial, score in beams:
            for option in range(k):
                expanded.append((partial + [option], score + float(lp[option])))
        expanded.sort(key=lambda t: t[1], reverse=True)
        beams = expanded[:beam_width]
    return beams


def evaluate_beam_recovery(
    model: SlotPredictor,
    schema: SnapSchema,
    n_targets: int = 200,
    beam_width: int = 16,
    n_samples_per_config: int = 32,
    seed: int = 42,
) -> dict:
    """Draw ``n_targets`` random ground-truth snaps, run beam search for
    each, and report how often the ground truth is in the top-``beam_width``
    beam."""
    X_test, Y_test = make_dataset(
        schema, n_targets, n_samples_per_config=n_samples_per_config, seed=seed
    )
    top1 = 0
    topk = 0
    for i in range(n_targets):
        beams = beam_search_snaps(
            model, schema, X_test[i], beam_width=beam_width
        )
        truth = Y_test[i].tolist()
        in_beam = any(b[0] == truth for b in beams)
        is_top1 = beams[0][0] == truth
        top1 += int(is_top1)
        topk += int(in_beam)
    return {
        "n_targets": n_targets,
        "beam_width": beam_width,
        "top1": top1 / n_targets,
        f"top{beam_width}": topk / n_targets,
    }


if __name__ == "__main__":
    schema = SnapSchema(depth=3, n_inputs=1)
    print(f"schema: depth=3 univariate; slot option counts = {schema.slot_option_counts()}")
    print(f"total snap space size = {np.prod(schema.slot_option_counts())}")

    model, training_info = train_predictor(
        schema,
        n_train=40_000,
        n_val=1_000,
        epochs=10,
        hidden=256,
        seed=0,
    )
    print("\nBeam-search evaluation on held-out random snaps:")
    for beam_width in (1, 8, 32, 128):
        metrics = evaluate_beam_recovery(
            model,
            schema,
            n_targets=300,
            beam_width=beam_width,
            seed=1000,
        )
        print(
            f"  beam_width={beam_width:4d} | top1={metrics['top1']:.3f} | "
            f"top-{beam_width}={metrics[f'top{beam_width}']:.3f}"
        )
