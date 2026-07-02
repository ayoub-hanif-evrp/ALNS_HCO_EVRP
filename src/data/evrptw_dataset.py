"""Dataset wrapper and batching utilities for E-VRPTW instances."""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .instance_loader import (
    NODE_FEATURE_COLUMNS,
    EVRPTWInstance,
    list_instance_dirs,
    load_instance,
)


def _passes_filter(name: str, filters: list[str] | None) -> bool:
    if not filters:
        return True
    return any(f in name for f in filters)


class EVRPTWDataset(Dataset):
    """Lazily loads Schneider-style E-VRPTW instances from converted folders."""

    def __init__(
        self,
        dataset_path: str | Path,
        size_filter: list[str] | None = None,
        *,
        load_capacity: float | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.load_capacity = load_capacity
        all_dirs = list_instance_dirs(self.dataset_path)
        self.dirs = [d for d in all_dirs if _passes_filter(d.name, size_filter)]
        if not self.dirs:
            raise ValueError(
                f"No instances found in {self.dataset_path} with filter {size_filter}"
            )
        self._cache: dict[int, EVRPTWInstance] = {}

    def __len__(self) -> int:
        return len(self.dirs)

    def _load_at(self, path: Path) -> EVRPTWInstance:
        return load_instance(path, load_capacity=self.load_capacity)

    def __getitem__(self, idx: int) -> EVRPTWInstance:
        if idx not in self._cache:
            self._cache[idx] = self._load_at(self.dirs[idx])
        return self._cache[idx]

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.dirs]


def _resolve_name_indices(
    full: EVRPTWDataset,
    names: list[str],
    *,
    label: str,
) -> list[int]:
    """Map instance names to indices in ``full``, preserving list order."""
    name_to_idx = {full.names[i]: i for i in range(len(full))}
    indices: list[int] = []
    for name in names:
        if name not in name_to_idx:
            raise ValueError(f"{label} instance not found in dataset: {name}")
        indices.append(name_to_idx[name])
    return indices


def split_dataset(
    dataset_path: str | Path,
    train_ratio: float,
    seed: int,
    train_filter: list[str] | None = None,
    eval_filter: list[str] | None = None,
    *,
    train_instances: list[str] | None = None,
    eval_instances: list[str] | None = None,
    load_capacity: float | None = None,
) -> tuple[EVRPTWDataset, EVRPTWDataset]:
    """Split instances into train / eval subsets.

    When ``train_instances`` is set, use that explicit train list. ``eval_instances``
    may be provided; otherwise every remaining instance in the loaded pool becomes eval.
    """
    base_filter = train_filter if train_filter == eval_filter else None
    full = EVRPTWDataset(
        dataset_path,
        size_filter=base_filter,
        load_capacity=load_capacity,
    )

    if train_instances is not None:
        train_names = list(train_instances)
        train_idx = _resolve_name_indices(full, train_names, label="train")
        train_name_set = set(train_names)
        if eval_instances is not None:
            eval_names = list(eval_instances)
            overlap = train_name_set & set(eval_names)
            if overlap:
                raise ValueError(f"train and eval instance lists overlap: {sorted(overlap)}")
            eval_idx = _resolve_name_indices(full, eval_names, label="eval")
        else:
            eval_idx = [i for i in range(len(full)) if full.names[i] not in train_name_set]
        train_ds = _SubsetDataset(full, train_idx, train_filter)
        eval_ds = _SubsetDataset(full, sorted(eval_idx), eval_filter)
        return train_ds, eval_ds

    indices = list(range(len(full)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_train = int(round(train_ratio * len(indices)))
    train_idx = set(indices[:n_train])

    train_ds = _SubsetDataset(full, sorted(train_idx), train_filter)
    eval_idx = sorted(set(indices) - train_idx)
    eval_ds = _SubsetDataset(full, eval_idx, eval_filter)
    return train_ds, eval_ds


class _SubsetDataset(Dataset):
    """A filtered view over an EVRPTWDataset."""

    def __init__(self, base: EVRPTWDataset, indices: list[int], extra_filter: list[str] | None):
        self.base = base
        if extra_filter:
            indices = [i for i in indices if _passes_filter(base.names[i], extra_filter)]
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> EVRPTWInstance:
        return self.base[self.indices[idx]]

    @property
    def names(self) -> list[str]:
        return [self.base.names[i] for i in self.indices]

    @property
    def dataset_path(self) -> Path:
        return self.base.dataset_path


def collate_instances(
    instances: list[EVRPTWInstance], device: torch.device | str = "cpu"
) -> dict[str, torch.Tensor]:
    """Pad a list of instances to a common node count for batched encoding."""
    device = torch.device(device)
    batch_size = len(instances)
    num_feats = len(NODE_FEATURE_COLUMNS)
    max_nodes = max(inst.num_nodes for inst in instances)

    node_features = torch.zeros(batch_size, max_nodes, num_feats, device=device)
    node_padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool, device=device)
    num_nodes = torch.zeros(batch_size, dtype=torch.long, device=device)

    for b, inst in enumerate(instances):
        n = inst.num_nodes
        feats = torch.tensor(inst.node_feature_matrix(), device=device)
        node_features[b, :n] = feats
        node_padding_mask[b, :n] = False
        num_nodes[b] = n

    return {
        "node_features": node_features,
        "node_padding_mask": node_padding_mask,
        "num_nodes": num_nodes,
    }
