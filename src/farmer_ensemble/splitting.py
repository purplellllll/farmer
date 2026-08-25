from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(slots=True)
class SplitResult:
    folds: list[tuple[np.ndarray, np.ndarray]]
    fold_ids: np.ndarray
    leakage_groups: np.ndarray
    audit: dict[str, Any]


def build_leakage_groups(
    n_samples: int,
    groups: np.ndarray | None,
    seeds: np.ndarray | None,
    group_by_seed: bool,
) -> np.ndarray:
    """Create connected components that never split an episode or reused seed.

    Combining strings such as ``episode + seed`` is insufficient because it can
    still place two rows with the same episode (or seed) in different folds.  A
    union-find over both identifiers preserves both constraints.
    """

    if groups is None and (seeds is None or not group_by_seed):
        raise ValueError("groups or seeds are required for leakage-safe OOF")
    union_find = _UnionFind(n_samples)

    def union_equal(values: np.ndarray | None) -> None:
        if values is None:
            return
        if len(values) != n_samples:
            raise ValueError("group metadata length does not match X")
        first: dict[str, int] = {}
        for index, value in enumerate(values):
            key = repr(value)
            if key in first:
                union_find.union(index, first[key])
            else:
                first[key] = index

    union_equal(None if groups is None else np.asarray(groups))
    if group_by_seed:
        union_equal(None if seeds is None else np.asarray(seeds))
    roots = np.asarray([union_find.find(i) for i in range(n_samples)])
    _, components = np.unique(roots, return_inverse=True)
    return components


def make_grouped_folds(
    y: np.ndarray,
    groups: np.ndarray | None,
    seeds: np.ndarray | None,
    seats: np.ndarray | None,
    n_splits: int,
    random_state: int,
    group_by_seed: bool = True,
) -> SplitResult:
    labels = np.asarray(y)
    leakage_groups = build_leakage_groups(len(labels), groups, seeds, group_by_seed)
    unique_groups = np.unique(leakage_groups)
    actual_splits = min(n_splits, len(unique_groups))
    if actual_splits < 2:
        raise ValueError("at least two independent episode/seed groups are required")

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=actual_splits, shuffle=True, random_state=random_state
        )
        folds = list(splitter.split(np.zeros((len(labels), 1)), labels, leakage_groups))
        splitter_name = "StratifiedGroupKFold"
    except (ImportError, ValueError):
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=actual_splits)
        folds = list(splitter.split(np.zeros((len(labels), 1)), labels, leakage_groups))
        splitter_name = "GroupKFold"

    all_classes = set(np.unique(labels).tolist())
    fold_ids = np.full(len(labels), -1, dtype=np.int16)
    overlap_checks: list[dict[str, Any]] = []
    for fold_index, (train_index, valid_index) in enumerate(folds):
        missing = all_classes - set(np.unique(labels[train_index]).tolist())
        if missing:
            raise ValueError(
                f"fold {fold_index} training partition is missing classes {sorted(missing)}; "
                "collect those classes across more independent episodes"
            )
        fold_ids[valid_index] = fold_index
        group_overlap = set(leakage_groups[train_index]) & set(leakage_groups[valid_index])
        seed_overlap: set[Any] = set()
        if group_by_seed and seeds is not None:
            seed_values = np.asarray(seeds)
            seed_overlap = set(seed_values[train_index]) & set(seed_values[valid_index])
        overlap_checks.append(
            {
                "fold": fold_index,
                "group_overlap": len(group_overlap),
                "seed_overlap": len(seed_overlap),
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
            }
        )
    if np.any(fold_ids < 0):
        raise RuntimeError("OOF splitter did not assign every row")

    seat_audit: dict[str, Any] = {"provided": seats is not None}
    if seats is not None:
        seat_values = np.asarray(seats)
        if len(seat_values) != len(labels):
            raise ValueError("seats length does not match X")
        # A component is atomic, so mirrored seat samples in it cannot leak.
        multi_seat_components = 0
        for component in unique_groups:
            if len(np.unique(seat_values[leakage_groups == component])) > 1:
                multi_seat_components += 1
        seat_audit.update(
            {
                "unique_seats": [str(v) for v in np.unique(seat_values)],
                "components_with_multiple_seats": multi_seat_components,
                "mirrored_seats_kept_together": True,
            }
        )

    return SplitResult(
        folds=[(np.asarray(a), np.asarray(b)) for a, b in folds],
        fold_ids=fold_ids,
        leakage_groups=leakage_groups,
        audit={
            "splitter": splitter_name,
            "requested_splits": n_splits,
            "actual_splits": actual_splits,
            "independent_components": len(unique_groups),
            "folds": overlap_checks,
            "seat_audit": seat_audit,
        },
    )
