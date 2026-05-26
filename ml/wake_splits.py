from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_train_val(
    idx_train: np.ndarray,
    strata: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if val_ratio <= 0 or idx_train.shape[0] < 2:
        return idx_train, np.array([], dtype=int)
    train_strata = strata[idx_train]
    idx_tr, idx_val = train_test_split(
        idx_train, test_size=val_ratio, random_state=seed, stratify=train_strata
    )
    return np.sort(idx_tr), np.sort(idx_val)
