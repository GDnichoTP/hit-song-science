import numpy as np
import pandas as pd
from pathlib import Path
# import json
from typing import Tuple
from config import DATA_DIR, TARGET


def load_dataset(week: int, subset: str, folder: Path = DATA_DIR) -> pd.DataFrame:
    """Read ``dataset_n{week}_{subset}.csv``; `subset` is ``'all'``, ``'old'`` or ``'new'``."""
    return pd.read_csv(Path(folder) / f"dataset_n{week}_{subset}.csv")


def get_xy(df: pd.DataFrame, features: list[str], target: str = TARGET) -> Tuple[np.ndarray, np.ndarray]:
    """Return the `features` columns as a float array and the `target` column as labels."""
    return df[features].astype(float).to_numpy(), df[target].to_numpy()


def stratified_train_test_split(
        X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int | None = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split into train and test sets with the same class proportions in both.

    Each class is shuffled and its first ``round(test_size * n_class)``
    samples go to the test set.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.

    Raises:
        ValueError: If `test_size` is not between 0 and 1.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"'test_size'({test_size}) must be between 0 and 1")
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(random_state)
    test_idx = []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        test_idx.append(idx[:round(test_size * len(idx))])
    is_test = np.zeros(len(y), dtype=bool)
    is_test[np.concatenate(test_idx)] = True
    return X[~is_test], X[is_test], y[~is_test], y[is_test]

def fit_scaler(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and std, and optionally save them as JSON.

    Fit on the training features only, then reuse the result for the test
    features, so no test information leaks into training.

    Args:
        data: Training features of shape ``(n_samples, n_features)``.
        path: JSON file to save to. ``None`` skips saving.

    Returns:
        ``(mean, std)``, each of shape ``(n_features,)``. A std of zero
        (constant feature) is replaced by 1 to avoid dividing by zero.
    """
    mean = np.mean(data, axis=0, dtype=np.float32)
    std = np.std(data, axis=0, dtype=np.float32)
    std[std == 0] = 1.0

    # if path is not None:
    #     with open(path, "w") as f:
    #         json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    return mean, std

def z_score_standardization(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Scale features to zero mean and unit variance: ``(data - mean) / std``.

    Args:
        data: Features of shape ``(n_samples, n_features)``.
        mean: Per-feature mean from `fit_scaler` or `load_scaler`.
        std: Per-feature std from `fit_scaler` or `load_scaler`.

    Returns:
        float32 array with the same shape as `data`.
    """
    return ((data - mean) / std).astype(np.float32)
