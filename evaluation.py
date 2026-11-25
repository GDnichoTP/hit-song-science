import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Sequence, Tuple
import contextlib
import inspect
import io
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools
from model import Standardized, Model
from config import C_GRID, K_FOLDS, SEED


def confusion_matrix(predictions: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count how often each true class was predicted as each class.

    Args:
        predictions: Predicted labels.
        y: True labels, same length as `predictions`.

    Returns:
        ``(cm, labels)``: ``cm[i, j]`` is the number of samples of class
        ``labels[i]`` predicted as ``labels[j]``; `labels` is sorted.
    """
    unique_labels = np.union1d(y, predictions)
    n_labels = len(unique_labels)

    cm = np.zeros((n_labels, n_labels), dtype=int)
    for i, true_label in enumerate(unique_labels):
        masked = predictions[y == true_label]          # predictions for this true class
        for j, pred_label in enumerate(unique_labels):
            cm[i, j] = np.sum(masked == pred_label)

    return cm, unique_labels

def show_confusion_matrix(cm: np.ndarray, labels: np.ndarray) -> None:
    """Plot a confusion matrix from `confusion_matrix` as a heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def check_accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of predictions equal to the true labels."""
    n_test = len(labels)
    right = np.sum(prediction == labels)
    return right/n_test

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def stratified_group_k_fold(
        y: np.ndarray, k: int = 5, groups: np.ndarray | None = None, random_state: int | None = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split sample indices into `k` folds that keep groups whole and classes balanced.

    Every group (e.g. one original clip plus all its augmented copies) goes
    into exactly one fold, so no clip is ever validated against its own
    copy. Groups of each class are shuffled and then dealt out one by one to
    the fold that has the fewest samples of that class so far, which gives
    every fold about the same class proportions as the whole set.

    Args:
        y: Labels, shape ``(n_samples,)``.
        k: Number of folds, at least 2.
        groups: Group of each sample, e.g. ``Features.sources``. ``None``
            makes every sample its own group (plain stratified k-fold).
        random_state: Seed for shuffling the groups.

    Returns:
        ``k`` pairs ``(train_indices, test_indices)``.

    Raises:
        ValueError: If `k` is below 2 or above the number of groups, if
            `groups` differs in length from `y`, or if one group holds more
            than one class.
    """
    y = np.asarray(y)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError(f"'groups' has {len(groups)} entries but 'y' has {len(y)}")

    unique_groups, group_index = np.unique(groups, return_inverse=True)
    if not 2 <= k <= len(unique_groups):
        raise ValueError(f"'k'({k}) must be between 2 and the number of groups ({len(unique_groups)})")

    # One label per group
    group_label = np.empty(len(unique_groups), dtype=y.dtype)
    group_label[group_index] = y
    if np.any(group_label[group_index] != y):
        raise ValueError("Every group must contain a single class")
    group_size = np.bincount(group_index)

    # Deal each class's groups out to the folds
    rng = np.random.default_rng(random_state)
    group_fold = np.empty(len(unique_groups), dtype=int)
    fold_total = np.zeros(k, dtype=int)
    for label in np.unique(group_label):
        class_groups = rng.permutation(np.flatnonzero(group_label == label))
        fold_class = np.zeros(k, dtype=int)
        for g in class_groups:
            # fewest of this class; on a tie, the smallest fold overall
            fold = min(range(k), key=lambda f: (fold_class[f], fold_total[f]))
            group_fold[g] = fold
            fold_class[fold] += group_size[g]
            fold_total[fold] += group_size[g]

    sample_fold = group_fold[group_index]
    return [(np.flatnonzero(sample_fold != fold), np.flatnonzero(sample_fold == fold))
            for fold in range(k)]


def stratified_k_fold_cross_validation(
        model: Model, X: np.ndarray, y: np.ndarray, k: int = 5,
        groups: np.ndarray | None = None, random_state: int | None = 42
) -> Tuple[float, float]:
    """Estimate a model's accuracy with stratified (group) k-fold cross-validation.

    The model is trained on ``k - 1`` folds and scored on the remaining one,
    ``k`` times. Folds come from `stratified_group_k_fold`. Anything the
    model prints while fitting is silenced during the folds.

    Args:
        model: An untrained model; it is refitted on every fold.
        X: Features, shape ``(n_samples, n_features)``.
        y: Labels, shape ``(n_samples,)``.
        k: Number of folds, at least 2.
        groups: Group of each sample, e.g. ``Features.sources``. Pass it
            whenever `X` contains augmented copies.
        random_state: Seed for the fold assignment.

    Returns:
        ``(mean, std)`` of the fold accuracies.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    accuracies = []

    for train_indices, test_indices in stratified_group_k_fold(y, k, groups, random_state):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        with contextlib.redirect_stdout(io.StringIO()):
            trained = model.fit(X_train, y_train)
        prediction = trained.predict(X_test)
        accuracies.append(check_accuracy(prediction, y_test))

    accuracies = np.asarray(accuracies, dtype=float)
    return accuracies.mean(), accuracies.std()


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def create_grid_combinations(
        hyperparams: Dict[str, Sequence], excludes: List[Tuple[Tuple[str, str], Tuple]] | None = None
) -> List[dict]:
    """List every combination of hyperparameter values.

    Args:
        hyperparams: Values to try per parameter, e.g.
            ``{"kernel": ["linear", "rbf"], "C": [1, 10]}``.
        excludes: Rules that drop combinations, each
            ``((key_a, key_b), (value_a, value_b))``: when `key_a` equals
            `value_a`, only `value_b` is kept for `key_b`. For example
            ``(("kernel", "gamma"), ("linear", "scale"))`` tries the linear
            kernel once instead of once per gamma, because gamma does not
            affect it.

    Returns:
        One dict per combination.
    """
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    configs = []

    for combination in itertools.product(*values):
        cfg = dict(zip(keys, combination))
        skip = False
        if excludes is not None:
            for ex_keys, ex_values in excludes:
                if cfg[ex_keys[0]] == ex_values[0] and cfg[ex_keys[1]] != ex_values[1]:
                    skip = True
                    break

        if not skip:
            configs.append(cfg)

    return configs


def grid_search(
        hyperparams: Dict[str, Sequence], model: type, X: np.ndarray, y: np.ndarray,
        excludes: List | None = None, random_state: int = 42,
        groups: np.ndarray | None = None, k: int = 5,
) -> Tuple[dict, float]:
    """Find the hyperparameters with the best cross-validated accuracy.

    Args:
        hyperparams: Values to try per parameter (see `create_grid_combinations`).
        model: The model class, e.g. ``SVM`` or ``KNN`` (not an instance).
        X: Training features, shape ``(n_samples, n_features)``.
        y: Training labels.
        excludes: Combinations to skip (see `create_grid_combinations`).
        random_state: Seed for the folds, and for the model if its
            constructor accepts ``random_state``.
        groups: Group of each sample, e.g. ``Features.sources``. Pass it
            whenever `X` contains augmented copies.
        k: Number of cross-validation folds.

    Returns:
        ``(best_params, best_accuracy)``. On a tie the earlier combination wins.
    """
    combinations = create_grid_combinations(hyperparams, excludes)
    takes_seed = "random_state" in inspect.signature(model).parameters
    best_accuracy = 0.0
    best_params = {}

    for combination in tqdm(combinations, desc="searching for the best hyperparameter combination"):
        if takes_seed:
            my_model = model(random_state=random_state, **combination)
        else:
            my_model = model(**combination)
        my_model = Standardized(model=my_model)
        mean, _ = stratified_k_fold_cross_validation(my_model, X, y, k, groups, random_state)
        if best_accuracy < mean:
            best_params = combination
            best_accuracy = mean

            print(f"new best params: {combination}")
            print(f"new best accuracy: {best_accuracy}")

    return best_params, best_accuracy


# ---------------------------------------------------------------------------
# Experiments: tune C, test, summarize
# ---------------------------------------------------------------------------

def tune_C(
        make_model: Callable[[float], Model], X: np.ndarray, y: np.ndarray,
        C_grid: Sequence[float] = C_GRID, k: int = K_FOLDS, random_state: int | None = SEED,
        desc: str = "",
) -> pd.DataFrame:
    """Cross-validated accuracy for every C.

    Args:
        make_model: Builds an untrained model for a given C, e.g.
            ``lambda C: Standardized(BasicSVM(C=C))``.
        X, y: Training data only (never the test set).

    Returns:
        DataFrame with columns ``C``, ``mean``, ``std`` (one row per C).
    """
    rows = []
    for C in tqdm(C_grid, desc=desc, leave=False):
        mean, std = stratified_k_fold_cross_validation(make_model(C), X, y, k=k, random_state=random_state)
        rows.append({"C": C, "mean": mean, "std": std})
    return pd.DataFrame(rows)


def best_of(curve: pd.DataFrame) -> pd.Series:
    """Row of `curve` with the highest mean accuracy; on a tie the smaller C wins."""
    return curve.loc[curve["mean"].idxmax()]


def tune_and_test(
        make_model: Callable[[float], Model], X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray, desc: str = "", **tune_kwargs,
) -> dict:
    """Tune C with k-fold CV on the training set, retrain on all of it, score on the test set.

    Returns:
        Dict with ``curve`` (from `tune_C`), ``best_C``, ``cv_mean``,
        ``cv_std``, ``test_accuracy``, the trained ``model`` and its test
        ``predictions``.
    """
    curve = tune_C(make_model, X_train, y_train, desc=desc, **tune_kwargs)
    best = best_of(curve)
    model = make_model(best["C"]).fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {"curve": curve, "best_C": best["C"], "cv_mean": best["mean"], "cv_std": best["std"],
            "test_accuracy": check_accuracy(predictions, y_test), "model": model, "predictions": predictions}


def curve_table(curves: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """CV accuracy per C (rows) and scenario (columns) as ``'mean ± std'``; the best C is marked ``*``."""
    table, index = {}, None
    for name, curve in curves.items():
        best_C = best_of(curve)["C"]
        table[name] = [f"{m:.3f} ± {s:.3f}" + (" *" if C == best_C else "")
                       for C, m, s in curve[["C", "mean", "std"]].itertuples(index=False)]
        index = [f"{C:g}" for C in curve["C"]]
    return pd.DataFrame(table, index=pd.Index(index, name="C"))


def summary_table(results: Dict[str, dict], extra: Dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per scenario: best C, CV accuracy ± std (training set) and test accuracy."""
    rows = []
    for name, r in results.items():
        row = {"scenario": name, "best C": f"{r['best_C']:g}", "CV accuracy": round(r["cv_mean"], 4),
               "CV std": round(r["cv_std"], 4), "test accuracy": round(r["test_accuracy"], 4)}
        if extra:
            row.update(extra.get(name, {}))
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


# ---------------------------------------------------------------------------
# Experiments: plots
# ---------------------------------------------------------------------------

# Categorical colors, assigned in fixed order (colorblind-safe palette)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID_COLOR = "#0b0b0b", "#52514e", "#e4e3df"


def set_plot_style() -> None:
    """Quiet axes and grid so the data stands out."""
    plt.rcParams.update({
        "figure.dpi": 110, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID_COLOR,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def plot_c_curves(curves: Dict[str, pd.DataFrame], title: str) -> None:
    """CV accuracy vs C (log scale), one line per scenario; band = ± 1 std, ring = best C."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, (name, curve) in zip(SERIES_COLORS, curves.items()):
        ax.fill_between(curve["C"], curve["mean"] - curve["std"], curve["mean"] + curve["std"],
                        color=color, alpha=0.12, linewidth=0)
        ax.plot(curve["C"], curve["mean"], color=color, lw=2, marker="o", ms=6, label=name)
        best = best_of(curve)
        ax.plot(best["C"], best["mean"], marker="o", ms=11, mfc="none", mec=color, mew=2)
    C_values = list(next(iter(curves.values()))["C"])
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.annotate("chance (50%)", xy=(C_values[0], 0.5), xytext=(0, 4), textcoords="offset points",
                color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(C_values, [f"{C:g}" for C in C_values])
    ax.set_xlabel("C (larger = weaker regularization)")
    ax.set_ylabel("CV accuracy (training set)")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", title="ring = best C, band = ± 1 std", title_fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_results(summary: pd.DataFrame, title: str) -> None:
    """Per scenario: CV accuracy (dot ± 1 std) and test accuracy (diamond)."""
    names = list(summary.index)
    fig, ax = plt.subplots(figsize=(8, 0.65 * len(names) + 1.4))
    ypos = np.arange(len(names))[::-1]
    for y_, (name, row), color in zip(ypos, summary.iterrows(), SERIES_COLORS):
        if not np.isnan(row["CV accuracy"]):
            ax.errorbar(row["CV accuracy"], y_ + 0.12, xerr=row["CV std"], fmt="o", ms=7, color=color,
                        elinewidth=2, capsize=4)
        ax.plot(row["test accuracy"], y_ - 0.12, marker="D", ms=8, color=color, mec=INK, mew=0.8)
        ax.annotate(f"test {row['test accuracy']:.3f}", xy=(row["test accuracy"], y_ - 0.12), xytext=(8, -3),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_yticks(ypos, names)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy")
    ax.set_title(title, loc="left")
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=MUTED, ls="", label="CV on training set (± 1 std)"),
                       plt.Line2D([], [], marker="D", color=MUTED, ls="", label="test set")],
              loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Confidence (Platt scaling)
# ---------------------------------------------------------------------------

def fit_sigmoid_slope(
        make_model: Callable[[float], Model], X: np.ndarray, y: np.ndarray, C: float,
        k: int = K_FOLDS, random_state: int | None = SEED, positive=1,
) -> float:
    """Slope `a` of ``P(positive) = 1 / (1 + exp(-a * score))`` for an SVM's decision scores.

    Platt scaling with the offset fixed at 0, so P = 0.5 exactly at the SVM's
    decision boundary and the confidence never disagrees with the predicted
    label. The scores are out-of-fold (each sample scored by a model trained
    without it, same folds as `tune_C`), because scores on training samples
    are over-confident. Fitted by Newton's method on the log-loss with
    Platt's smoothed targets.

    Args:
        make_model: Same builder as for `tune_C`; the model needs `decision_function`.
        X, y: Training data only (never the test set).
        C: The C the final model was trained with.
    """
    X, y = np.asarray(X), np.asarray(y)
    scores = np.empty(len(y))
    for train_indices, test_indices in stratified_group_k_fold(y, k, None, random_state):
        with contextlib.redirect_stdout(io.StringIO()):
            fold_model = make_model(C).fit(X[train_indices], y[train_indices])
        scores[test_indices] = fold_model.decision_function(X[test_indices])

    is_pos = y == positive
    n_pos, n_neg = is_pos.sum(), (~is_pos).sum()
    target = np.where(is_pos, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    a = 1.0
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-a * scores))
        grad = np.sum((p - target) * scores)
        hess = np.sum(p * (1 - p) * scores ** 2) + 1e-12
        step = grad / hess
        a -= step
        if abs(step) < 1e-10:
            break
    return float(a)
