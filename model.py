import warnings
import pickle
import numpy as np
from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod

from preprocessing import z_score_standardization, fit_scaler
from config import MODEL_FOLDER_PATH

class NotTrainedError(RuntimeError):
    """Raised when predict() is called before fit()."""
    def __init__(self, message="Model is not trained yet. Call fit() before predict()."):
        super().__init__(message)


class SaveModelError(Exception):
    """Raised when a model cannot be written to disk."""
    def __init__(self, message="Model cannot be saved"):
        super().__init__(message)


class ConvergenceWarning(UserWarning):
    """Warned when SVM training stops at `max_iteration` before converging."""


def _check_fit_input(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """Convert `X`, `y` to arrays and check their shapes match.

    Raises:
        ValueError: If `X` is not 2-D or `X` and `y` differ in length.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"'X' must be 2-D (n_samples, n_features), got shape {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"'X' has {len(X)} samples but 'y' has {len(y)} labels")
    return X, y

class Model(ABC):
    """Base class for all classifiers: `fit` on training data, then `predict`."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        """Learn from features `X` of shape ``(n_samples, n_features)`` and labels `y`."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return one predicted label per row of `X`."""
        pass

def _check_svm_params(C, tolerance, max_iteration) -> None:
    """Validate SVM hyperparameters.

    Raises:
        ValueError: If any value is out of range or unknown.
    """
    if C <= 0:
        raise ValueError(f"'C'({C}) must be greater than 0")
    if tolerance <= 0:
        raise ValueError(f"'tolerance'({tolerance}) must be greater than 0")
    if max_iteration == 0 or max_iteration < -1:
        raise ValueError(f"'max_iteration'({max_iteration}) must be -1 (no limit) or at least 1")

_MAX_PASSES_WITHOUT_CHANGE: int = 5


class BinarySVM(Model):
    """Linear soft-margin SVM trained with Pegasos (primal sub-gradient descent).

    Minimizes ``lam/2 * ||w||^2 + mean(max(0, 1 - y_i * (<w, x_i> + b)))`` with
    ``lam = 1 / (n_samples * C)``, the same trade-off as `C` in a standard
    soft-margin SVM. The bias `b` is not regularized and is learned with the
    same sub-gradient step as `w`. With ``fit_bias=False`` b stays 0 and the
    hyperplane passes through the origin. Features should be standardized
    before fitting either way.

    Training runs in epochs (shuffled passes over the data, one sample per
    step, step size ``1 / (lam * t)``). After each epoch the objective is
    computed; training stops once its relative change stays below `tolerance`
    for `_MAX_PASSES_WITHOUT_CHANGE` epochs in a row, or after `max_iteration`
    epochs (-1 = no limit).

    Labels can be any two distinct values; `classes[0]` is mapped to -1 and
    `classes[1]` to +1 internally, and `predict` returns the original labels.
    """

    def __init__(self,
                 C:float = 1.0,
                 tolerance:float = 1e-3,
                 max_iteration:int=-1,
                 random_state: int | None = None,
                 fit_bias: bool = True):
        _check_svm_params(C, tolerance, max_iteration)
        self.C = C
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.random_state = random_state
        self.fit_bias = fit_bias
        self.classes: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.n_iteration: int = 0  # epochs run by the last fit()

    @staticmethod
    def _objective(w:np.ndarray, b:float, X:np.ndarray, y_signed:np.ndarray, lam:float) -> float:
        """Regularized hinge loss: lam/2 * ||w||^2 + mean hinge loss (b is not regularized)."""
        hinge = np.maximum(0.0, 1.0 - y_signed * (X @ w + b))
        return 0.5 * lam * float(w @ w) + float(hinge.mean())

    def fit(self, X:np.ndarray, y:np.ndarray) -> "BinarySVM":
        """Train on features `X` ``(n_samples, n_features)`` and two-class labels `y`.

        Raises:
            ValueError: If the input shapes are wrong or `y` does not have exactly 2 classes.
        """
        X, y = _check_fit_input(X, y)
        X = X.astype(float)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"BinarySVM needs exactly 2 classes in 'y', got {len(classes)}: {classes}")
        y_signed = np.where(y == classes[1], 1.0, -1.0)

        n_samples, n_features = X.shape
        lam = 1.0 / (n_samples * self.C)
        radius = 1.0 / np.sqrt(lam)  # optimal w lies in the ball ||w|| <= 1/sqrt(lam)
        rng = np.random.default_rng(self.random_state)

        w = np.zeros(n_features)
        b = 0.0
        t = 0
        epoch = 0
        prev_objective = self._objective(w, b, X, y_signed, lam)
        passes_without_change = 0
        converged = False

        while self.max_iteration == -1 or epoch < self.max_iteration:
            epoch += 1
            for i in rng.permutation(n_samples):
                t += 1
                eta = 1.0 / (lam * t)
                margin = y_signed[i] * (X[i] @ w + b)
                w *= 1.0 - eta * lam  # shrink from the regularizer (w only, not b)
                if margin < 1.0:      # hinge active: step towards the sample
                    w += eta * y_signed[i] * X[i]
                    if self.fit_bias:
                        b += eta * y_signed[i]
                norm = np.linalg.norm(w)
                if norm > radius:     # project w back onto the ball
                    w *= radius / norm

            objective = self._objective(w, b, X, y_signed, lam)
            relative_change = abs(prev_objective - objective) / max(abs(prev_objective), 1e-12)
            prev_objective = objective
            passes_without_change = passes_without_change + 1 if relative_change < self.tolerance else 0
            if passes_without_change >= _MAX_PASSES_WITHOUT_CHANGE:
                converged = True
                break

        if not converged:
            warnings.warn(
                f"Pegasos stopped at max_iteration={self.max_iteration} epochs before the objective "
                f"converged (tolerance={self.tolerance}). Increase max_iteration or tolerance.",
                ConvergenceWarning,
            )

        self.classes = classes
        self.weights = w
        self.bias = float(b)
        self.n_iteration = epoch
        return self

    def decision_function(self, X:np.ndarray) -> np.ndarray:
        """Signed distance-like score ``<w, x> + b`` per row; >= 0 means `classes[1]`."""
        if self.weights is None:
            raise NotTrainedError()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError(f"'X' must have shape (n_samples, {len(self.weights)}), got {X.shape}")
        return X @ self.weights + self.bias

    def predict(self, X:np.ndarray) -> np.ndarray:
        """Return `classes[1]` where the score is >= 0, else `classes[0]`."""
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes[1], self.classes[0])


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weight ``n / (n_classes * n_in_class)``: errors on a smaller class cost more.

    Every weight is 1 when the classes are balanced.
    """
    classes, counts = np.unique(y, return_counts=True)
    class_weight = {c: len(y) / (len(classes) * k) for c, k in zip(classes, counts)}
    return np.array([class_weight[v] for v in y])


class BasicSVM(Model):
    """Linear soft-margin SVM trained with full-batch sub-gradient descent.

    Minimizes the same objective as `BinarySVM`,
    ``lam/2 * ||w||^2 + mean(sw_i * max(0, 1 - y_i * (<w, x_i> + b)))`` with
    ``lam = 1 / (n_samples * C)`` and an unregularized bias `b`, where `sw_i`
    are balanced class weights (all 1 for 50/50 data).

    Every epoch makes one update using the gradient over all samples, with
    step size ``learning_rate / sqrt(epoch)``. Training stops once the
    objective's relative change stays below `tolerance` for
    `_MAX_PASSES_WITHOUT_CHANGE` epochs in a row, or after `max_iteration`
    epochs (-1 = no limit). The steps are small and shrink slowly, so this
    needs far more epochs than Pegasos and a much smaller `tolerance`; with
    ``tolerance=1e-3`` it stops long before reaching the minimum.

    Labels can be any two distinct values; `classes[0]` is mapped to -1 and
    `classes[1]` to +1 internally, and `predict` returns the original labels.
    """

    def __init__(self,
                 C:float = 1.0,
                 learning_rate:float = 0.1,
                 tolerance:float = 1e-6,
                 max_iteration:int = 10000):
        _check_svm_params(C, tolerance, max_iteration)
        if learning_rate <= 0:
            raise ValueError(f"'learning_rate'({learning_rate}) must be greater than 0")
        self.C = C
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.classes: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.n_iteration: int = 0  # epochs run by the last fit()

    @staticmethod
    def _objective(w:np.ndarray, b:float, X:np.ndarray, y_signed:np.ndarray,
                   sample_weight:np.ndarray, lam:float) -> float:
        """Weighted regularized hinge loss (b is not regularized)."""
        hinge = np.maximum(0.0, 1.0 - y_signed * (X @ w + b))
        return 0.5 * lam * float(w @ w) + float(np.mean(sample_weight * hinge))

    def fit(self, X:np.ndarray, y:np.ndarray) -> "BasicSVM":
        """Train on features `X` ``(n_samples, n_features)`` and two-class labels `y`.

        Raises:
            ValueError: If the input shapes are wrong or `y` does not have exactly 2 classes.
        """
        X, y = _check_fit_input(X, y)
        X = X.astype(float)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"BasicSVM needs exactly 2 classes in 'y', got {len(classes)}: {classes}")
        y_signed = np.where(y == classes[1], 1.0, -1.0)
        sample_weight = _balanced_weights(y_signed)

        n_samples, n_features = X.shape
        lam = 1.0 / (n_samples * self.C)

        w = np.zeros(n_features)
        b = 0.0
        epoch = 0
        prev_objective = self._objective(w, b, X, y_signed, sample_weight, lam)
        passes_without_change = 0
        converged = False

        while self.max_iteration == -1 or epoch < self.max_iteration:
            epoch += 1
            step = self.learning_rate / np.sqrt(epoch)
            violated = y_signed * (X @ w + b) < 1.0          # samples inside the margin
            coef = sample_weight * y_signed * violated
            grad_w = lam * w - (X.T @ coef) / n_samples
            grad_b = -np.sum(coef) / n_samples
            w -= step * grad_w
            b -= step * grad_b

            objective = self._objective(w, b, X, y_signed, sample_weight, lam)
            relative_change = abs(prev_objective - objective) / max(abs(prev_objective), 1e-12)
            prev_objective = objective
            passes_without_change = passes_without_change + 1 if relative_change < self.tolerance else 0
            if passes_without_change >= _MAX_PASSES_WITHOUT_CHANGE:
                converged = True
                break

        if not converged:
            warnings.warn(
                f"BasicSVM stopped at max_iteration={self.max_iteration} epochs before the objective "
                f"converged (tolerance={self.tolerance}). Increase max_iteration or tolerance.",
                ConvergenceWarning,
            )

        self.classes = classes
        self.weights = w
        self.bias = float(b)
        self.n_iteration = epoch
        return self

    def decision_function(self, X:np.ndarray) -> np.ndarray:
        """Score ``<w, x> + b`` per row; >= 0 means `classes[1]`."""
        if self.weights is None:
            raise NotTrainedError()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError(f"'X' must have shape (n_samples, {len(self.weights)}), got {X.shape}")
        return X @ self.weights + self.bias

    def predict(self, X:np.ndarray) -> np.ndarray:
        """Return `classes[1]` where the score is >= 0, else `classes[0]`."""
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes[1], self.classes[0])


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------

def save_model(model: Model, file_name: str = "model", folder: Path = MODEL_FOLDER_PATH) -> Path:
    """Save a model to ``folder / f"{file_name}.pkl"`` with pickle.

    The folder is created if needed.

    Returns:
        Path of the saved file.

    Raises:
        SaveModelError: If the file cannot be written; the original error
            is attached as its cause.
    """
    path = Path(folder) / f"{file_name}.pkl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    except (OSError, pickle.PicklingError) as e:
        raise SaveModelError(f"Model cannot be saved to '{path}': {e}") from e
    print(f"model saved to {path}")
    return path


def load_model(file_name: str = "model", folder: Path = MODEL_FOLDER_PATH) -> Model:
    """Load a model saved by `save_model`.

    Only load files you created yourself: unpickling can run arbitrary code.
    """
    with open(Path(folder) / f"{file_name}.pkl", 'rb') as file:
        return pickle.load(file)


class Standardized(Model):
    def __init__(self, model:Model):
        self.model = model
        self.mean = self.std = None

    def fit(self, X: np.ndarray, y:np.ndarray):
        self.mean,self.std = fit_scaler(X)
        self.model.fit(z_score_standardization(X, self.mean, self.std), y)
        return self

    def predict(self, X):
        return self.model.predict(z_score_standardization(X, self.mean, self.std))

    def decision_function(self, X):
        """The wrapped model's score on standardized `X` (only for models that have one, e.g. `BinarySVM`)."""
        return self.model.decision_function(z_score_standardization(X, self.mean, self.std))