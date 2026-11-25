"""Holds the trained model in memory and turns a song clip into a prediction.

The model is loaded once, when the server starts (see ``server.lifespan``),
and the same instance answers every request. Prediction only reads the
model, so concurrent requests can share it safely.

Every request builds the same 36 features the model was trained on, in the
order saved in ``models/<APP_MODEL>.json``:

    explicit            from the form (cannot be heard in the audio)
    9 audio features    ReccoBeats, from the stereo clip
    26 MFCC stats       feature_extraction.mfcc_stats, from the mono mix

then ``model.decision_function`` (Standardized scales inside) gives the SVM
score; its sign is the label and a sigmoid of it the confidence.
"""
import json
import math
import time

import numpy as np

from backend.services.reccobeats import fetch_audio_features
from config import MODEL_FOLDER_PATH
from feature_extraction import mfcc_stats
from model import Model, load_model

# SVM classes (the training labels) -> ids the frontend knows (constants/labels.ts)
LABEL_IDS = {1: "popular", -1: "not_popular"}


class PredictorService:
    """The trained model plus what is needed to build its input.

    Args:
        model: A fitted `Standardized(BinarySVM)`.
        name: File name the model was loaded from (shown in /api/health).
        features: Feature names in the order the model expects them.
        platt_a: Slope of the sigmoid that turns an SVM score into P(popular).
    """

    def __init__(self, model: Model, name: str, features: list[str], platt_a: float):
        self.model = model
        self.name = name
        self.features = features
        self.platt_a = platt_a

    @classmethod
    def from_file(cls, name: str) -> "PredictorService":
        """Load ``models/<name>.pkl`` and its ``models/<name>.json`` (features, calibration)."""
        with open(MODEL_FOLDER_PATH / f"{name}.json", encoding="utf-8") as f:
            meta = json.load(f)
        if "platt_a" not in meta:
            raise RuntimeError(f"models/{name}.json has no 'platt_a': rerun the save cell in experiment.ipynb")
        return cls(load_model(name), name, meta["features"], float(meta["platt_a"]))

    def build_features(self, stereo: np.ndarray, sr: int, explicit: bool) -> np.ndarray:
        """One row of model input from a ``(2, n_samples)`` clip.

        Raises:
            FeatureServiceError: If ReccoBeats gives no features.
            ValueError: If a feature is missing or not finite.
        """
        values = {"explicit": float(explicit)}
        values.update(fetch_audio_features(stereo, sr))
        values.update(mfcc_stats(stereo.mean(axis=0), sr))
        missing = [f for f in self.features if f not in values]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        row = np.array([[values[f] for f in self.features]], dtype=float)
        if not np.isfinite(row).all():
            raise ValueError("Could not compute features for this audio")
        return row

    def predict(self, stereo: np.ndarray, sr: int, explicit: bool) -> dict:
        """Predict popular / not popular for one clip, with a confidence in [0.5, 1]."""
        start = time.perf_counter()
        row = self.build_features(stereo, sr, explicit)
        score = float(self.model.decision_function(row)[0])
        p_popular = 1.0 / (1.0 + math.exp(-self.platt_a * score))
        label = LABEL_IDS[1] if score >= 0 else LABEL_IDS[-1]
        return {
            "label": label,
            "confidence": round(p_popular if score >= 0 else 1.0 - p_popular, 4),
            "analysed_seconds": round(stereo.shape[1] / sr, 2),
            "processing_ms": round((time.perf_counter() - start) * 1000, 1),
        }
