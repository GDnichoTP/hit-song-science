from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent
MODEL_FOLDER_PATH: Path = PROJECT_ROOT / "models"

# Mean / std of the training features used for z-score standardization.
SCALER_PATH: Path = PROJECT_ROOT / "scaler_params.json"

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

DATA_DIR: Path = PROJECT_ROOT / "data" / "complete_only"
WEEKS: list[int] = [1, 2, 3, 4, 8]          # observation windows: dataset_n{week}_{subset}.csv

TARGET: str = "is_trending"
LABEL_NAMES: dict[int, str] = {1: "Popular", -1: "Not popular"}

# duration_ms is left out on purpose (not a musical feature)
SPOTIFY_FEATURES: list[str] = ["explicit", "danceability", "energy", "loudness", "speechiness",
                               "acousticness", "instrumentalness", "liveness", "valence", "tempo"]
MFCC_FEATURES: list[str] = [f"mfcc_{i}_{stat}" for i in range(13) for stat in ("mean", "std")]
FEATURE_SETS: dict[str, list[str]] = {
    "Spotify": SPOTIFY_FEATURES,
    "MFCC": MFCC_FEATURES,
    "Spotify + MFCC": SPOTIFY_FEATURES + MFCC_FEATURES,
}

C_GRID: list[float] = [1e-1, 1, 10, 100]
K_FOLDS: int = 5
TEST_SIZE: float = 0.2
SEED: int = 42
# ---------------------------------------------------------------------------
# Web app (backend/)
# ---------------------------------------------------------------------------
# Saved model the backend serves: models/<APP_MODEL>.pkl plus models/<APP_MODEL>.json
APP_MODEL: str = "svm_n8_spotify_mfcc"

# MFCC extraction, matching the training data: librosa defaults (sample rate
# 22,050 Hz, n_fft 2048, hop 512, 128 mel bands) with 13 coefficients, on
# audio in 16-bit integer scale (+-32768). Only mfcc_0 depends on that scale:
# the dataset's mfcc_0_mean is about +880, float audio in [-1, 1] would give
# roughly -150.
MFCC_SAMPLE_RATE: int = 22_050
N_MFCC: int = 13
INT16_SCALE: float = 32_768.0
