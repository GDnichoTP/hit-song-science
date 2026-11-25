"""Spotify-style audio features from ReccoBeats (https://reccobeats.com).

Spotify's own audio-features endpoint is closed to apps created after
27 Nov 2024. ReccoBeats estimates the same nine features from an uploaded
clip: free, no API key, but it analyses at most 30 s, accepts at most
5 MB, and its rate limits are not published (429 when exceeded).

Tested Nov 2026: mono uploads (WAV or MP3) fail with 400 "Get audio
features fail"; stereo works. 30 s of stereo 16-bit WAV is 5.3 MB, over
the limit, so clips are sent as stereo MP3 (~0.2-0.5 MB).
"""
import io
import math

import numpy as np
import requests
import soundfile as sf

from backend.utils.errors import ErrorCode

URL = "https://api.reccobeats.com/v1/analysis/audio-features"
TIMEOUT = 60                    # seconds
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# The nine features ReccoBeats returns (all of the model's Spotify features except `explicit`)
FEATURES = ["acousticness", "danceability", "energy", "instrumentalness", "liveness",
            "loudness", "speechiness", "tempo", "valence"]


class FeatureServiceError(Exception):
    """ReccoBeats could not give features for this clip; carries the HTTP status and error code to return."""

    def __init__(self, status_code: int, code: ErrorCode, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def to_mp3_bytes(stereo: np.ndarray, sr: int) -> bytes:
    """Encode a ``(2, n_samples)`` signal as stereo MP3."""
    buffer = io.BytesIO()
    sf.write(buffer, stereo.T, sr, format="MP3", subtype="MPEG_LAYER_III")
    return buffer.getvalue()


def fetch_audio_features(stereo: np.ndarray, sr: int) -> dict[str, float]:
    """Send one ``(2, n_samples)`` clip (at most 30 s) to ReccoBeats and return its nine features.

    Raises:
        FeatureServiceError: If the service is unreachable, rate-limited,
            rejects the clip or answers without valid features.
    """
    audio = to_mp3_bytes(stereo, sr)
    if len(audio) > MAX_UPLOAD_BYTES:
        raise FeatureServiceError(422, ErrorCode.FEATURE_SERVICE_FAILED,
                                  f"Clip is {len(audio)} bytes, ReccoBeats accepts at most {MAX_UPLOAD_BYTES}")

    try:
        response = requests.post(URL, files={"audioFile": ("clip.mp3", audio, "audio/mpeg")}, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise FeatureServiceError(503, ErrorCode.FEATURE_SERVICE_UNAVAILABLE, f"ReccoBeats unreachable: {e}")

    if response.status_code == 429:
        raise FeatureServiceError(503, ErrorCode.FEATURE_SERVICE_BUSY, "ReccoBeats rate limit reached")
    if response.status_code >= 500:
        raise FeatureServiceError(503, ErrorCode.FEATURE_SERVICE_UNAVAILABLE,
                                  f"ReccoBeats error {response.status_code}")
    if response.status_code != 200:
        raise FeatureServiceError(422, ErrorCode.FEATURE_SERVICE_FAILED,
                                  f"ReccoBeats rejected the clip ({response.status_code}): {response.text[:200]}")

    try:
        body = response.json()
        features = {name: float(body[name]) for name in FEATURES}
    except (ValueError, KeyError, TypeError):
        raise FeatureServiceError(502, ErrorCode.FEATURE_SERVICE_FAILED,
                                  f"Unexpected ReccoBeats answer: {response.text[:200]}")
    if not all(math.isfinite(v) for v in features.values()):
        raise FeatureServiceError(502, ErrorCode.FEATURE_SERVICE_FAILED, "ReccoBeats returned non-finite values")
    return features
