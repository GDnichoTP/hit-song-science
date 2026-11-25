"""MFCC features of one audio clip, computed the same way as the training data."""
import librosa as lb
import numpy as np

from config import INT16_SCALE, MFCC_SAMPLE_RATE, N_MFCC


def mfcc_stats(signal: np.ndarray, sr: int = MFCC_SAMPLE_RATE) -> dict[str, float]:
    """Mean and std over time of each MFCC: ``{"mfcc_0_mean": ..., "mfcc_0_std": ..., ...}``.

    Args:
        signal: Mono float signal in [-1, 1].
        sr: Sample rate of `signal`; it is resampled to ``MFCC_SAMPLE_RATE`` if needed.
    """
    if sr != MFCC_SAMPLE_RATE:
        signal = lb.resample(signal, orig_sr=sr, target_sr=MFCC_SAMPLE_RATE)
    mfcc = lb.feature.mfcc(y=signal.astype(np.float32) * INT16_SCALE, sr=MFCC_SAMPLE_RATE, n_mfcc=N_MFCC)
    stats = {}
    for i, row in enumerate(mfcc):
        stats[f"mfcc_{i}_mean"] = float(row.mean())
        stats[f"mfcc_{i}_std"] = float(row.std())
    return stats
