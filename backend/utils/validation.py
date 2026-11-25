"""Checks an uploaded song clip and decodes it into a stereo signal."""
from pathlib import Path

import librosa as lb
import numpy as np
import soundfile as sf
from fastapi import UploadFile

from backend.utils.errors import ErrorCode, api_error

# Decoding rate: ReccoBeats recommends 44.1 kHz; the MFCCs are computed after
# resampling to config.MFCC_SAMPLE_RATE.
SAMPLE_RATE = 44_100

# Keep in sync with LIMITS in frontend/src/constants/errors.ts.
ALLOWED_EXTENSIONS = [".mp3", ".wav"]
MAX_SIZE = 15 * 1024 * 1024                 # bytes: 30 s of 48 kHz stereo 32-bit float WAV is ~11.5 MB
MIN_UPLOAD_DURATION = 10                    # seconds: shorter clips give unreliable MFCC averages
MAX_UPLOAD_DURATION = 30                    # seconds: ReccoBeats analyses at most 30 s
# MP3 decoders add a few ms of padding, so a clip cut at exactly 30 s can decode
# slightly longer. Anything past MAX_UPLOAD_DURATION is cut off before analysis.
DURATION_TOLERANCE = 0.5                    # seconds
SILENCE_THRESHOLD = 1e-3                    # peak amplitude


def check_format(file: UploadFile) -> bool:
    ext = Path(file.filename or "").suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def check_size(file: UploadFile) -> bool:
    file.file.seek(0, 2)          # jump to the end
    size = file.file.tell()       # position = size in bytes
    file.file.seek(0)             # rewind
    return size <= MAX_SIZE


def is_silent(data: np.ndarray) -> bool:
    if data.size == 0:
        return True
    peak = np.max(np.abs(data - data.mean()))
    return peak < SILENCE_THRESHOLD


def validate_file(file: UploadFile) -> np.ndarray:
    """Check an upload and decode it.

    Returns:
        float32 signal of shape ``(2, n_samples)`` at ``SAMPLE_RATE``, at most
        ``MAX_UPLOAD_DURATION`` long. Mono files are copied to both channels
        (ReccoBeats only analyses stereo audio).

    Raises:
        HTTPException: with ``detail={"code", "message"}`` (see `errors`):
            415 UNSUPPORTED_FORMAT, 413 FILE_TOO_LARGE, and 422 for
            CORRUPT_FILE, DURATION_TOO_SHORT, DURATION_TOO_LONG, SILENT_AUDIO.
    """
    if not check_format(file):
        raise api_error(415, ErrorCode.UNSUPPORTED_FORMAT,
                        f"Only {', '.join(ALLOWED_EXTENSIONS)} files are accepted")

    if not check_size(file):
        raise api_error(413, ErrorCode.FILE_TOO_LARGE, f"File larger than {MAX_SIZE // (1024 * 1024)} MB")

    try:
        sf.info(file.file)                    # header only: is it audio at all?
    except RuntimeError:
        raise api_error(422, ErrorCode.CORRUPT_FILE, "Not a valid audio file")

    # librosa decodes and resamples to SAMPLE_RATE, keeping the channels
    file.file.seek(0)
    try:
        data, _ = lb.load(file.file, sr=SAMPLE_RATE, mono=False)
    except Exception:
        raise api_error(422, ErrorCode.CORRUPT_FILE, "Could not decode audio")
    if data.ndim == 1:
        data = np.stack([data, data])
    elif data.shape[0] != 2:                  # e.g. 5.1 audio: mix down, then copy
        mono = data.mean(axis=0)
        data = np.stack([mono, mono])

    # Measured on the decoded signal: a header's duration can be wrong for
    # MP3 files without a VBR header
    duration = data.shape[1] / SAMPLE_RATE
    if duration < MIN_UPLOAD_DURATION:
        raise api_error(422, ErrorCode.DURATION_TOO_SHORT,
                        f"Duration must be at least {MIN_UPLOAD_DURATION}s, got {duration:.2f}s")
    if duration > MAX_UPLOAD_DURATION + DURATION_TOLERANCE:
        raise api_error(422, ErrorCode.DURATION_TOO_LONG,
                        f"Duration must be at most {MAX_UPLOAD_DURATION}s, got {duration:.2f}s")

    data = data[:, :MAX_UPLOAD_DURATION * SAMPLE_RATE]
    if is_silent(data.mean(axis=0)):
        raise api_error(422, ErrorCode.SILENT_AUDIO, "Audio is silent")

    return data.astype(np.float32)
