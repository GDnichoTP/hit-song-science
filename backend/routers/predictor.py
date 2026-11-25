"""HTTP endpoints for the popular song predictor."""
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from backend.schemas.predictor import ErrorResponse, Health, Prediction
from backend.services.predictor import PredictorService
from backend.services.reccobeats import FeatureServiceError
from backend.utils.errors import ErrorCode, api_error
from backend.utils.validation import MAX_UPLOAD_DURATION, SAMPLE_RATE, validate_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["predictor"])


def get_predictor(request: Request) -> PredictorService:
    """The service loaded once at startup (see ``server.lifespan``)."""
    return request.app.state.predictor


@router.get("/health", response_model=Health)
def health(predictor: PredictorService = Depends(get_predictor)):
    return Health(status="ok", model=predictor.name, max_duration=MAX_UPLOAD_DURATION)


# A plain `def` (not `async def`): FastAPI runs it in a worker thread, so the
# decoding, the ReccoBeats call and the MFCCs do not block other requests.
@router.post(
    "/predict",
    response_model=Prediction,
    responses={
        413: {"model": ErrorResponse, "description": "FILE_TOO_LARGE"},
        415: {"model": ErrorResponse, "description": "UNSUPPORTED_FORMAT"},
        422: {"model": ErrorResponse, "description": "CORRUPT_FILE, DURATION_TOO_SHORT, DURATION_TOO_LONG, "
                                                     "SILENT_AUDIO, UNPROCESSABLE_AUDIO, FEATURE_SERVICE_FAILED "
                                                     "or INVALID_REQUEST"},
        500: {"model": ErrorResponse, "description": "INTERNAL_ERROR"},
        502: {"model": ErrorResponse, "description": "FEATURE_SERVICE_FAILED"},
        503: {"model": ErrorResponse, "description": "FEATURE_SERVICE_UNAVAILABLE or FEATURE_SERVICE_BUSY"},
    },
)
def predict(
        file: UploadFile = File(..., description="A .wav or .mp3 clip of a song, 10 to 30 seconds"),
        explicit: bool = Form(False, description="Whether the song has explicit lyrics"),
        predictor: PredictorService = Depends(get_predictor),
):
    try:
        stereo = validate_file(file)
        result = predictor.predict(stereo, SAMPLE_RATE, explicit)
    except HTTPException:
        raise
    except FeatureServiceError as e:
        logger.warning("ReccoBeats failed for %r: %s", file.filename, e)
        raise api_error(e.status_code, e.code, str(e))
    except ValueError as e:
        raise api_error(422, ErrorCode.UNPROCESSABLE_AUDIO, str(e))
    except Exception:
        # Raised as an HTTPException (not left unhandled) so the response still
        # passes through CORSMiddleware; otherwise the browser reports a CORS
        # error and the frontend shows "cannot connect" instead of this error.
        logger.exception("Prediction failed for %r", file.filename)
        raise api_error(500, ErrorCode.INTERNAL_ERROR, "Prediction failed")
    return Prediction(**result)
