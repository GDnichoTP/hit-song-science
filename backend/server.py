"""FastAPI app for Popular Song Prediction.

Run from the project root (so `config`, `model`, ... and the pickled model
can be imported):

    uvicorn backend.server:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from backend.routers import predictor
from backend.services.predictor import PredictorService
from backend.utils.errors import ErrorCode, error_detail
from config import APP_MODEL

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

ALLOWED_METHODS = [
    "GET",
    "POST",          # /api/predict uploads a file
]

ALLOWED_HEADERS = [
    "*"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load the model into memory ...
    app.state.predictor = PredictorService.from_file(APP_MODEL)
    yield
    # ... and once at shutdown
    app.state.predictor = None


app = FastAPI(title="Popular Song Prediction", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=ALLOWED_METHODS,
    allow_headers=ALLOWED_HEADERS
)

app.include_router(predictor.router)


@app.exception_handler(RequestValidationError)
async def request_validation_error(request: Request, exc: RequestValidationError):
    """FastAPI's own 422 (e.g. no `file` field) in the same shape as the others."""
    fields = ", ".join(".".join(str(p) for p in err["loc"]) for err in exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": error_detail(ErrorCode.INVALID_REQUEST, f"Invalid request: {fields}")},
    )


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")
