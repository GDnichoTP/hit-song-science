from typing import Literal

from pydantic import BaseModel


class Prediction(BaseModel):
    label: Literal["popular", "not_popular"]
    confidence: float           # probability of `label`, 0.5 to 1
    analysed_seconds: float
    processing_ms: float


class Health(BaseModel):
    status: str
    model: str
    max_duration: float


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail
