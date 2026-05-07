from fastapi import APIRouter

from app.schemas.model import ModelMetricsResponse, ModelTrainRequest, ModelTrainResponse
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/v1/model", tags=["model"])
service = AnalysisService()


@router.post("/train", response_model=ModelTrainResponse)
def train_model(payload: ModelTrainRequest) -> ModelTrainResponse:
    result = service.train_model(payload.symbol, payload.n_neighbors)
    return ModelTrainResponse(
        symbol=payload.symbol,
        model_path=result["model_path"],
        train_rows=result["train_rows"],
        valid_rows=result["valid_rows"],
        metrics=result["metrics"],
    )


@router.get("/metrics", response_model=ModelMetricsResponse)
def get_metrics(symbol: str) -> ModelMetricsResponse:
    metrics = service.metrics(symbol)
    return ModelMetricsResponse(symbol=symbol, metrics=metrics)
