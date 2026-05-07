from pydantic import BaseModel, Field


class ModelTrainRequest(BaseModel):
    symbol: str = Field(..., description="A-share symbol, e.g. 000001")
    n_neighbors: int = Field(default=7, ge=1, le=50)


class ModelTrainResponse(BaseModel):
    symbol: str
    model_path: str
    train_rows: int
    valid_rows: int
    metrics: dict[str, float]


class ModelMetricsResponse(BaseModel):
    symbol: str
    metrics: dict[str, float]
