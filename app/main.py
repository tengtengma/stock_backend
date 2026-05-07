from fastapi import FastAPI

from app.api.routes.analyze import router as analyze_router
from app.api.routes.data import router as data_router
from app.api.routes.model import router as model_router
from app.core.config import settings
from app.schemas.common import HealthResponse

app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)
app.include_router(data_router)
app.include_router(model_router)
app.include_router(analyze_router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version)
