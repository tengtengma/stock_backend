from fastapi import APIRouter

from app.schemas.analyze import AnalyzeResponse
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/v1", tags=["analyze"])
service = AnalysisService()


@router.get("/analyze/{symbol}", response_model=AnalyzeResponse)
def analyze(symbol: str) -> AnalyzeResponse:
    result = service.analyze_symbol(symbol)
    return AnalyzeResponse(**result)
