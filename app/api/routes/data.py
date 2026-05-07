from fastapi import APIRouter

from app.schemas.data import DataSyncRequest, DataSyncResponse
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/v1/data", tags=["data"])
service = AnalysisService()


@router.post("/sync", response_model=DataSyncResponse)
def sync_data(payload: DataSyncRequest) -> DataSyncResponse:
    result = service.sync_data(payload.symbol, payload.start_date, payload.end_date)
    return DataSyncResponse(**result)
