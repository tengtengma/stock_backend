from pydantic import BaseModel, Field


class DataSyncRequest(BaseModel):
    symbol: str = Field(..., description="A-share symbol, e.g. 000001")
    start_date: str | None = Field(default=None, description="YYYYMMDD")
    end_date: str | None = Field(default=None, description="YYYYMMDD")


class DataSyncResponse(BaseModel):
    symbol: str
    rows: int
    file_path: str
