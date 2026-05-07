from pydantic import BaseModel, Field


class DataSyncRequest(BaseModel):
    symbol: str = Field(..., description="A-share symbol, e.g. 000001")
    start_date: str | None = Field(default=None, description="YYYYMMDD")
    end_date: str | None = Field(default=None, description="YYYYMMDD")
    data_provider: str | None = Field(
        default=None,
        description="数据源选择：auto/tushare/akshare。仅影响 sync 拉取数据，不影响训练/分析。"
    )


class DataSyncResponse(BaseModel):
    symbol: str
    rows: int
    file_path: str
