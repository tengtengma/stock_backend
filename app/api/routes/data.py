import io

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.data import DataSyncRequest, DataSyncResponse
from app.services.analysis_service import AnalysisService

router = APIRouter(prefix="/v1/data", tags=["data"])
service = AnalysisService()


@router.post("/sync", response_model=DataSyncResponse)
def sync_data(payload: DataSyncRequest) -> DataSyncResponse:
    result = service.sync_data(
        payload.symbol,
        payload.start_date,
        payload.end_date,
        data_provider=payload.data_provider,
    )
    return DataSyncResponse(**result)


@router.post("/upload", response_model=DataSyncResponse)
async def upload_csv(
    symbol: str = Form(..., description="A-share symbol, e.g. 000001"),
    file: UploadFile = File(..., description="CSV file"),
) -> DataSyncResponse:
    """
    上传历史K线CSV并保存为 `{symbol}_daily.csv`，供后续训练/分析使用。

    支持 AkShare 常见列名：
    - 日期/开盘/最高/最低/收盘/成交量/成交额
    或已标准化列名：
    - trade_date/open/high/low/close/volume/amount
    """

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty CSV file")

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # Normalize to required columns.
    columns_map = {
        "日期": "trade_date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }

    if "trade_date" not in df.columns and "日期" in df.columns:
        df = df.rename(columns=columns_map)

    required = ["trade_date", "open", "high", "low", "close", "volume", "amount"]
    missing = [c for c in required if c not in df.columns]

    # If amount is missing but volume/close exist, approximate amount.
    if "amount" in missing and "close" in df.columns and "volume" in df.columns:
        df["amount"] = df["close"] * df["volume"]
        missing = [c for c in required if c not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {missing}. Supported columns: {list(columns_map.keys())} or {required}.",
        )

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=required).sort_values("trade_date").reset_index(drop=True)

    # Save to cache.
    try:
        cached_path = service.provider.cache_daily(symbol, df)  # type: ignore[attr-defined]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cache CSV: {e}")

    return DataSyncResponse(symbol=symbol, rows=int(len(df)), file_path=str(cached_path))
