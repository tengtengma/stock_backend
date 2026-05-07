from pathlib import Path

import akshare as ak
import pandas as pd

from app.core.config import settings
from app.data.providers.base import MarketDataProvider


class AkshareProvider(MarketDataProvider):
    columns_map = {
        "日期": "trade_date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.rename(columns=self.columns_map)[list(self.columns_map.values())].copy()
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.strftime("%Y-%m-%d")
        normalized = normalized.dropna().sort_values("trade_date").reset_index(drop=True)
        return normalized

    def fetch_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if raw.empty:
            raise ValueError(f"No data returned for symbol={symbol}")
        return self._normalize(raw)

    def cache_daily(self, symbol: str, data: pd.DataFrame) -> Path:
        output = settings.data_dir / f"{symbol}_daily.csv"
        data.to_csv(output, index=False)
        return output

    def load_cached_daily(self, symbol: str) -> pd.DataFrame:
        file_path = settings.data_dir / f"{symbol}_daily.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Cached data not found: {file_path}")
        return pd.read_csv(file_path)
