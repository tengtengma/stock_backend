from pathlib import Path

import pandas as pd
import tushare as ts

from app.core.config import settings
from app.data.providers.base import MarketDataProvider


class TushareProvider(MarketDataProvider):
    @staticmethod
    def _to_ts_code(symbol: str) -> str:
        if "." in symbol:
            return symbol.upper()
        if symbol.startswith(("6", "9")):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.rename(
            columns={
                "trade_date": "trade_date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "amount": "amount",
            }
        )[["trade_date", "open", "high", "low", "close", "volume", "amount"]].copy()

        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.strftime("%Y-%m-%d")
        normalized = normalized.dropna().sort_values("trade_date").reset_index(drop=True)
        return normalized

    def fetch_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not settings.tushare_token:
            raise ValueError("Tushare token is not configured. Set STOCK_AI_TUSHARE_TOKEN.")

        pro = ts.pro_api(settings.tushare_token)
        ts_code = self._to_ts_code(symbol)
        raw = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if raw.empty:
            raise ValueError(f"No data returned for symbol={symbol} ({ts_code})")
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
