from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def fetch_daily(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily OHLCV data."""
