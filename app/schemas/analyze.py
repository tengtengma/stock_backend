from pydantic import BaseModel


class KlineRangeItem(BaseModel):
    low: float
    high: float
    downside_return: float
    upside_return: float


class KlineRangeForecast(BaseModel):
    next_3d: KlineRangeItem
    next_5d: KlineRangeItem
    next_10d: KlineRangeItem


class AnalyzeResponse(BaseModel):
    symbol: str
    latest_trade_date: str
    prediction_label: int
    prediction_prob_up: float
    chan_summary: dict
    feature_snapshot: dict[str, float]
    final_signal: str
    signal_reason: str
    kline_range_forecast: KlineRangeForecast
