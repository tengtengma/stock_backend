from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    symbol: str
    latest_trade_date: str
    prediction_label: int
    prediction_prob_up: float
    chan_summary: dict
    feature_snapshot: dict[str, float]
    final_signal: str
    signal_reason: str
