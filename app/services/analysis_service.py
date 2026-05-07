from app.core.config import settings
from app.data.providers.akshare_provider import AkshareProvider
from app.domain.chan.engine import ChanTheoryEngine
from app.domain.features.builder import FeatureBuilder
from app.ml.knn_service import KNNModelService


class AnalysisService:
    def __init__(self) -> None:
        self.provider = self._build_provider_by_name(settings.data_provider)
        self.feature_builder = FeatureBuilder()
        self.chan_engine = ChanTheoryEngine()
        self.model_service = KNNModelService()

    @staticmethod
    def _build_provider_by_name(provider_name: str):
        provider_name = (provider_name or "").lower().strip()
        if provider_name == "akshare":
            return AkshareProvider()

        if provider_name in {"tushare", "auto"}:
            # Lazy import to avoid hard crash when `tushare` dependency is missing.
            if settings.tushare_token:
                try:
                    from app.data.providers.tushare_provider import TushareProvider

                    return TushareProvider()
                except ModuleNotFoundError as e:
                    if provider_name == "tushare":
                        raise ValueError(
                            "Tushare provider selected but `tushare` package is not installed. "
                            "Run `pip install tushare` or set STOCK_AI_DATA_PROVIDER=akshare/auto."
                        ) from e
                    return AkshareProvider()
            # No token: allow auto to fall back to akshare.
            return AkshareProvider()

        raise ValueError(f"Unsupported data provider: {settings.data_provider}")

    @staticmethod
    def _derive_final_signal(prob_up: float, latest_segment_direction: str) -> tuple[str, str]:
        if prob_up >= 0.65 and latest_segment_direction == "up":
            return "strong_long", "KNN bullish probability is high and Chan segment is up"
        if 0.55 <= prob_up < 0.65 and latest_segment_direction != "down":
            return "watch_long", "KNN is mildly bullish and Chan structure is not bearish"
        if prob_up <= 0.35 and latest_segment_direction == "down":
            return "strong_avoid", "KNN bearish probability is high and Chan segment is down"
        return "neutral", "Model probability and Chan structure are not aligned strongly"

    @staticmethod
    def _build_kline_range_forecast(close_price: float, range_returns: dict[str, float]) -> dict:
        forecast = {}
        for horizon in (3, 5, 10):
            upside_ret = range_returns[f"upside_{horizon}d"]
            downside_ret = range_returns[f"downside_{horizon}d"]
            forecast[f"next_{horizon}d"] = {
                "low": round(close_price * (1 + downside_ret), 4),
                "high": round(close_price * (1 + upside_ret), 4),
                "downside_return": round(downside_ret, 6),
                "upside_return": round(upside_ret, 6),
            }
        return forecast

    def sync_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        data_provider: str | None = None,
    ) -> dict:
        provider = self.provider if data_provider is None else self._build_provider_by_name(data_provider)

        data = provider.fetch_daily(
            symbol=symbol,
            start_date=start_date or settings.default_start_date,
            end_date=end_date or settings.default_end_date,
        )
        output = provider.cache_daily(symbol, data)
        return {"symbol": symbol, "rows": len(data), "file_path": str(output)}

    def train_model(self, symbol: str, n_neighbors: int) -> dict:
        df = self.provider.load_cached_daily(symbol)
        dataset = self.feature_builder.build_dataset(df)
        return self.model_service.train(
            symbol=symbol,
            dataset=dataset,
            feature_cols=self.feature_builder.feature_columns(),
            n_neighbors=n_neighbors,
        )

    def analyze_symbol(self, symbol: str) -> dict:
        df = self.provider.load_cached_daily(symbol)
        dataset = self.feature_builder.build_dataset(df)
        prediction = self.model_service.predict_latest(symbol, dataset)
        range_returns = self.model_service.predict_latest_ranges(symbol, dataset)
        chan = self.chan_engine.analyze(df)
        latest = dataset.iloc[-1]
        feature_cols = self.feature_builder.feature_columns()
        feature_snapshot = {k: float(latest[k]) for k in feature_cols}
        final_signal, signal_reason = self._derive_final_signal(
            prediction["prob_up"], chan["summary"]["latest_segment_direction"]
        )
        close_price = float(df.iloc[-1]["close"])
        return {
            "symbol": symbol,
            "latest_trade_date": str(df.iloc[-1]["trade_date"]),
            "prediction_label": prediction["label"],
            "prediction_prob_up": prediction["prob_up"],
            "chan_summary": chan["summary"],
            "feature_snapshot": feature_snapshot,
            "final_signal": final_signal,
            "signal_reason": signal_reason,
            "kline_range_forecast": self._build_kline_range_forecast(close_price, range_returns),
        }

    def metrics(self, symbol: str) -> dict[str, float]:
        return self.model_service.load_metrics(symbol)
