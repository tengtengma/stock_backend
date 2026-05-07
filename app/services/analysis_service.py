from app.core.config import settings
from app.data.providers.akshare_provider import AkshareProvider
from app.domain.chan.engine import ChanTheoryEngine
from app.domain.features.builder import FeatureBuilder
from app.ml.knn_service import KNNModelService


class AnalysisService:
    def __init__(self) -> None:
        self.provider = AkshareProvider()
        self.feature_builder = FeatureBuilder()
        self.chan_engine = ChanTheoryEngine()
        self.model_service = KNNModelService()

    @staticmethod
    def _derive_final_signal(prob_up: float, latest_segment_direction: str) -> tuple[str, str]:
        if prob_up >= 0.65 and latest_segment_direction == "up":
            return "strong_long", "KNN bullish probability is high and Chan segment is up"
        if 0.55 <= prob_up < 0.65 and latest_segment_direction != "down":
            return "watch_long", "KNN is mildly bullish and Chan structure is not bearish"
        if prob_up <= 0.35 and latest_segment_direction == "down":
            return "strong_avoid", "KNN bearish probability is high and Chan segment is down"
        return "neutral", "Model probability and Chan structure are not aligned strongly"

    def sync_data(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> dict:
        data = self.provider.fetch_daily(
            symbol=symbol,
            start_date=start_date or settings.default_start_date,
            end_date=end_date or settings.default_end_date,
        )
        output = self.provider.cache_daily(symbol, data)
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
        chan = self.chan_engine.analyze(df)
        latest = dataset.iloc[-1]
        feature_cols = self.feature_builder.feature_columns()
        feature_snapshot = {k: float(latest[k]) for k in feature_cols}
        final_signal, signal_reason = self._derive_final_signal(
            prediction["prob_up"], chan["summary"]["latest_segment_direction"]
        )
        return {
            "symbol": symbol,
            "latest_trade_date": str(df.iloc[-1]["trade_date"]),
            "prediction_label": prediction["label"],
            "prediction_prob_up": prediction["prob_up"],
            "chan_summary": chan["summary"],
            "feature_snapshot": feature_snapshot,
            "final_signal": final_signal,
            "signal_reason": signal_reason,
        }

    def metrics(self, symbol: str) -> dict[str, float]:
        return self.model_service.load_metrics(symbol)
