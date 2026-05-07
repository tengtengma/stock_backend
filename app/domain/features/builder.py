import numpy as np
import pandas as pd

from app.domain.chan.engine import ChanTheoryEngine


class FeatureBuilder:
    def __init__(self) -> None:
        self.chan_engine = ChanTheoryEngine()

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ret_1"] = out["close"].pct_change(1)
        out["ret_5"] = out["close"].pct_change(5)
        out["ret_10"] = out["close"].pct_change(10)
        out["volatility_10"] = out["close"].pct_change().rolling(10).std()
        out["volatility_20"] = out["close"].pct_change().rolling(20).std()
        return out

    def _add_volume_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["volume_ratio_5"] = out["volume"] / out["volume"].rolling(5).mean()
        out["amount_ratio_5"] = out["amount"] / out["amount"].rolling(5).mean()
        out["price_volume_corr_10"] = out["close"].rolling(10).corr(out["volume"])
        return out

    def _chan_stat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        analysis = self.chan_engine.analyze(df)
        bis = analysis["bis"]
        segments = analysis["segments"]

        bi_slope = float(np.mean([b["slope"] for b in bis[-5:]])) if bis else 0.0
        bi_abs_change = float(np.mean([abs(b["price_change"]) for b in bis[-5:]])) if bis else 0.0
        latest_segment_strength = float(segments[-1]["strength"]) if segments else 0.0
        latest_segment_dir = 1.0 if segments and segments[-1]["direction"] == "up" else 0.0

        out = df.copy()
        out["chan_bi_slope_mean_5"] = bi_slope
        out["chan_bi_abs_change_mean_5"] = bi_abs_change
        out["chan_segment_strength_latest"] = latest_segment_strength
        out["chan_segment_dir_latest"] = latest_segment_dir
        out["chan_bi_count"] = float(len(bis))
        out["chan_segment_count"] = float(len(segments))
        return out

    @staticmethod
    def _future_extreme_return(
        close: pd.Series, prices: pd.Series, horizon: int, use_max: bool
    ) -> pd.Series:
        values = prices.to_list()
        close_values = close.to_list()
        output: list[float] = []

        for idx, close_now in enumerate(close_values):
            future_window = values[idx + 1 : idx + 1 + horizon]
            if len(future_window) < horizon or close_now == 0:
                output.append(np.nan)
                continue
            target_price = max(future_window) if use_max else min(future_window)
            output.append(target_price / close_now - 1)
        return pd.Series(output, index=close.index)

    def build_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._add_return_features(df)
        out = self._add_volume_price_features(out)
        out = self._chan_stat_features(out)
        out["next_1d_return"] = out["close"].shift(-1) / out["close"] - 1
        out["label"] = (out["next_1d_return"] > 0).astype(int)
        for horizon in (3, 5, 10):
            out[f"upside_{horizon}d"] = self._future_extreme_return(
                out["close"], out["high"], horizon, use_max=True
            )
            out[f"downside_{horizon}d"] = self._future_extreme_return(
                out["close"], out["low"], horizon, use_max=False
            )
        out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        return out

    @staticmethod
    def feature_columns() -> list[str]:
        return [
            "ret_1",
            "ret_5",
            "ret_10",
            "volatility_10",
            "volatility_20",
            "volume_ratio_5",
            "amount_ratio_5",
            "price_volume_corr_10",
            "chan_bi_slope_mean_5",
            "chan_bi_abs_change_mean_5",
            "chan_segment_strength_latest",
            "chan_segment_dir_latest",
            "chan_bi_count",
            "chan_segment_count",
        ]
