from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Fractal:
    idx: int
    trade_date: str
    fractal_type: str  # top / bottom
    high: float
    low: float


@dataclass
class Bi:
    start_idx: int
    end_idx: int
    direction: str  # up / down
    bars: int
    price_change: float
    slope: float


@dataclass
class Segment:
    start_idx: int
    end_idx: int
    direction: str
    bi_count: int
    strength: float


class ChanTheoryEngine:
    def detect_fractals(self, df: pd.DataFrame) -> list[Fractal]:
        fractals: list[Fractal] = []
        if len(df) < 3:
            return fractals

        for i in range(1, len(df) - 1):
            prev_row, row, next_row = df.iloc[i - 1], df.iloc[i], df.iloc[i + 1]
            if row["high"] > prev_row["high"] and row["high"] > next_row["high"]:
                fractals.append(
                    Fractal(i, row["trade_date"], "top", float(row["high"]), float(row["low"]))
                )
            elif row["low"] < prev_row["low"] and row["low"] < next_row["low"]:
                fractals.append(
                    Fractal(i, row["trade_date"], "bottom", float(row["high"]), float(row["low"]))
                )
        return fractals

    def build_bis(self, df: pd.DataFrame, fractals: list[Fractal]) -> list[Bi]:
        bis: list[Bi] = []
        if len(fractals) < 2:
            return bis

        filtered = [fractals[0]]
        for f in fractals[1:]:
            if f.fractal_type != filtered[-1].fractal_type:
                filtered.append(f)

        for i in range(1, len(filtered)):
            start, end = filtered[i - 1], filtered[i]
            start_close = float(df.iloc[start.idx]["close"])
            end_close = float(df.iloc[end.idx]["close"])
            delta = end_close - start_close
            bars = end.idx - start.idx
            if bars <= 0:
                continue
            bis.append(
                Bi(
                    start_idx=start.idx,
                    end_idx=end.idx,
                    direction="up" if delta >= 0 else "down",
                    bars=bars,
                    price_change=delta,
                    slope=delta / bars,
                )
            )
        return bis

    def build_segments(self, bis: list[Bi], segment_size: int = 3) -> list[Segment]:
        segments: list[Segment] = []
        if len(bis) < segment_size:
            return segments

        for i in range(segment_size - 1, len(bis)):
            window = bis[i - segment_size + 1 : i + 1]
            up_cnt = sum(1 for b in window if b.direction == "up")
            down_cnt = segment_size - up_cnt
            direction = "up" if up_cnt >= down_cnt else "down"
            strength = sum(abs(b.price_change) for b in window)
            segments.append(
                Segment(
                    start_idx=window[0].start_idx,
                    end_idx=window[-1].end_idx,
                    direction=direction,
                    bi_count=segment_size,
                    strength=float(strength),
                )
            )
        return segments

    def analyze(self, df: pd.DataFrame) -> dict:
        fractals = self.detect_fractals(df)
        bis = self.build_bis(df, fractals)
        segments = self.build_segments(bis)
        return {
            "fractals": [asdict(f) for f in fractals],
            "bis": [asdict(b) for b in bis],
            "segments": [asdict(s) for s in segments],
            "summary": {
                "fractal_count": len(fractals),
                "bi_count": len(bis),
                "segment_count": len(segments),
                "latest_segment_direction": segments[-1].direction if segments else "none",
            },
        }
