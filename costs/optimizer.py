import numpy as np
import pandas as pd
from typing import Optional, Dict, List


def make_synthetic_input_costs(start: str = "2021-01-01", periods: int = 36, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="MS")
    inputs = ["Fertilizer", "Seeds", "Fuel"]
    base_prices = {"Fertilizer": 480.0, "Seeds": 260.0, "Fuel": 3.2}
    volatility = {"Fertilizer": 0.06, "Seeds": 0.04, "Fuel": 0.10}
    data = []
    for inp in inputs:
        price = base_prices[inp]
        for i, dt in enumerate(dates):
            seasonal = 1 + 0.04 * np.sin(2 * np.pi * (i % 12) / 12)
            noise = rng.normal(0, volatility[inp])
            trend = 1 + (0.02 if inp == "Fertilizer" else 0.015) / 12
            price = max(price * seasonal * trend * (1 + noise), 0.1)
            data.append((dt, inp, round(price, 2)))
    return pd.DataFrame(data, columns=["date", "input", "price"])


def load_input_costs_csv(uploaded: Optional[bytes]) -> pd.DataFrame:
    if uploaded is None:
        return make_synthetic_input_costs()
    df = pd.read_csv(uploaded)
    required = {"date", "input", "price"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input cost CSV must contain columns: {required}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Input cost CSV has invalid dates.")
    df["input"] = df["input"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    return df.sort_values(["input", "date"]).reset_index(drop=True)


def _linear_trend(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    y = series.values
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def compute_input_cost_signals(cost_df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    df = cost_df.copy()
    df = df.sort_values(["input", "date"])
    summary_rows = []
    history = {}
    for inp, group in df.groupby("input"):
        group = group.set_index("date").sort_index()
        last_price = float(group["price"].iloc[-1])
        last_date = group.index[-1]
        six_month = float(group["price"].tail(6).mean()) if len(group) >= 6 else float(group["price"].mean())
        twelve_month = group["price"].tail(12)
        pct_rank = float((twelve_month.rank(pct=True).iloc[-1] if len(twelve_month) else group["price"].rank(pct=True).iloc[-1]))
        slope = _linear_trend(group["price"].tail(min(6, len(group))))
        signal = "Hold"
        if pct_rank <= 0.25:
            signal = "Buy now"
        elif pct_rank >= 0.75:
            signal = "Delay purchase"
        history[inp] = group.reset_index().values.tolist()
        summary_rows.append(
            {
                "input": inp,
                "last_price": last_price,
                "last_date": last_date,
                "six_month_avg": six_month,
                "percentile": pct_rank,
                "trend_slope": slope,
                "recommendation": signal,
            }
        )
    return {
        "summary": summary_rows,
        "history": history,
    }
