import numpy as np
import pandas as pd
from typing import Dict


def _pct_change(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return np.nan
    past = float(series.iloc[-periods - 1])
    now = float(series.iloc[-1])
    if past == 0:
        return np.nan
    return (now - past) / past


def compute_market_signals(prices_df: pd.DataFrame, lookback_months: int = 36) -> Dict:
    """Compute momentum, percentile rank, and MA cross signals for the uploaded prices.

    Returns a dict with 'metrics' and a small 'series' DataFrame for plotting.
    """
    df = prices_df.sort_values("date").reset_index(drop=True)
    s = df["price"].astype(float)
    d = pd.to_datetime(df["date"]) if "date" in df.columns else pd.RangeIndex(len(df))

    # Moving averages
    ma6 = s.rolling(6, min_periods=1).mean()
    ma12 = s.rolling(12, min_periods=1).mean()
    ma_cross = bool(s.iloc[-1] > ma6.iloc[-1] > ma12.iloc[-1])

    # Momentum
    mom_1m = _pct_change(s, 1)
    mom_3m = _pct_change(s, 3)
    mom_6m = _pct_change(s, 6)

    # Percentile vs lookback
    window = s.tail(lookback_months)
    if len(window) == 0:
        pctl = np.nan
    else:
        rank = (window <= window.iloc[-1]).mean()
        pctl = float(rank)

    sell_zone = bool((pctl >= 0.85) and (mom_3m is not np.nan and mom_3m > 0) and ma_cross)
    caution_zone = bool(pctl <= 0.15)

    out_df = pd.DataFrame({
        "date": d,
        "price": s,
        "ma6": ma6,
        "ma12": ma12,
    })

    return {
        "metrics": {
            "percentile_36m": pctl,
            "momentum_1m": float(mom_1m) if mom_1m == mom_1m else None,
            "momentum_3m": float(mom_3m) if mom_3m == mom_3m else None,
            "momentum_6m": float(mom_6m) if mom_6m == mom_6m else None,
            "ma_cross": ma_cross,
            "sell_zone": sell_zone,
            "caution_zone": caution_zone,
        },
        "series": out_df,
    }

