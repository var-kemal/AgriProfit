# models/forecast_quantile_gbr.py (исправленный)

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict
from features.engineering import add_calendar_features, add_lag_features


def _make_train_table(prices: pd.DataFrame) -> pd.DataFrame:
    x = add_calendar_features(prices)
    x = add_lag_features(x)
    x = x.dropna().reset_index(drop=True)
    return x


def quantile_gbr_forecast(prices: pd.DataFrame, steps: int = 3) -> Dict:
    x = _make_train_table(prices)
    # Exclude raw month/year to avoid stale values in iterative forecasting
    feat = [c for c in x.columns if c not in ("date", "price", "month", "year")]
    y = x["price"].values
    X = x[feat].values

    # More robust settings for stability on small monthly datasets
    def _gbr(alpha: float) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=500,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )

    q10 = _gbr(0.10).fit(X, y)
    q50 = _gbr(0.50).fit(X, y)
    q90 = _gbr(0.90).fit(X, y)

    # Helper to construct features for the NEXT month using only information available up to now
    def _next_feature_row(hist_df: pd.DataFrame) -> pd.DataFrame:
        last = hist_df.iloc[-1:].copy()
        # Advance time index
        last["t"] = last["t"] + 1
        # Advance calendar month and cyclical encodings
        prev_month = int(last["month"]) if "month" in last.columns else int(prices.iloc[-1]["date"].month)
        next_month = (prev_month % 12) + 1
        if "month" in last.columns:
            last["month"] = next_month
        if "month_sin" in last.columns:
            last["month_sin"] = np.sin(2 * np.pi * next_month / 12.0)
        if "month_cos" in last.columns:
            last["month_cos"] = np.cos(2 * np.pi * next_month / 12.0)
        for m in range(1, 13):
            colm = f"m_{m}"
            if colm in last.columns:
                last[colm] = 1 if m == next_month else 0

        # Compute lagged features from known history (no leakage of current-step prediction)
        hist_prices = hist_df["price"].values
        n = len(hist_prices)
        def _lag(L: int):
            if n - L >= 0:
                return float(hist_prices[-L])
            # fallback to median if not enough history
            return float(np.median(hist_prices))

        for L in (1, 3, 6, 12):
            if f"lag_{L}" in last.columns:
                last[f"lag_{L}"] = _lag(L)

        # Rolling statistics from available history
        if "ma_3" in last.columns:
            last["ma_3"] = float(pd.Series(hist_prices[-3:]).mean()) if n >= 1 else float(np.nan)
        if "ma_6" in last.columns:
            last["ma_6"] = float(pd.Series(hist_prices[-6:]).mean()) if n >= 1 else float(np.nan)
        if "std_6" in last.columns:
            last["std_6"] = float(pd.Series(hist_prices[-6:]).std(ddof=0)) if n >= 2 else 0.0
        if "yoy" in last.columns:
            if n >= 12:
                last["yoy"] = float(hist_prices[-1] / (hist_prices[-12] + 1e-9) - 1.0)
            else:
                last["yoy"] = 0.0
        return last

    # Iteratively predict future steps using only past observed/predicted values
    hist = x.copy()
    p10_list, p50_list, p90_list = [], [], []
    for _ in range(steps):
        feat_row = _next_feature_row(hist)
        X_next = feat_row[feat].values
        y10 = float(q10.predict(X_next)[0])
        y50 = float(q50.predict(X_next)[0])
        y90 = float(q90.predict(X_next)[0])
        # Enforce non-crossing quantiles
        lo, med, hi = sorted([y10, y50, y90])
        p10_list.append(lo)
        p50_list.append(med)
        p90_list.append(hi)

        # Append the median prediction as the next observed price for subsequent-step features
        new_row = feat_row.copy()
        new_row["price"] = med
        hist = pd.concat([hist, new_row], ignore_index=True)

    start_date = prices["date"].max() + pd.offsets.MonthBegin(1)
    dates = pd.date_range(start_date, periods=steps, freq="MS")
    out = pd.DataFrame({
        "date": dates,
        "yhat": np.array(p50_list),
        "yhat_lo": np.array(p10_list),
        "yhat_hi": np.array(p90_list),
    })
    return {"models": {"q10": q10, "q50": q50, "q90": q90}, "forecast": out, "feature_names": feat}
