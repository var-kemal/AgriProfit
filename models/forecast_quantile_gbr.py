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

    future_rows = []
    hist = x.copy()
    for s in range(1, steps + 1):
        last = hist.iloc[-1:].copy()
        last["t"] = last["t"] + 1
        # Advance calendar month and update cyclical encodings if present
        prev_month = int(last["month"]) if "month" in last.columns else int(hist.iloc[-1]["date"].month)
        last_month = (prev_month % 12) + 1
        if "month" in last.columns:
            last["month"] = last_month
        if "month_sin" in last.columns:
            last["month_sin"] = np.sin(2 * np.pi * last_month / 12.0)
        if "month_cos" in last.columns:
            last["month_cos"] = np.cos(2 * np.pi * last_month / 12.0)
        for m in range(1, 13):
            colm = f"m_{m}"
            if colm in last.columns:
                last[colm] = 1 if m == last_month else 0
        med_pred = q50.predict(last[feat].values)[0]
        for L in (1, 3, 6, 12):
            if L == 1:
                last[f"lag_{L}"] = med_pred
            else:
                last[f"lag_{L}"] = hist.iloc[-min(L, len(hist)) :]["price"].median()
        # исправленные строки с pd.concat
        last["ma_3"] = np.nanmean([med_pred, hist.iloc[-1]["price"], hist.iloc[-2]["price"] if len(hist) > 2 else med_pred])
        last["ma_6"] = float(pd.concat([hist["price"].tail(6), pd.Series([med_pred])]).rolling(6).mean().iloc[-1]) if len(hist) >= 5 else med_pred
        last["std_6"] = float(pd.concat([hist["price"].tail(6), pd.Series([med_pred])]).rolling(6).std().iloc[-1]) if len(hist) >= 5 else np.std(hist["price"].values)
        if "yoy" in last.columns:
            if len(hist) >= 12:
                last_12_price = hist.iloc[-12]["price"]
                last["yoy"] = (med_pred / last_12_price) - 1.0
            else:
                last["yoy"] = 0.0
        hist = pd.concat([hist, last], ignore_index=True)
        future_rows.append(last[feat].values[0])

    Xf = np.vstack(future_rows)
    p10 = q10.predict(Xf)
    p50 = q50.predict(Xf)
    p90 = q90.predict(Xf)

    start_date = prices["date"].max() + pd.offsets.MonthBegin(1)
    dates = pd.date_range(start_date, periods=steps, freq="MS")
    out = pd.DataFrame({"date": dates, "yhat": p50, "yhat_lo": p10, "yhat_hi": p90})
    return {"models": {"q10": q10, "q50": q50, "q90": q90}, "forecast": out}
