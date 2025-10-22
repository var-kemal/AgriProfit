import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["month"] = x["date"].dt.month
    x["year"] = x["date"].dt.year
    x["t"] = np.arange(len(x))
    # Cyclical month encoding
    x["month_sin"] = np.sin(2 * np.pi * x["month"] / 12.0)
    x["month_cos"] = np.cos(2 * np.pi * x["month"] / 12.0)
    month_dummies = pd.get_dummies(x["month"], prefix="m", dtype=int)
    return pd.concat([x, month_dummies], axis=1)


def add_lag_features(df: pd.DataFrame, lags=(1, 3, 6, 12)) -> pd.DataFrame:
    x = df.copy().sort_values("date")
    for L in lags:
        x[f"lag_{L}"] = x["price"].shift(L)
    # rolling stats
    x["ma_3"] = x["price"].rolling(3).mean()
    x["ma_6"] = x["price"].rolling(6).mean()
    x["std_6"] = x["price"].rolling(6).std()
    # year-over-year change (uses lag_12)
    if "lag_12" in x.columns:
        x["yoy"] = (x["price"] / x["lag_12"]) - 1.0
    return x
