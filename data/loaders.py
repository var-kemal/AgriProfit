import io
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from .validators import validate_prices, validate_costs, validate_yields

def load_price_csv(uploaded: Optional[io.BytesIO]) -> pd.DataFrame:
    if uploaded is None:
        return make_synthetic_prices()
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        raise ValueError(f"Не удалось прочитать CSV: {e}")

    required_cols = {"date", "price"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV должен содержать колонки: {required_cols}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("В колонке 'date' есть некорректные значения")
    return validate_prices(df)


def load_costs_csv(uploaded: Optional[io.BytesIO]) -> pd.DataFrame:
    if uploaded is None:
        _, costs = make_synthetic_yield_costs()
        return validate_costs(costs)
    df = pd.read_csv(uploaded)
    if not set(["item", "amount"]).issubset(df.columns):
        raise ValueError("Costs CSV must contain columns: item, amount")
    df = validate_costs(df)
    return df


def load_yields_csv(uploaded: Optional[io.BytesIO]) -> pd.DataFrame:
    if uploaded is None:
        yields, _ = make_synthetic_yield_costs()
        return validate_yields(yields)
    df = pd.read_csv(uploaded)
    if not set(["year", "yield_per_ha"]).issubset(df.columns):
        raise ValueError("Yields CSV must contain columns: year, yield_per_ha")
    df = validate_yields(df)
    return df


def make_synthetic_prices(start="2021-01-01", periods=48, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="MS")
    t = np.arange(periods)
    seasonal = 0.12 * np.sin(2 * np.pi * t / 12.0)
    trend = 0.006 * t
    noise = rng.normal(0, 0.03, size=periods)
    idx = 1.5 + seasonal + trend + noise
    base, scale = 16, 4
    price = base + scale * (idx - idx.mean()) / idx.std()
    return pd.DataFrame({"date": dates, "price": price.round(2)})


def make_synthetic_yield_costs(seed=7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    years = np.arange(2021, 2026)
    yield_per_ha = np.maximum(3.0, rng.normal(5.0, 0.5, size=len(years)))
    yields = pd.DataFrame({"year": years, "yield_per_ha": yield_per_ha.round(2)})
    costs = pd.DataFrame([
        ("Seeds", 1200), ("Fertilizer", 3500), ("Fuel", 2200), ("Labor", 4000), ("Irrigation", 1500)
    ], columns=["item", "amount"])
    return yields, costs