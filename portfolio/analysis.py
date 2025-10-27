import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple


def make_synthetic_portfolio_prices(start: str = "2021-01-01", periods: int = 48, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic monthly prices for multiple commodities."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="MS")
    commodities = ["Wheat", "Corn", "Soybeans"]
    data = []
    base_prices = {"Wheat": 220.0, "Corn": 180.0, "Soybeans": 260.0}
    vol = {"Wheat": 0.08, "Corn": 0.10, "Soybeans": 0.12}
    trend = {"Wheat": 0.15, "Corn": 0.10, "Soybeans": 0.20}  # per year %

    for commodity in commodities:
        price = base_prices[commodity]
        series = []
        for i, date in enumerate(dates):
            seasonal = 1 + 0.05 * np.sin(2 * np.pi * (i % 12) / 12)
            noise = rng.normal(0, vol[commodity])
            price = price * (1 + trend[commodity] / 12 / 100) * seasonal * (1 + noise)
            price = max(price, 30)
            series.append((date, commodity, round(price, 2)))
        data.extend(series)
    return pd.DataFrame(data, columns=["date", "commodity", "price"])


def load_portfolio_prices_csv(uploaded: Optional[bytes]) -> pd.DataFrame:
    if uploaded is None:
        return make_synthetic_portfolio_prices()
    df = pd.read_csv(uploaded)
    required = {"date", "commodity", "price"}
    if not required.issubset(df.columns):
        raise ValueError(f"Portfolio prices CSV must contain columns: {required}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Portfolio prices CSV has invalid dates.")
    df["commodity"] = df["commodity"].astype(str)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    return df.sort_values(["commodity", "date"]).reset_index(drop=True)


def load_portfolio_allocations_csv(
    uploaded: Optional[bytes], commodities: Optional[pd.Index] = None
) -> pd.DataFrame:
    if uploaded is None:
        if commodities is None or len(commodities) == 0:
            commodities = pd.Index(["Wheat", "Corn", "Soybeans"])
        weight = 1.0 / len(commodities)
        return pd.DataFrame({"commodity": list(commodities), "allocation": [weight] * len(commodities)})
    df = pd.read_csv(uploaded)
    required = {"commodity", "allocation"}
    if not required.issubset(df.columns):
        raise ValueError(f"Portfolio allocation CSV must contain columns: {required}")
    df = df.copy()
    df["commodity"] = df["commodity"].astype(str)
    df["allocation"] = pd.to_numeric(df["allocation"], errors="coerce")
    df = df.dropna(subset=["allocation"])
    if df["allocation"].sum() == 0:
        raise ValueError("Allocations sum to zero.")
    df["allocation"] = df["allocation"] / df["allocation"].sum()
    if commodities is not None and len(commodities):
        missing = set(commodities) - set(df["commodity"])
        if missing:
            equal = 1.0 / len(missing)
            missing_df = pd.DataFrame({"commodity": list(missing), "allocation": [equal] * len(missing)})
            df = pd.concat([df[["commodity", "allocation"]], missing_df], ignore_index=True)
            df["allocation"] = df["allocation"] / df["allocation"].sum()
    return df.reset_index(drop=True)


def _pivot_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    table = prices_df.pivot(index="date", columns="commodity", values="price").sort_index()
    table = table.ffill().dropna(axis=0, how="any")
    return table


def compute_portfolio_metrics(
    prices_df: pd.DataFrame, allocations_df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    table = _pivot_prices(prices_df)
    if table.empty:
        raise ValueError("Not enough overlapping data across commodities for portfolio metrics.")

    returns = table.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough price variation to compute returns.")

    ann_factor = 12  # monthly data
    mean_returns = returns.mean() * ann_factor
    vol = returns.std() * np.sqrt(ann_factor)
    latest_price = table.iloc[-1]
    allocations = allocations_df.set_index("commodity")["allocation"]
    allocations = allocations.reindex(table.columns).fillna(0.0)
    allocations = allocations / allocations.sum() if allocations.sum() else allocations

    cov = returns.cov() * ann_factor
    portfolio_return = float(np.dot(allocations.values, mean_returns.values))
    portfolio_vol = float(np.sqrt(allocations.values @ cov.values @ allocations.values))

    inverse_vol = 1 / vol.replace(0, np.nan)
    inverse_vol = inverse_vol / inverse_vol.sum()

    summary = pd.DataFrame(
        {
            "latest_price": latest_price,
            "annual_return": mean_returns,
            "annual_vol": vol,
            "allocation": allocations,
        }
    )

    corr = returns.corr()

    insights = {
        "top_return": mean_returns.idxmax(),
        "lowest_vol": vol.idxmin(),
        "portfolio_return": portfolio_return,
        "portfolio_vol": portfolio_vol,
        "inverse_vol_weights": inverse_vol.to_dict(),
    }

    return {
        "summary": summary.reset_index().rename(columns={"index": "commodity"}),
        "correlation": corr,
        "covariance": cov,
        "returns": returns,
        "insights": insights,
    }


def simulate_reallocation(
    prices_df: pd.DataFrame,
    base_allocations: pd.DataFrame,
    source: str,
    target: str,
    shift_pct: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Shift a percentage from source to target and recompute metrics."""
    allocations = base_allocations.set_index("commodity")["allocation"].copy()
    if source == target:
        raise ValueError("Source and target commodities must differ.")
    if source not in allocations.index or target not in allocations.index:
        raise ValueError("Selected commodities not found in allocations.")
    shift = min(shift_pct / 100.0, allocations[source])
    allocations[source] -= shift
    allocations[target] += shift
    allocations = allocations / allocations.sum()
    metrics = compute_portfolio_metrics(prices_df, allocations.reset_index())
    insights = metrics["insights"]
    return allocations.reset_index(name="allocation"), {
        "portfolio_return": insights["portfolio_return"],
        "portfolio_vol": insights["portfolio_vol"],
    }
