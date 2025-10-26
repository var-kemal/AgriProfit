import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf as sm_acf, pacf as sm_pacf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from features.engineering import add_calendar_features, add_lag_features


def _series(prices: pd.DataFrame) -> pd.Series:
    s = prices.set_index("date")["price"].sort_index()
    s = s.asfreq("MS") if s.index.inferred_freq is None else s
    return s.astype(float)


def price_overview(prices: pd.DataFrame) -> Dict:
    s = _series(prices)
    data = {
        "count": int(s.count()),
        "start": s.index.min(),
        "end": s.index.max(),
        "min": float(s.min()) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
        "mean": float(s.mean()) if len(s) else None,
        "median": float(s.median()) if len(s) else None,
        "std": float(s.std()) if len(s) else None,
        "missing": int(s.isna().sum()),
    }
    last_12 = s.last("365D")
    data.update(
        {
            "last_value": float(s.dropna().iloc[-1]) if len(s.dropna()) else None,
            "mean_12m": float(last_12.mean()) if len(last_12) else None,
            "vol_12m": float(last_12.std()) if len(last_12) else None,
        }
    )
    return data


def rolling_stats(prices: pd.DataFrame, windows: Tuple[int, ...] = (3, 6, 12)) -> pd.DataFrame:
    s = _series(prices)
    df = s.to_frame("price").copy()
    for w in windows:
        df[f"ma_{w}"] = s.rolling(w).mean()
        df[f"std_{w}"] = s.rolling(w).std()
    df = df.reset_index().rename(columns={"index": "date"})
    return df


def monthly_profile(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = prices.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    by_month = (
        df.groupby("month")["price"].agg(["mean", "median", "std", "count"]).reset_index()
    )
    grid = df.pivot_table(index="year", columns="month", values="price", aggfunc="mean")
    return {"by_month": by_month, "grid": grid}


def stationarity(prices: pd.DataFrame) -> Dict[str, Optional[float]]:
    s = _series(prices).dropna()
    out: Dict[str, Optional[float]] = {"adf_p": None, "kpss_p": None}
    try:
        out["adf_p"] = float(adfuller(s, autolag="AIC")[1])
    except Exception:
        pass
    try:
        # regression="c" (constant); nlags auto via 'legacy' rule if not specified
        out["kpss_p"] = float(kpss(s, regression="c", nlags="auto")[1])
    except Exception:
        pass
    return out


def decompose(prices: pd.DataFrame, period: int = 12) -> Optional[pd.DataFrame]:
    s = _series(prices).dropna()
    if len(s) < period * 2:
        return None
    try:
        res = seasonal_decompose(s, model="additive", period=period, two_sided=True, extrapolate_trend="freq")
        return (
            pd.DataFrame(
                {
                    "date": res.observed.index,
                    "observed": res.observed.values,
                    "trend": res.trend.values,
                    "seasonal": res.seasonal.values,
                    "resid": res.resid.values,
                }
            )
            .reset_index(drop=True)
        )
    except Exception:
        return None


def anomalies_mad(prices: pd.DataFrame, window: int = 12, thresh: float = 3.5) -> pd.DataFrame:
    s = _series(prices)
    df = s.to_frame("price").copy()
    med = s.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    mad = (np.abs(s - med)).rolling(window, center=True, min_periods=max(3, window // 2)).median()
    robust_z = 0.6745 * (s - med) / (mad + 1e-9)
    df["robust_z"] = robust_z
    df["is_outlier"] = np.abs(robust_z) > thresh
    return df.reset_index().rename(columns={"index": "date"})


def acf_pacf_values(prices: pd.DataFrame, nlags: int = 24) -> Dict[str, List[float]]:
    s = _series(prices).dropna()
    nl = min(nlags, max(1, len(s) - 2))
    try:
        a = sm_acf(s, nlags=nl, fft=True)
        p = sm_pacf(s, nlags=nl, method="yw")
        return {"acf": a.tolist(), "pacf": p.tolist()}
    except Exception:
        return {"acf": [], "pacf": []}


def change_points(prices: pd.DataFrame, window: int = 6, z_thresh: float = 2.5) -> pd.DataFrame:
    s = _series(prices)
    df = s.to_frame("price")
    fwd = s.shift(-window).rolling(window).mean()
    bwd = s.rolling(window).mean()
    diff = (fwd - bwd)
    z = (diff - diff.mean()) / (diff.std() + 1e-9)
    out = pd.DataFrame({"date": df.index, "score": z, "is_cp": (np.abs(z) > z_thresh)})
    return out.reset_index(drop=True)


def feature_table(prices: pd.DataFrame) -> pd.DataFrame:
    x = add_calendar_features(prices)
    x = add_lag_features(x)
    x = x.dropna().reset_index(drop=True)
    # keep numeric features except raw date and price to avoid leakage when clustering states
    keep = [c for c in x.columns if c not in ("date",)]
    return x[keep]


def cluster_states(prices: pd.DataFrame, k: int = 3) -> Optional[pd.DataFrame]:
    x = feature_table(prices)
    if len(x) < max(30, k * 5):
        return None
    feat_cols = [c for c in x.columns if c not in ("price",)]
    X = x[feat_cols].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(Xs)
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(Xs)
    # Align with x by reconstructing dates from original prices offset by dropped rows
    # number of rows dropped due to lags
    offset = len(prices) - len(x)
    dates = prices.sort_values("date").iloc[offset:]["date"].values
    out = pd.DataFrame({
        "date": dates,
        "cluster": labels,
        "pc1": emb[:, 0],
        "pc2": emb[:, 1],
    })
    return out


# -----------------------
# Additional analysis helpers: quality, returns, distribution, drawdown, insight
# -----------------------

def infer_frequency(prices: pd.DataFrame) -> str:
    idx = pd.to_datetime(prices["date"]).sort_values()
    try:
        f = pd.infer_freq(idx)
    except Exception:
        f = None
    return f or "MS"


def data_quality(prices: pd.DataFrame) -> Dict:
    df = prices.copy().sort_values("date").reset_index(drop=True)
    idx = pd.to_datetime(df["date"]) if len(df) else pd.to_datetime(pd.Series([], dtype="datetime64[ns]"))
    freq = infer_frequency(df)
    # Build expected date index (assume monthly if unknown)
    if len(idx):
        start, end = idx.min(), idx.max()
        try:
            full = pd.date_range(start, end, freq=freq if freq else "MS")
        except Exception:
            full = pd.date_range(start, end, freq="MS")
    else:
        full = pd.DatetimeIndex([])
    missing = full.difference(idx)
    duplicates = int(idx.duplicated().sum())
    gaps = pd.DataFrame({"date": missing}) if len(missing) else pd.DataFrame(columns=["date"]) 
    return {
        "frequency": freq,
        "is_monotonic": bool(idx.is_monotonic_increasing) if len(idx) else True,
        "n_rows": int(len(df)),
        "n_unique_dates": int(idx.nunique()) if len(idx) else 0,
        "n_duplicates": duplicates,
        "n_missing_periods": int(len(missing)),
        "gaps": gaps,
    }


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    s = _series(prices)
    df = s.to_frame("price").copy()
    df["ret_pct"] = df["price"].pct_change()
    df["ret_log"] = np.log(df["price"]).diff()
    return df.reset_index().rename(columns={"index": "date"})


def distribution_stats(prices: pd.DataFrame) -> Dict:
    s = _series(prices).dropna()
    q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
    return {
        "min": float(s.min()) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
        "mean": float(s.mean()) if len(s) else None,
        "median": float(s.median()) if len(s) else None,
        "std": float(s.std()) if len(s) else None,
        "skew": float(s.skew()) if len(s) else None,
        "kurt": float(s.kurt()) if len(s) else None,
        "quantiles": {str(k): float(v) for k, v in q.items()},
    }


def seasonal_strength(prices: pd.DataFrame, period: int = 12) -> Optional[float]:
    dec = decompose(prices, period=period)
    if dec is None or dec[["seasonal", "resid"]].isna().any().any():
        return None
    s = dec["seasonal"].values
    r = dec["resid"].values
    var_r = float(np.nanvar(r))
    var_s_r = float(np.nanvar(r + s))
    if var_s_r <= 0:
        return None
    strength = max(0.0, min(1.0, 1.0 - (var_r / var_s_r)))
    return strength


def drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    s = _series(prices).dropna()
    if not len(s):
        return pd.DataFrame(columns=["date", "price", "cummax", "drawdown"])
    cummax = s.cummax()
    dd = (s - cummax) / (cummax + 1e-12)
    out = pd.DataFrame({"date": s.index, "price": s.values, "cummax": cummax.values, "drawdown": dd.values})
    return out.reset_index(drop=True)


def insights_report(prices: pd.DataFrame) -> Dict:
    # Summarize key diagnostics into actionable bullets
    qual = data_quality(prices)
    stat = stationarity(prices)
    dist = distribution_stats(prices)
    an = anomalies_mad(prices)
    outlier_rate = float(an["is_outlier"].mean()) if len(an) else 0.0
    ss = seasonal_strength(prices)
    rs = rolling_stats(prices)
    vol_long = float(rs["price"].std()) if len(rs) else None
    vol_recent = float(rs.set_index("date")["price"].last("365D").std()) if len(rs) else None
    bullets: List[str] = []
    # Data quality
    if qual["n_missing_periods"] > 0:
        bullets.append(f"Missing periods detected: {qual['n_missing_periods']}. Consider imputing or restricting analysis window.")
    if qual["n_duplicates"] > 0:
        bullets.append(f"Duplicate dates detected: {qual['n_duplicates']}. Deduplicate or aggregate.")
    # Stationarity
    adf_p = stat.get("adf_p")
    kpss_p = stat.get("kpss_p")
    if adf_p is not None and kpss_p is not None:
        if adf_p > 0.1 and kpss_p < 0.05:
            bullets.append("Series likely non-stationary (ADF>0.1, KPSS<0.05). Consider log/differencing.")
    # Seasonality
    if ss is not None and ss >= 0.3:
        bullets.append(f"Strong seasonality (strength≈{ss:.2f}). Use seasonal adjustments when comparing periods.")
    # Outliers
    if outlier_rate > 0.05:
        bullets.append(f"Outlier rate is elevated ({outlier_rate*100:.1f}%). Use robust stats / winsorization.")
    # Volatility regime
    if vol_long and vol_recent and vol_long > 0:
        ratio = vol_recent / vol_long if vol_long else None
        if ratio and ratio > 1.3:
            bullets.append("Recent 12m volatility is much higher than long-term. Beware unstable patterns.")
        elif ratio and ratio < 0.7:
            bullets.append("Recent 12m volatility is much lower than long-term. Patterns may be calmer recently.")
    # Distribution shape
    if dist.get("skew") and abs(dist["skew"]) > 1.0:
        bullets.append("Distribution is highly skewed. Consider log scale and robust measures.")
    if dist.get("kurt") and dist["kurt"] > 3:
        bullets.append("Heavy tails detected (kurtosis>3). Risk metrics should consider tail events.")
    return {
        "quality": qual,
        "stationarity": stat,
        "seasonality_strength": ss,
        "outlier_rate": outlier_rate,
        "distribution": dist,
        "volatility": {"recent_12m_std": vol_recent, "long_std": vol_long},
        "bullets": bullets,
    }


# -----------------------
# Utilities to print the whole data clearly
# -----------------------

def full_data_report(df: pd.DataFrame, include_summary: bool = True, include_dtypes: bool = True, max_rows: Optional[int] = None) -> str:
    """Return a human-readable text of the entire DataFrame (or head/tail if huge).

    - include_summary: add basic stats and shape
    - include_dtypes: list column dtypes
    - max_rows: if None, attempt to include all rows; if an integer and len(df) > max_rows,
      show head and tail around a truncation notice to avoid massive console spam.
    """
    parts: List[str] = []
    n, m = df.shape
    if include_summary:
        parts.append(f"Shape: {n} rows × {m} columns")
    if include_dtypes:
        dtypes_str = (df.dtypes.astype(str)).to_string()
        parts.append("\nDtypes:\n" + dtypes_str)
    if include_summary:
        try:
            desc = df.describe(include="all", datetime_is_numeric=True).transpose().to_string()
            parts.append("\nSummary describe():\n" + desc)
        except Exception:
            pass

    # Determine how much to print
    if (max_rows is None) or (n <= (max_rows or n)):
        # Print everything
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 0,
            "display.max_colwidth", None,
        ):
            parts.append("\nFull data:\n" + df.to_string(index=False))
    else:
        k = max_rows // 2
        head_str = df.head(k).to_string(index=False)
        tail_str = df.tail(k).to_string(index=False)
        parts.append(f"\nData (truncated to head {k} and tail {k} of {n} rows):\n" + head_str)
        parts.append("\n... (omitting middle rows) ...\n")
        parts.append(tail_str)

    return "\n\n".join(parts)


def print_full_data(df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    """Print the entire DataFrame to stdout (or truncated head/tail if very large).

    Example:
        from analysis.eda import print_full_data
        print_full_data(prices_df)               # print all rows/cols
        print_full_data(prices_df, max_rows=200) # show head 100 + tail 100
    """
    text = full_data_report(df, include_summary=True, include_dtypes=True, max_rows=max_rows)
    print(text)


