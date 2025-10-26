import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis.eda import change_points as eda_change_points


class UltimateTimeSeriesAnalyzer:
    """Lightweight, dependency-friendly time-series analyzer.

    Designed to integrate with AgriProfit without heavy external packages.
    Provides a stable API referenced by uncertainty/agriprofit_integration.py.
    """

    def __init__(self, prices_df: pd.DataFrame, frequency: str = "M"):
        self.df = prices_df.copy()
        if "date" not in self.df.columns or "price" not in self.df.columns:
            raise ValueError("prices_df must contain columns: date, price")
        self.df["date"] = pd.to_datetime(self.df["date"])  # ensure datetime
        self.df = self.df.sort_values("date").reset_index(drop=True)
        s = self.df.set_index("date")["price"].astype(float)
        # Coerce to regular monthly frequency if possible
        try:
            s = s.asfreq("MS")
        except Exception:
            pass
        self.series = s.interpolate(limit_direction="both")

    # ------------------------------
    # Distribution
    # ------------------------------
    def analyze_distribution_advanced(self) -> Dict:
        s = self.series.dropna()
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        out = {
            "summary": {
                "min": float(s.min()) if len(s) else None,
                "max": float(s.max()) if len(s) else None,
                "mean": float(s.mean()) if len(s) else None,
                "median": float(s.median()) if len(s) else None,
                "std": float(s.std()) if len(s) else None,
                "skew": float(s.skew()) if len(s) else None,
                "kurt": float(s.kurt()) if len(s) else None,
                "quantiles": {str(k): float(v) for k, v in q.to_dict().items()},
            }
        }
        return out

    # ------------------------------
    # Trend
    # ------------------------------
    def analyze_trend_advanced(self) -> Dict:
        s = self.series.dropna()
        n = len(s)
        if n < 3:
            return {"slope": None, "intercept": None, "r2": None}
        x = np.arange(n)
        y = s.values
        x_mean, y_mean = x.mean(), y.mean()
        slope = float(((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum())
        intercept = float(y_mean - slope * x_mean)
        y_pred = slope * x + intercept
        ss_res = float(((y - y_pred) ** 2).sum())
        ss_tot = float(((y - y_mean) ** 2).sum()) or 1.0
        r2 = float(1 - ss_res / ss_tot)
        return {
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
            "monthly_change": slope,
            "annual_change": slope * 12,
        }

    # ------------------------------
    # Spectral (simple FFT peak)
    # ------------------------------
    def analyze_spectral(self) -> Dict:
        s = self.series.dropna().values
        n = len(s)
        if n < 16:
            return {"dominant_periods": [], "note": "too short"}
        y = s - s.mean()
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(n, d=1)
        power = np.abs(fft) ** 2
        # Ignore zero frequency
        freqs, power = freqs[1:], power[1:]
        # Convert to periods (months per cycle), guard division by zero
        periods = np.where(freqs > 0, 1.0 / freqs, np.nan)
        order = np.argsort(power)[::-1]
        dom = []
        for idx in order[:3]:
            p = float(periods[idx]) if np.isfinite(periods[idx]) else None
            if p is not None and 1.5 <= p <= 60:
                dom.append({"period_months": round(p, 1), "power": float(power[idx])})
        return {"dominant_periods": dom}

    # ------------------------------
    # Nonlinear (Hurst via rescaled range)
    # ------------------------------
    def analyze_nonlinear(self) -> Dict:
        s = self.series.dropna().values
        if len(s) < 32:
            return {"hurst_exponent": {"rs_hurst": None}}
        # returns
        r = np.diff(np.log(s + 1e-12))
        # compute rescaled range across window sizes
        def rs_stat(x):
            x = x - x.mean()
            y = np.cumsum(x)
            R = y.max() - y.min()
            S = x.std(ddof=1) + 1e-12
            return R / S
        sizes = np.unique(np.floor(np.logspace(np.log10(8), np.log10(len(r)//2), 8)).astype(int))
        RS = []
        n_vals = []
        for m in sizes:
            if m < 8:
                continue
            k = len(r) // m
            if k < 2:
                continue
            vals = [rs_stat(r[i*m:(i+1)*m]) for i in range(k)]
            RS.append(np.mean(vals))
            n_vals.append(m)
        if len(n_vals) < 2:
            return {"hurst_exponent": {"rs_hurst": None}}
        x = np.log(n_vals)
        y = np.log(RS)
        slope = float(np.polyfit(x, y, 1)[0])
        return {"hurst_exponent": {"rs_hurst": slope}}

    # ------------------------------
    # Stationarity (ADF + KPSS)
    # ------------------------------
    def analyze_stationarity_advanced(self) -> Dict:
        s = self.series.dropna()
        out: Dict[str, Optional[float]] = {"adf_p": None, "kpss_p": None}
        try:
            out["adf_p"] = float(adfuller(s, autolag="AIC")[1])
        except Exception:
            pass
        try:
            out["kpss_p"] = float(kpss(s, regression="c", nlags="auto")[1])
        except Exception:
            pass
        concl = None
        if out["adf_p"] is not None and out["kpss_p"] is not None:
            if out["adf_p"] < 0.05 and out["kpss_p"] > 0.1:
                concl = "stationary"
            else:
                concl = "non-stationary"
        out["conclusion"] = concl
        return out

    # ------------------------------
    # ARIMA identification (small grid)
    # ------------------------------
    def identify_arima_models(self, seasonal: bool = True) -> Dict:
        s = self.series.dropna()
        if len(s) < 24:
            return {"best_models": []}
        orders = [(1,1,0), (1,1,1), (2,1,1), (0,1,1)]
        seas = [(0,1,1,12), (1,1,1,12)] if seasonal else [(0,0,0,0)]
        results: List[Dict] = []
        for o in orders:
            for so in seas:
                try:
                    model = SARIMAX(s, order=o, seasonal_order=so if seasonal else (0,0,0,0), enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False)
                    results.append({"order": o, "seasonal_order": so, "aic": float(res.aic)})
                except Exception:
                    continue
        results = sorted(results, key=lambda x: x.get("aic", np.inf))[:3]
        return {"best_models": results}

    # ------------------------------
    # Structural breaks (reuse EDA change points)
    # ------------------------------
    def detect_structural_breaks_advanced(self) -> Dict:
        cp = eda_change_points(self.df)
        brks = cp[cp["is_cp"]][["date", "score"]].to_dict(orient="records") if len(cp) else []
        return {"breakpoints": brks}

    # ------------------------------
    # Extreme values
    # ------------------------------
    def analyze_extreme_values(self) -> Dict:
        s = self.series.dropna()
        q01, q99 = float(s.quantile(0.01)), float(s.quantile(0.99))
        return {"p01": q01, "p99": q99, "span": q99 - q01}

    # ------------------------------
    # Information content (entropy of returns histogram)
    # ------------------------------
    def analyze_information_content(self, bins: int = 20) -> Dict:
        s = self.series.dropna().values
        if len(s) < 5:
            return {"entropy": None}
        r = np.diff(np.log(s + 1e-12))
        hist, _ = np.histogram(r, bins=bins, density=True)
        p = hist / (hist.sum() + 1e-12)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        return {"entropy": ent}

    # ------------------------------
    # Orchestrator
    # ------------------------------
    def run_ultimate_analysis(self) -> Dict:
        return {
            "distribution_advanced": self.analyze_distribution_advanced(),
            "trend_advanced": self.analyze_trend_advanced(),
            "spectral": self.analyze_spectral(),
            "nonlinear": self.analyze_nonlinear(),
            "stationarity_advanced": self.analyze_stationarity_advanced(),
            "arima_identification": self.identify_arima_models(),
            "structural_breaks_advanced": self.detect_structural_breaks_advanced(),
            "extreme_values": self.analyze_extreme_values(),
            "information_content": self.analyze_information_content(),
        }

