import numpy as np
import pandas as pd
from typing import Tuple


def conformal_interval(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Return quantile radius for symmetric intervals (absolute residuals)."""
    abs_res = np.abs(residuals)
    q = np.quantile(abs_res, 1 - alpha)
    return float(q)


def apply_conformal(forecast_df: pd.DataFrame, radius: float) -> pd.DataFrame:
    out = forecast_df.copy()
    out["yhat_lo"] = out["yhat"] - radius
    out["yhat_hi"] = out["yhat"] + radius
    return out
