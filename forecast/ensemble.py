import pandas as pd
from typing import Dict, List, Optional


def blend_forecasts(*pieces: Dict, weights: Optional[List[float]] = None) -> pd.DataFrame:
    """Blend yhat / yhat_lo / yhat_hi across forecast dicts.

    - If `weights` is None, uses equal weights.
    - Otherwise, uses normalized weights (len must match number of pieces).
    """
    if len(pieces) == 0:
        return pd.DataFrame(columns=["date", "yhat", "yhat_lo", "yhat_hi"])

    dfs = [p["forecast"].set_index("date")[["yhat", "yhat_lo", "yhat_hi"]] for p in pieces]
    combo = pd.concat(dfs, axis=1, keys=range(len(dfs)))  # MultiIndex columns: (model, col)

    n = len(dfs)
    if weights is None:
        w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError("weights length must match number of forecasts")
        s = float(sum(weights)) or 1.0
        w = [wi / s for wi in weights]

    out = pd.DataFrame(index=dfs[0].index)
    for col in ("yhat", "yhat_lo", "yhat_hi"):
        cols = combo.xs(col, axis=1, level=1)
        out[col] = (cols * w).sum(axis=1)
    out = out.reset_index()
    return out


