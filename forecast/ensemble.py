import pandas as pd
from typing import Dict


def blend_forecasts(*pieces: Dict) -> pd.DataFrame:
    """Average yhat / yhat_lo / yhat_hi across provided forecast dicts."""
    dfs = [p["forecast"].set_index("date") for p in pieces]
    combo = pd.concat(dfs, axis=1)
    out = pd.DataFrame(index=dfs[0].index)
    for col in ("yhat", "yhat_lo", "yhat_hi"):
        cols = [c for c in combo.columns if c.endswith(col)]
        out[col] = combo[cols].mean(axis=1)
    out = out.reset_index()
    return out


