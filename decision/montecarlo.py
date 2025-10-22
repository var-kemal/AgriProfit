import numpy as np
import pandas as pd
from .profit import FarmParams, profit

def simulate_profit(prices_df: pd.DataFrame, params: FarmParams, residuals=None, n=5000, seed=7):
    rng = np.random.default_rng(seed)
    mu = prices_df["yhat"].values

    sims = []
    for i in range(len(mu)):
        if residuals is not None and len(residuals) > 10:
            noise = rng.choice(residuals, size=n, replace=True)
            sampled = mu[i] + noise
        else:
            lo, hi = prices_df.loc[i, ["yhat_lo", "yhat_hi"]]
            sigma = (hi - lo) / 3.92
            sampled = rng.normal(mu[i], sigma, size=n)

        pr = np.array([profit(p, params) for p in sampled])
        sims.append(pr)

    arr = np.vstack(sims).T
    best = arr.max(axis=1)
    return {
        "mean": float(best.mean()),
        "p_loss": float((best < 0).mean()),
        "var5": float(np.quantile(best, 0.05)),
        "cvar5": float(best[best <= np.quantile(best, 0.05)].mean()) if (best < 0).any() else float(best.mean()),
        "dist": best,
    }
