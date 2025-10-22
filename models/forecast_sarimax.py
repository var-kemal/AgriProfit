import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict


def sarimax_forecast(prices: pd.DataFrame, steps: int = 3, order=(1,1,1), seasonal_order=(0,1,1,12)) -> Dict:
    series = prices.set_index("date")["price"].asfreq("MS")
    series = series.interpolate(limit_direction="both")
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=steps)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)  # 95%
    out = pd.DataFrame({
        "date": mean.index,
        "yhat": mean.values,
        "yhat_lo": conf.iloc[:, 0].values,
        "yhat_hi": conf.iloc[:, 1].values,
    })
    return {"model": res, "forecast": out}
