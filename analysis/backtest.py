import numpy as np
import pandas as pd
from typing import Callable, Dict

def rolling_backtest(
    prices: pd.DataFrame,
    fit_predict_fn: Callable[[pd.DataFrame, int], Dict],
    horizon: int = 3,
    min_train: int = 24,
    step: int = 1
) -> dict:
    """
    Роллинговый бэктест для оценки качества моделей прогноза.

    Аргументы:
        prices: DataFrame с колонками ["date", "price"].
        fit_predict_fn: функция вида fit(prices_subset, steps) -> {"forecast": DataFrame(date,yhat,yhat_lo,yhat_hi)}.
        horizon: шаги вперёд (например, 3 месяца).
        min_train: минимальное число точек для начала бэктеста.
        step: шаг окна (по умолчанию 1 месяц).

    Возвращает:
        dict с метриками:
          - mae: средняя абсолютная ошибка
          - rmse: корень из MSE
          - smape: симметричная MAPE (%)
          - r2: коэффициент детерминации (out-of-sample)
          - mase: MAE, нормированный на наивный прогноз
          - coverage90: доля фактических значений в интервале [yhat_lo, yhat_hi]
          - avg_interval_width: средняя ширина интервала (если есть)
          - pinball_q10 / pinball_q90: средний pinball loss для 0.1/0.9 (если есть)
          - by_horizon: метрики по горизонту (1..horizon)
          - residuals: массив ошибок (y_true - y_pred)
          - idxs: индексы дат прогнозов
    """

    prices = prices.sort_values("date").reset_index(drop=True)
    dates = prices["date"].values
    y = prices["price"].values

    preds = []
    trues = []
    idxs = []
    lows = []
    highs = []
    leads = []

    # проходимся по окнам
    for start in range(min_train, len(prices) - horizon + 1, step):
        train = prices.iloc[:start]
        piece = fit_predict_fn(train, horizon)
        fc = piece["forecast"]

        # сравниваем прогноз с фактами
        for j, d in enumerate(fc["date"]):
            if d in prices["date"].values:
                i = int(np.where(prices["date"] == d)[0][0])
                preds.append(fc.iloc[j]["yhat"])
                trues.append(y[i])
                lows.append(fc.iloc[j]["yhat_lo"] if "yhat_lo" in fc.columns else np.nan)
                highs.append(fc.iloc[j]["yhat_hi"] if "yhat_hi" in fc.columns else np.nan)
                idxs.append(i)
                leads.append(j + 1)

    preds = np.array(preds)
    trues = np.array(trues)
    residuals = trues - preds

    mae = float(np.mean(np.abs(residuals))) if len(residuals) else float("nan")
    smape = (
        float(100 * np.mean(2 * np.abs(preds - trues) / (np.abs(preds) + np.abs(trues) + 1e-9)))
        if len(residuals)
        else float("nan")
    )
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if len(residuals) else float("nan")
    r2 = (
        float(1.0 - (np.sum(residuals ** 2) / (np.sum((trues - np.mean(trues)) ** 2) + 1e-12)))
        if len(trues) > 1
        else float("nan")
    )

    # MASE denominator: seasonal naive if enough history, otherwise naive-1
    try:
        inferred = pd.infer_freq(pd.to_datetime(prices["date"]).sort_values())
    except Exception:
        inferred = None
    seasonal_m = 12 if (inferred and "M" in inferred) else (12 if len(y) >= 24 else 1)
    if len(y) > seasonal_m:
        denom = float(np.mean(np.abs(y[seasonal_m:] - y[:-seasonal_m])))
    else:
        denom = float(np.mean(np.abs(np.diff(y)))) if len(y) > 1 else float("nan")
    mase = float(mae / (denom + 1e-12)) if not np.isnan(denom) else float("nan")

    # Coverage интервалов
    coverage90 = None
    if lows and highs and not all(np.isnan(lows)) and not all(np.isnan(highs)):
        hits = (trues >= np.array(lows)) & (trues <= np.array(highs))
        coverage90 = float(hits.mean())
    # Interval width
    avg_interval_width = None
    if lows and highs and not all(np.isnan(lows)) and not all(np.isnan(highs)):
        widths = np.array(highs) - np.array(lows)
        avg_interval_width = float(np.nanmean(widths))

    # Pinball losses for available quantiles (0.1 and 0.9 if provided)
    def _pinball(y_true, q_pred, tau):
        e = y_true - q_pred
        return np.mean(np.maximum(tau * e, (tau - 1) * e))
    pinball_q10 = None
    pinball_q90 = None
    if lows and not all(np.isnan(lows)):
        pinball_q10 = float(_pinball(np.array(trues), np.array(lows), 0.10))
    if highs and not all(np.isnan(highs)):
        pinball_q90 = float(_pinball(np.array(trues), np.array(highs), 0.90))

    # Metrics by horizon lead
    by_horizon: Dict[int, Dict[str, float]] = {}
    if len(leads):
        leads_arr = np.array(leads)
        res_arr = residuals
        lows_arr, highs_arr = np.array(lows), np.array(highs)
        trues_arr = np.array(trues)
        for k in sorted(set(leads_arr)):
            mask = leads_arr == k
            if not np.any(mask):
                continue
            errs = res_arr[mask]
            entry: Dict[str, float] = {
                "mae": float(np.mean(np.abs(errs))),
                "rmse": float(np.sqrt(np.mean(errs ** 2))),
                "count": int(np.sum(mask)),
            }
            # coverage and width for this lead if intervals exist
            if lows and highs and not all(np.isnan(lows)) and not all(np.isnan(highs)):
                k_lows = lows_arr[mask]
                k_highs = highs_arr[mask]
                k_trues = trues_arr[mask]
                val_mask = ~np.isnan(k_lows) & ~np.isnan(k_highs)
                if np.any(val_mask):
                    hits_k = (k_trues[val_mask] >= k_lows[val_mask]) & (k_trues[val_mask] <= k_highs[val_mask])
                    entry["coverage"] = float(hits_k.mean())
                    entry["avg_width"] = float(np.nanmean(k_highs[val_mask] - k_lows[val_mask]))
            by_horizon[int(k)] = entry

    return {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "r2": r2,
        "mase": mase,
        "coverage90": coverage90,
        "avg_interval_width": avg_interval_width,
        "pinball_q10": pinball_q10,
        "pinball_q90": pinball_q90,
        "by_horizon": by_horizon,
        "residuals": residuals,
        "idxs": idxs,
    }