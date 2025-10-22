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
          - smape: симметричная MAPE (%)
          - coverage90: доля фактических значений в интервале [yhat_lo, yhat_hi]
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

    preds = np.array(preds)
    trues = np.array(trues)
    residuals = trues - preds

    mae = float(np.mean(np.abs(residuals))) if len(residuals) else float("nan")
    smape = (
        float(100 * np.mean(2 * np.abs(preds - trues) / (np.abs(preds) + np.abs(trues) + 1e-9)))
        if len(residuals)
        else float("nan")
    )

    # Coverage интервалов
    coverage90 = None
    if lows and highs and not all(np.isnan(lows)) and not all(np.isnan(highs)):
        hits = (trues >= np.array(lows)) & (trues <= np.array(highs))
        coverage90 = float(hits.mean())

    return {
        "mae": mae,
        "smape": smape,
        "coverage90": coverage90,
        "residuals": residuals,
        "idxs": idxs,
    }
