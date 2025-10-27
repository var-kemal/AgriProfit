from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional

from decision.profit import FarmParams, profit


def _ensure_prices(prices_df: pd.DataFrame, horizon: int = 12) -> pd.Series:
    series = prices_df.sort_values("date")["price"]
    if len(series) == 0:
        raise ValueError("No price data available for cash flow projection.")
    if len(series) >= horizon:
        return series.tail(horizon).reset_index(drop=True)
    pad = pd.Series([series.mean()] * (horizon - len(series)))
    return pd.concat([series, pad], ignore_index=True)


def project_cash_flows(
    prices_df: pd.DataFrame,
    params: FarmParams,
    months: int = 12,
    price_adjustment: Optional[float] = None,
) -> pd.DataFrame:
    """Returns projected monthly revenue, profit, and DSCR placeholders."""
    series = _ensure_prices(prices_df, months)
    if price_adjustment is not None:
        series = series * (1 + price_adjustment)
    cash = []
    for i, price in enumerate(series, 1):
        p = float(profit(price, params))
        cash.append({"month": i, "price": float(price), "profit": p})
    return pd.DataFrame(cash)


def _monthly_payment(loan_amount: float, annual_rate: float, term_months: int) -> float:
    if term_months <= 0:
        raise ValueError("Term months must be positive.")
    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        return loan_amount / term_months
    factor = (1 + monthly_rate) ** term_months
    payment = loan_amount * monthly_rate * factor / (factor - 1)
    return payment


def loan_capacity_estimate(
    cashflows: pd.DataFrame,
    annual_rate: float,
    term_months: int,
    dscr_target: float = 1.2,
) -> Dict:
    if cashflows.empty:
        raise ValueError("Cash flow data is empty.")
    avg_profit = float(cashflows["profit"].mean())
    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        capacity = avg_profit * term_months / dscr_target
    else:
        factor = (1 + monthly_rate) ** term_months
        annuity = monthly_rate * factor / (factor - 1)
        capacity = avg_profit / (annuity * dscr_target)
    capacity = max(capacity, 0.0)
    return {
        "average_monthly_profit": avg_profit,
        "recommended_max_loan": capacity,
        "dscr_target": dscr_target,
    }


def compute_dscr_series(
    cashflows: pd.DataFrame, loan_amount: float, annual_rate: float, term_months: int
) -> pd.DataFrame:
    payment = _monthly_payment(loan_amount, annual_rate, term_months)
    cashflows = cashflows.copy()
    cashflows["debt_service"] = payment
    cashflows["dscr"] = cashflows["profit"] / (payment + 1e-9)
    return cashflows
