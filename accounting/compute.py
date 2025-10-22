from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


def _class_of(code: str) -> int:
    try:
        return int(str(code).strip()[0])
    except Exception:
        return -1


def balance_sheet(tb: pd.DataFrame) -> Dict:
    """Compute balance sheet group totals from trial balance.

    Rules (TDHP-style):
      - Assets (Aktif): 1xx (current), 2xx (non-current) – use net debit balances max(debit-credit, 0).
      - Liabilities (Pasif): 3xx (short-term), 4xx (long-term) – use net credit balances max(credit-debit, 0).
      - Equity: 5xx – use net credit balances max(credit-debit, 0).
    Returns dict with totals and grouped frames for display.
    """
    df = tb.copy()
    df["class"] = df["account_code"].apply(_class_of)
    df["net_debit"] = (df["debit"] - df["credit"]).clip(lower=0)
    df["net_credit"] = (df["credit"] - df["debit"]).clip(lower=0)

    assets_current = df.loc[df["class"] == 1, "net_debit"].sum()
    assets_noncurrent = df.loc[df["class"] == 2, "net_debit"].sum()
    liabilities_short = df.loc[df["class"] == 3, "net_credit"].sum()
    liabilities_long = df.loc[df["class"] == 4, "net_credit"].sum()
    equity = df.loc[df["class"] == 5, "net_credit"].sum()

    total_assets = assets_current + assets_noncurrent
    total_pasif = liabilities_short + liabilities_long + equity
    gap = total_assets - total_pasif

    groups = {
        "assets_current": df.loc[df["class"] == 1, ["account_code", "account_name", "net_debit"]]
        .rename(columns={"net_debit": "balance"})
        .sort_values("account_code"),
        "assets_noncurrent": df.loc[df["class"] == 2, ["account_code", "account_name", "net_debit"]]
        .rename(columns={"net_debit": "balance"})
        .sort_values("account_code"),
        "liabilities_short": df.loc[df["class"] == 3, ["account_code", "account_name", "net_credit"]]
        .rename(columns={"net_credit": "balance"})
        .sort_values("account_code"),
        "liabilities_long": df.loc[df["class"] == 4, ["account_code", "account_name", "net_credit"]]
        .rename(columns={"net_credit": "balance"})
        .sort_values("account_code"),
        "equity": df.loc[df["class"] == 5, ["account_code", "account_name", "net_credit"]]
        .rename(columns={"net_credit": "balance"})
        .sort_values("account_code"),
    }

    return {
        "assets_current": float(assets_current),
        "assets_noncurrent": float(assets_noncurrent),
        "liabilities_short": float(liabilities_short),
        "liabilities_long": float(liabilities_long),
        "equity": float(equity),
        "total_assets": float(total_assets),
        "total_pasif": float(total_pasif),
        "gap": float(gap),
        "groups": groups,
    }


def income_statement(tb: pd.DataFrame) -> Dict:
    """Compute simple income statement from trial balance.

    - Revenues: class 6 – net credit (credit - debit)
    - Expenses/COGS: class 7 – net debit (debit - credit)
    - Net income: revenues - expenses
    """
    df = tb.copy()
    df["class"] = df["account_code"].apply(_class_of)
    revenue = df.loc[df["class"] == 6, ["credit", "debit"]].eval("credit - debit").clip(lower=0).sum()
    expense = df.loc[df["class"] == 7, ["debit", "credit"]].eval("debit - credit").clip(lower=0).sum()
    net = revenue - expense
    return {"revenue": float(revenue), "expense": float(expense), "net": float(net)}


def monthly_income_expense(tb: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly revenues (class 6) and expenses (class 7) if 'date' present.

    Returns DataFrame with columns: date, revenue, expense, net
    """
    if "date" not in tb.columns or tb["date"].isna().all():
        return pd.DataFrame(columns=["date", "revenue", "expense", "net"])

    df = tb.copy()
    df = df.dropna(subset=["date"]).copy()
    df["class"] = df["account_code"].apply(_class_of)
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

    rev = (
        df.loc[df["class"] == 6]
        .groupby("month")[["credit", "debit"]]
        .sum()
        .eval("credit - debit")
        .clip(lower=0)
        .rename("revenue")
    )
    exp = (
        df.loc[df["class"] == 7]
        .groupby("month")[["debit", "credit"]]
        .sum()
        .eval("debit - credit")
        .clip(lower=0)
        .rename("expense")
    )
    out = pd.concat([rev, exp], axis=1).fillna(0.0)
    out["net"] = out["revenue"] - out["expense"]
    out = out.reset_index().rename(columns={"month": "date"})
    return out

