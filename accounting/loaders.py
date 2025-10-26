from __future__ import annotations

import io
from typing import Optional

import pandas as pd


REQUIRED_COLS = {"account_code", "account_name", "debit", "credit"}


def load_trial_balance_csv(uploaded: Optional[io.BytesIO]) -> pd.DataFrame:
    """Load Turkish-style trial balance (mizan) CSV.

    Expected columns:
      - account_code (string, e.g., 100, 102, 600, 700...)
      - account_name (string)
      - debit (number)
      - credit (number)
      - date (optional, parseable; used for monthly budget aggregation)
    """
    if uploaded is None:
        return make_example_trial_balance()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        raise ValueError(f"Could not read Trial Balance CSV: {e}")

    if not REQUIRED_COLS.issubset(df.columns):
        raise ValueError(
            f"Trial Balance CSV must include columns: {sorted(REQUIRED_COLS)}"
        )

    # Basic cleaning
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["account_code"] = df["account_code"].astype(str).str.strip()
    for c in ("debit", "credit"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df


def make_example_trial_balance() -> pd.DataFrame:
    data = [
        # Assets (1xx current), (2xx non-current)
        {"date": "2024-01-31", "account_code": "100", "account_name": "Kasa", "debit": 150000, "credit": 0},
        {"date": "2024-01-31", "account_code": "102", "account_name": "Bankalar", "debit": 350000, "credit": 0},
        {"date": "2024-01-31", "account_code": "153", "account_name": "Ticari Mallar", "debit": 120000, "credit": 0},
        {"date": "2024-01-31", "account_code": "254", "account_name": "Taşıtlar", "debit": 500000, "credit": 0},
        # Liabilities (3xx short), (4xx long) and Equity (5xx)
        {"date": "2024-01-31", "account_code": "300", "account_name": "Banka Kredileri", "debit": 0, "credit": 200000},
        {"date": "2024-01-31", "account_code": "320", "account_name": "Satıcılar", "debit": 0, "credit": 80000},
        {"date": "2024-01-31", "account_code": "500", "account_name": "Sermaye", "debit": 0, "credit": 700000},
        # Income (6xx) and Expenses (7xx)
        {"date": "2024-01-31", "account_code": "600", "account_name": "Yurt İçi Satışlar", "debit": 0, "credit": 220000},
        {"date": "2024-01-31", "account_code": "620", "account_name": "Satılan Mamul Maliyeti", "debit": 150000, "credit": 0},
        {"date": "2024-01-31", "account_code": "770", "account_name": "Genel Yönetim Giderleri", "debit": 30000, "credit": 0},
    ]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])  # parse date
    return df