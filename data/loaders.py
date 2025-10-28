import io
import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from .validators import validate_prices, validate_costs, validate_yields

def load_price_csv(uploaded: Optional[io.BytesIO]) -> pd.DataFrame:
    """Load prices CSV with sensible fallbacks for locale formats.

    Supported variations:
      - Standard: comma separator, dot decimal (date,price)
      - Locale CSV: semicolon separator, comma decimal (date;price)
      - Strips BOM/whitespace; coercively parses price if it's a string with commas
    """
    if uploaded is None:
        return make_synthetic_prices()

    def _try_read(f, **kwargs) -> pd.DataFrame:
        if hasattr(f, "seek"):
            try:
                f.seek(0)
            except Exception:
                pass
        return pd.read_csv(f, **kwargs)

    # Attempt reads with common variants
    read_attempts = [
        {},  # default (comma, dot)
        {"encoding": "utf-8-sig"},  # BOM-safe
        {"sep": ";"},  # semicolon
        {"sep": ";", "decimal": ","},  # semicolon + comma decimal
        {"sep": ";", "decimal": ",", "encoding": "utf-8-sig"},
    ]

    last_err = None
    df = None
    for opts in read_attempts:
        try:
            df = _try_read(uploaded, **opts)
            break
        except Exception as e:  # pragma: no cover
            last_err = e
            continue
    if df is None:
        raise ValueError(f"Не удалось прочитать CSV: {last_err}")

    # Normalize column names (strip BOMs and spaces, lowercase)
    df.columns = [str(c).replace("\ufeff", "").strip().lower() for c in df.columns]
    # Minimal rename support (in case of localized headers)
    rename_map = {"датa": "date", "дата": "date", "цена": "price"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required_cols = {"date", "price"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"CSV должен содержать колонки: {required_cols}. Найдено: {list(df.columns)}"
        )

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("В колонке 'date' есть некорректные значения")

    # Coerce price (handle comma decimals and thousands separators if present)
    if not np.issubdtype(df["price"].dtype, np.number):
        ser = df["price"].astype(str)
        # Remove spaces and NBSPs, strip thousands dots if decimal comma used
        ser = ser.str.replace("\xa0", "").str.replace(" ", "")
        ser = ser.str.replace(".", "", regex=False)  # drop thousands '.'
        ser = ser.str.replace(",", ".", regex=False)  # decimal comma -> dot
        df["price"] = pd.to_numeric(ser, errors="coerce")

    return validate_prices(df)


def load_costs_csv(uploaded: Optional[io.BytesIO]) -> pd.DataFrame:
    if uploaded is None:
        _, costs = make_synthetic_yield_costs()
        return validate_costs(costs)
    df = pd.read_csv(uploaded)
    if not set(["item", "amount"]).issubset(df.columns):
        raise ValueError("Costs CSV must contain columns: item, amount")
    df = validate_costs(df)
    return df


def load_yields_csv(uploaded: Optional[io.BytesIO]) -> pd.DataFrame:
    if uploaded is None:
        yields, _ = make_synthetic_yield_costs()
        return validate_yields(yields)
    df = pd.read_csv(uploaded)
    if not set(["year", "yield_per_ha"]).issubset(df.columns):
        raise ValueError("Yields CSV must contain columns: year, yield_per_ha")
    df = validate_yields(df)
    return df


def make_synthetic_prices(start="2015-01-01", periods=11*12, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="MS")
    t = np.arange(periods)
    seasonal = 0.12 * np.sin(2 * np.pi * t / 12.0)
    trend = 0.006 * t
    noise = rng.normal(0, 0.03, size=periods)
    idx = 1.5 + seasonal + trend + noise
    base, scale = 16, 4
    price = base + scale * (idx - idx.mean()) / idx.std()
    return pd.DataFrame({"date": dates, "price": price.round(2)})



def make_synthetic_yield_costs(seed=7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    years = np.arange(2021, 2026)
    yield_per_ha = np.maximum(3.0, rng.normal(5.0, 0.5, size=len(years)))
    yields = pd.DataFrame({"year": years, "yield_per_ha": yield_per_ha.round(2)})
    costs = pd.DataFrame([
        ("Seeds", 1200), ("Fertilizer", 3500), ("Fuel", 2200), ("Labor", 4000), ("Irrigation", 1500)
    ], columns=["item", "amount"])
    return yields, costs


# -------------------------------------------------------------
# Utilities to write synthetic datasets to CSV (developer aid)
# -------------------------------------------------------------
def write_synthetic_prices_csv(
    output_path: Optional[str] = None,
    start: str = "2021-01-01",
    periods: int = 48,
    seed: int = 7,
) -> str:
    """Write a synthetic prices CSV into the data folder (or a custom path).

    Default path: data/synthetic_prices.csv (next to this file).

    Returns the absolute path to the written file.
    """
    df = make_synthetic_prices(start=start, periods=periods, seed=seed)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "synthetic_prices.csv")
    df.to_csv(output_path, index=False)
    return os.path.abspath(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data loaders utilities")
    parser.add_argument(
        "--make-synthetic-prices",
        action="store_true",
        help="Write synthetic prices CSV to data/synthetic_prices.csv (or --output)",
    )
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date for synthetic series")
    parser.add_argument("--periods", type=int, default=48, help="Number of monthly periods")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    if args.make_synthetic_prices:
        path = write_synthetic_prices_csv(
            output_path=args.output, start=args.start, periods=args.periods, seed=args.seed
        )
        print(f"Wrote synthetic prices to {path}")
    else:
        parser.print_help()
