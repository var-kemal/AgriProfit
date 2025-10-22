import pandas as pd


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date", "price"]).copy()
    df = df.sort_values("date")
    # basic sanity
    df = df[(df["price"] > 0) & (df["price"] < df["price"].quantile(0.995))]
    return df.reset_index(drop=True)


def validate_costs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["item", "amount"]).copy()
    df = df[df["amount"] >= 0]
    return df.reset_index(drop=True)


def validate_yields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["year", "yield_per_ha"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[(df["yield_per_ha"] >= 0.5) & (df["yield_per_ha"] <= 30)]
    return df.reset_index(drop=True)
