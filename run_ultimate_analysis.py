import argparse
import os
import pandas as pd
from ultimate_visualizer import integrate_with_agriprofit


def main():
    parser = argparse.ArgumentParser(description="Run Ultimate Time Series Analysis")
    parser.add_argument("--input", required=True, help="Input CSV file (columns: date,price)")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load
    df = pd.read_csv(args.input)
    if "date" not in df.columns or "price" not in df.columns:
        raise SystemExit("Input must contain columns: date, price")
    df["date"] = pd.to_datetime(df["date"])  # parse dates

    out = integrate_with_agriprofit(df)

    # Move artifacts into output folder
    for name in ("agriprofit_ultimate_analysis.png", "agriprofit_ultimate_report.pdf"):
        if os.path.exists(name):
            os.replace(name, os.path.join(args.output, name))

    print(f"\n✓ Results saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

