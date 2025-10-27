# AgriProfit — Data Analysis & Profit Planning (Temporary README)

This project turns raw agricultural price and finance data into actionable insights. It provides a rich analysis toolkit (EDA), optional short‑horizon forecasting, profit scenarios, risk metrics, and basic accounting summaries — all available in a Streamlit app, a React demo app, and programmatic utilities.

## Quick Start

- Python (Streamlit app)
  - Create env and install deps:
    - `python -m venv venv && source venv/bin/activate` (or Windows `venv\Scripts\activate`)
    - `pip install -r requirements.txt`
  - Run the app: `streamlit run ui/app_streamlit.py`
  - Default starts in analysis‑only mode; enable forecasting in the sidebar if needed.

- React demo app (Deep Data Analyzer)
  - `cd my-app && npm install && npm start`
  - Opens a styled analysis demo on http://localhost:3000 (uses synthetic data).

- “Ultimate Analysis” (dashboard + report)
  - Inside Streamlit: open the tab "🔬 Ultimate Analysis" and it will render a compact dashboard and expose PNG/PDF downloads.
  - From CLI: `python run_ultimate_analysis.py --input path/to/prices.csv --output results/`

## Data Formats

- Prices (monthly)
  - CSV columns: `date, price`
  - Use ISO dates (YYYY‑MM‑DD), typically the first day of each month (e.g., `2024‑01‑01`).
- Yields (annual)
  - CSV columns: `year, yield_per_ha` (year is an integer).
- Costs (per hectare)
  - CSV columns: `item, amount` (no date column).
- Trial Balance (optional, TDHP style)
  - CSV columns: `account_code, account_name, debit, credit[, date]` (date optional).

If you don’t upload files, the app uses synthetic defaults:
- Prices/yields/costs: `data/loaders.py` (see `make_synthetic_prices` and `make_synthetic_yield_costs`).
- Example trial balance: `accounting/loaders.py` (`make_example_trial_balance`).

## What’s Inside (Repo Map)

- `analysis/eda.py` — Core analysis (data quality, rolling stats, seasonality, stationarity, decomposition, anomalies, change points, ACF/PACF, returns, distribution, drawdowns, clustering, insights report, full data printer).
- `analysis/backtest.py` — Rolling backtest helper with MAE, RMSE, MASE, R², pinball loss, coverage, width, and per‑horizon metrics.
- `models/forecast_quantile_gbr.py` — Quantile Gradient Boosted Regression with corrected iterative multi‑step logic (non‑crossing quantiles).
- `models/forecast_sarimax.py` — SARIMAX forecast with confidence intervals.
- `forecast/ensemble.py` — Simple forecast blender (yhat/lo/hi).
- `uncertainty/conformal.py` — Conformal symmetric interval helper.
- `ultimate_time_series_analysis.py` — Lightweight analyzer (distribution/trend/spectral/entropy/Hurst/stationarity/ARIMA ID/structural breaks).
- `ultimate_visualizer.py` — Builds a compact Matplotlib dashboard + PNG/PDF and integrates into Streamlit.
- `ui/app_streamlit.py` — Main Streamlit app (analysis‑first; optional forecasting; profit, risk, accounting; ultimate tab; ZIP download).
- `my-app/` — React demo app (Deep Data Analyzer) with Tailwind styling, Recharts, dark mode toggle.
- `data/`, `accounting/`, `decision/` — Loaders/validators, trial balance aggregations, profit & Monte Carlo.

## Streamlit App Overview

Tabs (analysis‑first; some require forecasting to be enabled in the sidebar):
- Data Analysis
  - Overview metrics, Data Quality (gaps/duplicates/frequency), Rolling mean/std, Seasonality profile + heatmap, Seasonality strength, Stationarity (ADF/KPSS), Decomposition, Anomalies (MAD) + Change Points, Distribution & Returns, Drawdowns, State Clustering, Insights & Recommendations.
  - Download all current datasets as a single ZIP.
- Prices & Forecast (optional)
  - Observed series, blended forecast, interval band; enable forecasting in sidebar.
- Profit Scenarios (optional)
  - P10/P50/P90 monthly scenarios, comparison vs “sell now”, animated views.
- Accounting
  - Balance sheet and income statement from TDHP trial balance; optional monthly budget forecast if forecasting is enabled.
- Risk & Advice (optional)
  - Monte Carlo best‑month profit distribution, probability of loss, and VaR.
- Quality (optional)
  - Backtest metrics for each model (MAE/RMSE/MASE/R²/coverage/pinball) + per‑horizon tables.
- 🔬 Ultimate Analysis
  - Compact Matplotlib dashboard from the Ultimate analyzer + PNG/PDF downloads.

Sidebar controls:
- Upload CSVs (prices/costs/yields/trial balance), farm parameters, mode toggle (Enable forecasting), forecast settings (horizon, coverage, blending).

## React Demo (my‑app)

- Component: `src/deep_data_analyzer.tsx` — A styled React/TypeScript analyzer demo using Recharts and Tailwind (dark mode, pill tabs, glass cards).
- Entry: `src/App.tsx` renders the analyzer. Run via `npm start`.
- Tailwind config is already set up (`tailwind.config.js`, `postcss.config.js`, `src/index.css`).

## Forecasting Notes (optional feature)

- Models: SARIMAX and quantile GBDT (P10/P50/P90). Iterative forecasting logic in GBDT is fixed and enforces non‑crossing quantiles.
- Evaluation: rolling backtest reports MAE, RMSE, MASE (vs naive/seasonal naive), R² (out‑of‑sample), pinball loss, coverage, average interval width, and per‑horizon breakdown.
- Calibration: conformal symmetric radius (basic). For production, prefer calibrating the blended pipeline and/or asymmetric conformal methods.

## Profit & Risk

- Profit calculator converts price → revenue → net profit with farm area, yield, unit, and costs.
- Monte Carlo simulates best‑month profit using forecast residuals (or historical variability) and reports mean, probability of loss, and VaR.

## Accounting

- Trial balance loader (TDHP) and simple aggregations for Balance Sheet/Income Statement.
- Optional monthly budget forecast when dates are provided and forecasting is enabled.

## Synthetic Data

- Prices: `data/loaders.py` → `make_synthetic_prices(start, periods, seed)` uses a seasonal + trend + noise index scaled to a base, monthly frequency.
- Yields & Costs: `data/loaders.py` → `make_synthetic_yield_costs(seed)` returns a small annual yields table and fixed per‑ha costs.
- Trial Balance: `accounting/loaders.py` → `make_example_trial_balance()` returns a small TDHP‑style example.

## Known Limitations & Next Steps

- Quantile blending is not a true quantile of the mixture; can miscalibrate; consider quantile regression averaging or conformalized quantile regression post‑blend.
- Change point detection is simple; upgrade to a dedicated method (e.g., ruptures/PELT) if needed.
- Univariate price models only; adding exogenous factors (FX, futures, inputs, weather) improves medium‑horizon skill.
- API/SDK not yet scaffolded; see “Roadmap” below for planned endpoints.

## Roadmap Snapshot

- API (FastAPI) MVP: `/analyze`, `/profit-scenarios`, optional `/forecast`.
- Multi‑page PDF reports with metrics + narratives.
- Portfolio view (multi‑dataset heatmaps), shareable links, authentication.

## Contributing (Temporary)

- Keep PRs focused and scoped. Avoid changing file names unless necessary.
- Follow existing code style (TypeScript strict in React app; Python type hints where practical).
- Please open an issue for questions or proposed features.

---

Questions or ideas? Open an issue or leave notes in `uncertainty/agriprofit_integration.py` for integration tasks.

