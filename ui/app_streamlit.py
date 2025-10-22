import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sys, os

# Ensure repo modules importable when running `streamlit run ui/app_streamlit.py`
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.loaders import load_price_csv, load_costs_csv, load_yields_csv
from accounting.loaders import load_trial_balance_csv
from accounting.compute import balance_sheet, income_statement, monthly_income_expense
from decision.profit import FarmParams, profit
from decision.montecarlo import simulate_profit
from models.forecast_sarimax import sarimax_forecast
from models.forecast_quantile_gbr import quantile_gbr_forecast
from forecast.ensemble import blend_forecasts
from analysis.backtest import rolling_backtest
from uncertainty.conformal import conformal_interval, apply_conformal


# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="AgriProfit v2", layout="wide")
st.title("AgriProfit v2 - Profit Planning From Price Forecasts")
st.caption("Upload price history, costs, and yields, or use built-in synthetic data to explore scenarios.")


# -----------------------
# Sidebar: inputs
# -----------------------
with st.sidebar:
    st.header("Data Uploads")
    up_price = st.file_uploader("Prices CSV (columns: date,price)", type=["csv"], key="price")
    up_costs = st.file_uploader("Costs CSV (columns: item,amount) — amounts per hectare", type=["csv"], key="costs")
    up_yield = st.file_uploader("Yields CSV (columns: year,yield_per_ha)", type=["csv"], key="yield")
    with st.expander("Download CSV templates"):
        tpl_prices = "date,price\n2022-01-01,10.5\n2022-02-01,11.2\n"
        tpl_costs = "item,amount\nSeeds,1200\nFertilizer,3500\n"
        tpl_yields = "year,yield_per_ha\n2023,5.0\n2024,5.2\n"
        st.download_button("Template: prices.csv", data=tpl_prices, file_name="prices_template.csv", mime="text/csv")
        st.download_button("Template: costs.csv", data=tpl_costs, file_name="costs_template.csv", mime="text/csv")
        st.download_button("Template: yields.csv", data=tpl_yields, file_name="yields_template.csv", mime="text/csv")

    st.header("Accounting (Trial Balance)")
    up_tb = st.file_uploader(
        "Trial Balance CSV (account_code,account_name,debit,credit[,date])",
        type=["csv"],
        key="trial_balance",
        help="Upload a Turkish TDHP-style trial balance (mizan). Date optional for monthly budgets.",
    )

    st.header("Farm Settings")
    area = st.number_input("Area (hectares)", min_value=0.5, max_value=500.0, value=10.0, step=0.5)
    y_manual = st.checkbox("Override yield manually?", value=False)
    y_t_ha = st.number_input("Yield (tons/ha)", min_value=0.5, max_value=30.0, value=5.0, step=0.1) if y_manual else None
    price_unit = st.selectbox("Price unit", ["per_kg", "per_ton"], index=1)

    st.header("Forecast Settings")
    horizon = st.slider("Forecast horizon (months)", 1, 12, 3, 1)
    coverage = st.slider("Prediction interval coverage (%)", 80, 98, 90, 1, help="Higher coverage = wider bands")
    interval_alpha = 1 - (coverage / 100.0)
    blend_mode = st.selectbox("Blending mode", ["Auto-weighted (MAE)", "Equal weights", "Manual weights"], index=0)
    manual_w_sarimax = st.slider("Manual weight: SARIMAX", 0.0, 1.0, 0.5, 0.05) if blend_mode == "Manual weights" else None


# -----------------------
# Load data (with sensible defaults)
# -----------------------
try:
    prices = load_price_csv(up_price)
    costs = load_costs_csv(up_costs)
    yields_df = load_yields_csv(up_yield)
except Exception as e:
    st.error(f"Could not read inputs: {e}")
    st.stop()

if y_t_ha is None:
    y_t_ha = float(yields_df.sort_values("year").iloc[-1]["yield_per_ha"]) if len(yields_df) else 5.0

params = FarmParams(
    area_ha=area,
    yield_t_per_ha=y_t_ha,
    price_unit=price_unit,
    costs_total=float(costs["amount"].sum() * area),  # assume per-ha costs
)


# -----------------------
# Cached helpers
# -----------------------
@st.cache_data(show_spinner=False)
def _df_signature(df: pd.DataFrame) -> str:
    # Lightweight signature for caching: dates+prices only
    cols = [c for c in df.columns if c in ("date", "price")]
    return df[cols].to_csv(index=False)


@st.cache_data(show_spinner=True, ttl=60, hash_funcs={pd.DataFrame: _df_signature})
def cached_backtest_qgbr(prices_df: pd.DataFrame, horizon_bt: int) -> dict:
    fit_qgbr = lambda df, steps: quantile_gbr_forecast(df, steps)
    return rolling_backtest(
        prices_df,
        fit_qgbr,
        horizon=horizon_bt,
        min_train=min(24, max(12, len(prices_df) // 2)),
    )


@st.cache_data(show_spinner=True, ttl=60, hash_funcs={pd.DataFrame: _df_signature})
def cached_backtest_sarimax(prices_df: pd.DataFrame, horizon_bt: int) -> dict:
    fit_sarimax = lambda df, steps: sarimax_forecast(df, steps)
    return rolling_backtest(
        prices_df,
        fit_sarimax,
        horizon=horizon_bt,
        min_train=min(24, max(12, len(prices_df) // 2)),
    )


# Separate cached fits for weighted blending
@st.cache_data(show_spinner=True, ttl=60, hash_funcs={pd.DataFrame: _df_signature})
def cached_sarimax(prices_df: pd.DataFrame, steps: int):
    return sarimax_forecast(prices_df, steps)


@st.cache_data(show_spinner=True, ttl=60, hash_funcs={pd.DataFrame: _df_signature})
def cached_qgbr(prices_df: pd.DataFrame, steps: int):
    return quantile_gbr_forecast(prices_df, steps)


# -----------------------
# Compute backtest, conformal radius, and forecasts
# -----------------------
if len(prices) < 8:
    st.warning("Very short price history (<8 months). Forecast quality may be poor.")

bt_q = cached_backtest_qgbr(prices, horizon_bt=3)
bt_s = cached_backtest_sarimax(prices, horizon_bt=3)
radius = conformal_interval(bt_q["residuals"], alpha=float(interval_alpha)) if len(prices) > 18 else 0.0

# Weights per model
eps = 1e-6
if blend_mode == "Equal weights":
    weights = [0.5, 0.5]  # [SARIMAX, QGBR]
elif blend_mode == "Manual weights" and manual_w_sarimax is not None:
    ws = float(manual_w_sarimax)
    weights = [ws, 1 - ws]
else:
    w_q = 1.0 / (float(bt_q["mae"]) + eps) if not math.isnan(bt_q["mae"]) else 1.0
    w_s = 1.0 / (float(bt_s["mae"]) + eps) if not math.isnan(bt_s["mae"]) else 1.0
    s_w = (w_q + w_s) or 1.0
    weights = [w_s / s_w, w_q / s_w]  # [SARIMAX, QGBR]

m1 = cached_sarimax(prices, steps=horizon)
m2 = cached_qgbr(prices, steps=horizon)
blend = blend_forecasts(m1, m2, weights=weights)
blend = apply_conformal(blend, radius)


# -----------------------
# Charts and results
# -----------------------
tabs = st.tabs(["Prices & Forecast", "Profit Scenarios", "Accounting", "Risk & Advice", "Quality"])

with tabs[0]:
    st.subheader("Price History and Forecast")
    # KPIs
    last_price = float(prices.dropna().iloc[-1]["price"]) if len(prices) else float("nan")
    last12 = prices.set_index("date")["price"].last("365D") if len(prices) else pd.Series(dtype=float)
    colK1, colK2, colK3 = st.columns(3)
    colK1.metric("Latest price", f"{last_price:,.2f}")
    colK2.metric("Avg last 12m", f"{last12.mean():,.2f}" if len(last12) else "—")
    colK3.metric("Volatility 12m", f"{last12.std():,.2f}" if len(last12) else "—")

    # Forecast with shaded band
    base = px.line(prices, x="date", y="price", title="Observed prices")
    fig = go.Figure(base.data)
    lo_label = f"P{int((interval_alpha/2)*100)}"
    hi_label = f"P{int((1-interval_alpha/2)*100)}"
    fig.add_trace(go.Scatter(x=blend["date"], y=blend["yhat_lo"], name=lo_label, mode="lines", line=dict(color="rgba(0, 123, 255, 0.2)")))
    fig.add_trace(go.Scatter(x=blend["date"], y=blend["yhat_hi"], name=hi_label, mode="lines", fill="tonexty", line=dict(color="rgba(0, 123, 255, 0.2)")))
    fig.add_trace(go.Scatter(x=blend["date"], y=blend["yhat"], name="Forecast P50", mode="lines", line=dict(color="#007bff")))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Preview input data"):
        c1, c2, c3 = st.columns(3)
        c1.write("Prices")
        c1.dataframe(prices.head(12))
        c2.write("Costs")
        c2.dataframe(costs.head(12))
        c3.write("Yields")
        c3.dataframe(yields_df.head(12))

    # Download forecast CSV
    fc_csv = blend.to_csv(index=False).encode("utf-8")
    st.download_button("Download forecast (CSV)", data=fc_csv, file_name="price_forecast.csv", mime="text/csv")

with tabs[1]:
    st.subheader("Profit: Sell Now vs. Future Months")
    now_profit = profit(last_price, params)

    rows = [
        {
            "when": "Sell now",
            "date": pd.to_datetime(prices.iloc[-1]["date"]).date() if len(prices) else None,
            "scenario": "Current",
            "ref_price": last_price,
            "profit": now_profit,
        }
    ]
    for _, r in blend.iterrows():
        for name, p in [("P10", r["yhat_lo"]), ("P50", r["yhat"]), ("P90", r["yhat_hi"])]:
            rows.append(
                {
                    "when": "Future",
                    "date": pd.to_datetime(r["date"]).date(),
                    "scenario": name,
                    "ref_price": float(r["yhat"]),
                    "profit": profit(float(p), params),
                }
            )

    tab_df = pd.DataFrame(rows)

    # Choose a target month and scenario
    if len(blend):
        sell_month = st.selectbox("Plan to sell in", [d.date() for d in pd.to_datetime(blend["date"])])
        sell_scenario = st.selectbox("Scenario", ["P50", "P10", "P90"], index=0)
    else:
        sell_month, sell_scenario = None, "P50"

    # Animated visualization
    anim_type = st.radio("Animation", ["By Scenario", "By Months"], horizontal=True)
    df_future = tab_df[tab_df["when"] == "Future"].copy()
    if sell_month is not None:
        df_future["selected"] = pd.to_datetime(df_future["date"]) == pd.to_datetime(sell_month)
    else:
        df_future["selected"] = False

    if anim_type == "By Scenario":
        fig2 = px.bar(
            df_future,
            x="date",
            y="profit",
            animation_frame="scenario",
            color="selected",
            color_discrete_map={True: "crimson", False: "steelblue"},
            title="Profit by month — animating scenarios",
        )
        fig2.update_yaxes(title_text="Profit")
        fig2.add_hline(y=now_profit, line_dash="dot", line_color="gray", annotation_text=f"Sell now: {now_profit:,.0f}")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        # map date to month index starting at 1
        date_order = sorted(df_future["date"].unique())
        idx_map = {d: i + 1 for i, d in enumerate(date_order)}
        df_future["month_idx"] = df_future["date"].map(idx_map)
        frames = []
        for k in range(1, len(date_order) + 1):
            frames.append(df_future[df_future["month_idx"] <= k].assign(frame=k))
        df_anim = pd.concat(frames, ignore_index=True)
        fig3 = px.line(
            df_anim,
            x="date",
            y="profit",
            color="scenario",
            animation_frame="frame",
            markers=True,
            title="Profit by month — animating months",
        )
        fig3.update_yaxes(title_text="Profit")
        fig3.add_hline(y=now_profit, line_dash="dot", line_color="gray", annotation_text=f"Sell now: {now_profit:,.0f}")
        if sell_month is not None:
            fig3.add_vline(x=pd.to_datetime(sell_month), line_dash="dash", line_color="crimson")
        st.plotly_chart(fig3, use_container_width=True)

    # Download scenarios CSV
    csv_bytes = tab_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download profit scenarios (CSV)", data=csv_bytes, file_name="agri_profit_scenarios.csv", mime="text/csv")

    if sell_month is not None:
        dd = tab_df[(tab_df["when"] == "Future") & (tab_df["scenario"] == sell_scenario) & (tab_df["date"] == sell_month)]
        if len(dd):
            target_profit = float(dd.iloc[0]["profit"])
            delta = target_profit - now_profit
            st.metric("Planned month profit (selected)", f"{target_profit:,.0f}", delta=f"{delta:,.0f}")

with tabs[2]:
    st.subheader("Accounting Overview (TDHP)")
    # Load or example trial balance
    try:
        tb = load_trial_balance_csv(up_tb)
    except Exception as e:
        st.error(f"Could not read trial balance: {e}")
        tb = None

    if tb is None or len(tb) == 0:
        st.info("Using example trial balance. Upload your CSV for real data.")
        tb = load_trial_balance_csv(None)

    # Balance sheet
    bs = balance_sheet(tb)
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Assets", f"{bs['total_assets']:,.0f}")
    colB.metric("Liabilities", f"{(bs['liabilities_short']+bs['liabilities_long']):,.0f}")
    colC.metric("Equity", f"{bs['equity']:,.0f}")
    colD.metric("Assets - (Liab+Equity)", f"{bs['gap']:,.0f}")

    fig_bs = go.Figure()
    fig_bs.add_trace(go.Bar(name="Assets", x=["Current", "Non-Current"], y=[bs["assets_current"], bs["assets_noncurrent"]], marker_color="#2a9d8f"))
    fig_bs.add_trace(go.Bar(name="Liabilities", x=["Short", "Long"], y=[bs["liabilities_short"], bs["liabilities_long"]], marker_color="#e76f51"))
    fig_bs.update_layout(barmode="group", title="Balance Sheet Breakdown")
    st.plotly_chart(fig_bs, use_container_width=True)

    with st.expander("Details (groups)"):
        c1, c2 = st.columns(2)
        c1.write("Assets – Current")
        c1.dataframe(bs["groups"]["assets_current"].head(20))
        c1.write("Assets – Non-Current")
        c1.dataframe(bs["groups"]["assets_noncurrent"].head(20))
        c2.write("Liabilities – Short")
        c2.dataframe(bs["groups"]["liabilities_short"].head(20))
        c2.write("Liabilities – Long")
        c2.dataframe(bs["groups"]["liabilities_long"].head(20))
        c2.write("Equity")
        c2.dataframe(bs["groups"]["equity"].head(20))

    # Income statement
    is_ = income_statement(tb)
    cI1, cI2, cI3 = st.columns(3)
    cI1.metric("Revenue (6xx)", f"{is_['revenue']:,.0f}")
    cI2.metric("Expenses (7xx)", f"{is_['expense']:,.0f}")
    cI3.metric("Net Income", f"{is_['net']:,.0f}")

    fig_is = go.Figure(go.Waterfall(
        name="Income",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["Revenue", "- Expenses", "Net"],
        textposition="outside",
        y=[is_["revenue"], -is_["expense"], is_["net"]],
        connector={"line": {"color": "#2c3e50"}},
    ))
    fig_is.update_layout(title="Income Statement (simplified)")
    st.plotly_chart(fig_is, use_container_width=True)

    st.markdown("Budget Forecast (monthly)")
    mon = monthly_income_expense(tb)
    if len(mon):
        horizon_mon = st.slider("Forecast horizon (months)", 1, 12, min(6, max(3, 12)), 1, key="acct_horizon")
        # Use existing SARIMAX helper by adapting columns
        inc_df = mon[["date", "revenue"]].rename(columns={"revenue": "price"})
        exp_df = mon[["date", "expense"]].rename(columns={"expense": "price"})
        inc_fc = sarimax_forecast(inc_df, steps=horizon_mon)["forecast"].rename(columns={"yhat": "inc_yhat", "yhat_lo": "inc_lo", "yhat_hi": "inc_hi"})
        exp_fc = sarimax_forecast(exp_df, steps=horizon_mon)["forecast"].rename(columns={"yhat": "exp_yhat", "yhat_lo": "exp_lo", "yhat_hi": "exp_hi"})

        st.dataframe(mon.tail(12))
        fig_mon = go.Figure()
        fig_mon.add_trace(go.Scatter(x=mon["date"], y=mon["revenue"], name="Revenue", mode="lines+markers", line=dict(color="#2a9d8f")))
        fig_mon.add_trace(go.Scatter(x=mon["date"], y=mon["expense"], name="Expense", mode="lines+markers", line=dict(color="#e76f51")))
        fig_mon.add_trace(go.Scatter(x=inc_fc["date"], y=inc_fc["inc_yhat"], name="Rev forecast", mode="lines", line=dict(color="#2a9d8f", dash="dash")))
        fig_mon.add_trace(go.Scatter(x=exp_fc["date"], y=exp_fc["exp_yhat"], name="Exp forecast", mode="lines", line=dict(color="#e76f51", dash="dash")))
        st.plotly_chart(fig_mon, use_container_width=True)

        # Simple recommendation
        next_net = (inc_fc["inc_yhat"].values - exp_fc["exp_yhat"].values).mean()
        trend_rev = (inc_fc["inc_yhat"].mean() - mon["revenue"].tail(6).mean()) / (mon["revenue"].tail(6).mean() + 1e-6)
        trend_exp = (exp_fc["exp_yhat"].mean() - mon["expense"].tail(6).mean()) / (mon["expense"].tail(6).mean() + 1e-6)
        if next_net < 0 or trend_exp > trend_rev + 0.05:
            rec = "Expenses are trending above revenues. Consider cost controls, renegotiating inputs, or delaying discretionary capex."
        else:
            rec = "Revenues expected to cover expenses. Consider allocating surplus to working capital or prudent growth investments."
        st.info(rec)
    else:
        st.warning("No 'date' column in trial balance. Monthly budget forecast unavailable. Include a date per entry to enable it.")

with tabs[3]:
    st.subheader("Monte Carlo: Best-Month Profit Distribution")
    mc = simulate_profit(blend, params, residuals=bt_q.get("residuals"), n=3000)
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected best-month profit", f"{mc['mean']:,.0f}")
    col2.metric("Probability of loss", f"{mc['p_loss']*100:.1f}%")
    col3.metric("VaR 5%", f"{mc['var5']:,.0f}")

    # Rule-of-thumb advice
    best_future = tab_df[tab_df["when"] == "Future"]["profit"].max() if len(tab_df) else now_profit
    diff = best_future - now_profit
    rel = diff / (abs(now_profit) + 1e-6)
    if diff > 0 and rel > 0.05:
        advice = "Consider waiting to sell: projected best-month profit is >5% above selling now."
    elif diff < 0 and abs(rel) > 0.05:
        advice = "Consider selling now: risk-adjusted future scenarios are >5% worse than current."
    else:
        advice = "Differences are small (<5%). Operational factors may dominate the decision."

    st.info(advice)

    # Distribution chart
    if "dist" in mc and len(mc["dist"]) > 0:
        hist_df = pd.DataFrame({"profit": mc["dist"]})
        h = px.histogram(hist_df, x="profit", nbins=40, title="Distribution of best-month profit (Monte Carlo)")
        st.plotly_chart(h, use_container_width=True)

with tabs[4]:
    st.subheader("Forecast Quality (rolling backtest)")
    st.write(
        {
            "MAE (QGBR)": round(bt_q["mae"], 2) if not math.isnan(bt_q["mae"]) else None,
            "MAE (SARIMAX)": round(bt_s["mae"], 2) if not math.isnan(bt_s["mae"]) else None,
            "sMAPE% (QGBR)": round(bt_q["smape"], 2) if not math.isnan(bt_q["smape"]) else None,
            "Conformal radius": round(float(radius), 2),
        }
    )
    with st.expander("How to use"):
        st.markdown(
            "- Upload your CSVs or use the built-in synthetic defaults.\n"
            "- Adjust forecast horizon and interval coverage.\n"
            "- Choose blending mode: auto (based on backtest MAE), equal, or manual.\n"
            "- Review Profit Scenarios and pick a sell month to compare vs. selling now.\n"
            "- Inspect Risk to understand distribution of best-month outcomes."
        )
