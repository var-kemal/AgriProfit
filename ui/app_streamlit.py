import io
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.loaders import load_price_csv, load_costs_csv, load_yields_csv
from decision.profit import FarmParams, profit
from decision.montecarlo import simulate_profit
from models.forecast_sarimax import sarimax_forecast
from models.forecast_quantile_gbr import quantile_gbr_forecast
from forecast.ensemble import blend_forecasts
from analysis.backtest import rolling_backtest
from uncertainty.conformal import conformal_interval, apply_conformal

st.set_page_config(page_title="AgriProfit v2", layout="wide")
st.title("AgriProfit v2 — Прогноз цены, риск и прибыль")
st.caption("Модульная версия с бэктестом, ансамблем и калиброванными интервалами")

with st.sidebar:
    st.header("Данные")
    up_price = st.file_uploader("Цены (CSV: date,price)", type=["csv"], key="price")
    up_costs = st.file_uploader("Затраты (CSV: item,amount)", type=["csv"], key="costs")
    up_yield = st.file_uploader("Урожайность (CSV: year,yield_per_ha)", type=["csv"], key="yield")

    st.header("Хозяйство")
    area = st.number_input("Площадь, га", 0.5, 500.0, 10.0, 0.5)
    y_manual = st.checkbox("Ввести урожайность вручную", value=False)
    y_t_ha = st.number_input("Урожайность, т/га", 0.5, 30.0, 5.0, 0.1) if y_manual else None
    price_unit = st.selectbox("Единица цены", ["per_kg", "per_ton"], index=1)

    st.header("Прогноз")
    horizon = st.slider("Горизонт (месяцы)", 1, 12, 3, 1)

# Load data
try:
    prices = load_price_csv(up_price)
    costs = load_costs_csv(up_costs)
    yields_df = load_yields_csv(up_yield)
except Exception as e:
    st.error(f"Ошибка данных: {e}")
    st.stop()

if y_t_ha is None:
    y_t_ha = float(yields_df.sort_values("year").iloc[-1]["yield_per_ha"]) if len(yields_df) else 5.0

params = FarmParams(area_ha=area, yield_t_per_ha=y_t_ha, price_unit=price_unit, costs_total=float(costs["amount"].sum()*area))

# Backtest helper wrappers
fit_sarimax = lambda df, steps: sarimax_forecast(df, steps)
fit_qgbr = lambda df, steps: quantile_gbr_forecast(df, steps)

# Rolling backtest for conformal radius (using QGBR median as point)
bt = rolling_backtest(prices, fit_qgbr, horizon=3, min_train=min(24, max(12, len(prices)//2)))
radius = conformal_interval(bt["residuals"], alpha=0.10) if len(prices) > 18 else 0.0

# Fit two models on full data
m1 = fit_sarimax(prices, horizon)
m2 = fit_qgbr(prices, horizon)

# Blend forecasts (equal weights)
blend = blend_forecasts(m1, m2)
# Apply conformal adjustment around blended median (yhat)
blend = apply_conformal(blend, radius)

# Price chart
st.subheader("Цена: история и прогноз (интерактивный график)")

import plotly.express as px

fig = px.line(prices, x="date", y="price", title="Исторические цены")
fig.add_scatter(x=blend["date"], y=blend["yhat"], mode="lines", name="Прогноз (P50)")
fig.add_scatter(x=blend["date"], y=blend["yhat_lo"], mode="lines", name="P10", line=dict(dash="dot"))
fig.add_scatter(x=blend["date"], y=blend["yhat_hi"], mode="lines", name="P90", line=dict(dash="dot"))

st.plotly_chart(fig, use_container_width=True)


# Profit table (sell now vs future months, choose best month)
last_price = float(prices.dropna().iloc[-1]["price"]) if len(prices) else float('nan')
now_profit = profit(last_price, params)

rows = [{"when":"Sell now", "date": pd.to_datetime(prices.iloc[-1]["date"]).date() if len(prices) else None, "scenario":"Current", "ref_price": last_price, "profit": now_profit}]
for _, r in blend.iterrows():
    for name, p in [("P10", r["yhat_lo"]), ("P50", r["yhat"]), ("P90", r["yhat_hi"])]:
        rows.append({"when":"Future", "date": pd.to_datetime(r["date"]).date(), "scenario": name, "ref_price": float(r["yhat"]), "profit": profit(float(p), params)})

tab = pd.DataFrame(rows)

st.subheader("Прибыль по сценариям")
fig2, ax2 = plt.subplots(figsize=(8,4))
for sc in ["P10", "P50", "P90"]:
    d = tab[(tab["when"]=="Future") & (tab["scenario"]==sc)]
    ax2.plot(pd.to_datetime(d["date"]), d["profit"], label=sc)
ax2.axhline(now_profit, linestyle=":", label=f"Сейчас: {now_profit:,.0f}")
ax2.grid(True, alpha=0.3); ax2.legend(); st.pyplot(fig2)

# Monte-Carlo choose-best-month distribution
mc = simulate_profit(blend, params, n=3000)

st.subheader("Риск-профиль (Монте-Карло, лучший месяц)")
col1, col2, col3 = st.columns(3)
col1.metric("Ожидаемая прибыль", f"{mc['mean']:,.0f}")
col2.metric("Риск убытка", f"{mc['p_loss']*100:.1f}%")
col3.metric("VaR 5%", f"{mc['var5']:,.0f}")

# Recommendation (rule-of-thumb)
best_future = tab[tab["when"]=="Future"]["profit"].max() if len(tab) else now_profit
diff = best_future - now_profit
rel = diff/(abs(now_profit)+1e-6)
if diff > 0 and rel > 0.05:
    advice = "Рекомендуем подождать: ожидаемая лучшая прибыль выше >5% vs продажа сейчас."
elif diff < 0 and abs(rel) > 0.05:
    advice = "Лучше продавать сейчас: ожидание с большой вероятностью снижает прибыль."
else:
    advice = "Разница небольшая (<5%). Решайте по логистике/риску."


st.subheader("Рекомендация")
st.info(advice)

# Export
csv_bytes = tab.to_csv(index=False).encode("utf-8")
st.download_button("Скачать сценарии (CSV)", data=csv_bytes, file_name="agri_profit_scenarios.csv", mime="text/csv")

# Passport (quality)
st.subheader("Паспорт прогноза (качество)")
st.write({"MAE": round(bt["mae"],2) if not math.isnan(bt["mae"]) else None, "sMAPE%": round(bt["smape"],2) if not math.isnan(bt["smape"]) else None, "Conformal radius": round(float(radius),2)})


