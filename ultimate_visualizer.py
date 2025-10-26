from __future__ import annotations

import io
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ultimate_time_series_analysis import UltimateTimeSeriesAnalyzer
from analysis.eda import rolling_stats, acf_pacf_values


def integrate_with_agriprofit(prices_df: pd.DataFrame) -> Dict:
    """Run the ultimate analyzer and prepare a compact dashboard figure.

    Returns a dict with:
      - figure: Matplotlib figure handle
      - results: analysis results dict
    Also saves PNG and a simple PDF report to the repo root for easy downloads.
    """
    analyzer = UltimateTimeSeriesAnalyzer(prices_df)
    results = analyzer.run_ultimate_analysis()

    # Build a compact figure: time series + trend, ACF, seasonal strength summary
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 1) Time series with rolling means
    rs = rolling_stats(prices_df)
    ax1.plot(rs["date"], rs["price"], color="#2563eb", lw=1.5, label="price")
    for w, col in [(3, "ma_3"), (6, "ma_6"), (12, "ma_6")]:
        if col in rs.columns:
            ax1.plot(rs["date"], rs[col], lw=1, alpha=0.7, label=col)
    ax1.set_title("Price with rolling means")
    ax1.legend(loc="upper left", fontsize=8)

    # 2) Trend line
    s = analyzer.series.dropna()
    x = np.arange(len(s))
    t = results["trend_advanced"]
    if t.get("slope") is not None:
        y_pred = t["slope"] * x + t["intercept"]
        ax2.plot(s.index, s.values, color="#94a3b8", lw=1)
        ax2.plot(s.index, y_pred, color="#10b981", lw=2)
        ax2.set_title(f"Trend: slope={t['slope']:.3f}, R2={t['r2']:.2f}")
    else:
        ax2.text(0.5, 0.5, "Trend unavailable (too short)", ha="center", va="center")
        ax2.set_axis_off()

    # 3) ACF
    ap = acf_pacf_values(prices_df)
    acf_vals = ap.get("acf", [])
    if acf_vals:
        ax3.bar(range(len(acf_vals)), acf_vals, color="#2563eb")
        ax3.set_title("ACF")
    else:
        ax3.text(0.5, 0.5, "ACF not available", ha="center", va="center")
        ax3.set_axis_off()

    # 4) Spectral dominant periods
    sp = results.get("spectral", {}).get("dominant_periods", [])
    if sp:
        periods = [d["period_months"] for d in sp]
        power = [d["power"] for d in sp]
        ax4.bar(periods, power, color="#f59e0b")
        ax4.set_xlabel("Period (months)")
        ax4.set_title("Dominant spectral periods")
    else:
        ax4.text(0.5, 0.5, "No strong periods detected", ha="center", va="center")
        ax4.set_axis_off()

    fig.tight_layout()

    # Save artifacts for download
    png_path = "agriprofit_ultimate_analysis.png"
    pdf_path = "agriprofit_ultimate_report.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    return {"figure": fig, "results": results}


def create_streamlit_tab(prices_df: pd.DataFrame):  # pragma: no cover (UI side)
    import streamlit as st

    st.subheader("🔬 Ultimate Time Series Analysis")
    colA, colB = st.columns([2, 1])
    with st.spinner("Running analysis..."):
        out = integrate_with_agriprofit(prices_df)
    with colA:
        st.pyplot(out["figure"])
    with colB:
        t = out["results"].get("trend_advanced", {})
        s = out["results"].get("stationarity_advanced", {})
        h = out["results"].get("nonlinear", {}).get("hurst_exponent", {})
        st.write({
            "Trend slope": round(t.get("slope"), 4) if t.get("slope") is not None else None,
            "Trend R2": round(t.get("r2"), 3) if t.get("r2") is not None else None,
            "ADF p": s.get("adf_p"),
            "KPSS p": s.get("kpss_p"),
            "Hurst (RS)": h.get("rs_hurst"),
        })

    # Download buttons (use files saved by integrate_with_agriprofit)
    with open("agriprofit_ultimate_analysis.png", "rb") as f_png:
        st.download_button("Download Dashboard (PNG)", data=f_png, file_name="ultimate_analysis.png", mime="image/png")
    with open("agriprofit_ultimate_report.pdf", "rb") as f_pdf:
        st.download_button("Download Report (PDF)", data=f_pdf, file_name="ultimate_report.pdf", mime="application/pdf")

