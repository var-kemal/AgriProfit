from __future__ import annotations

from typing import Dict, List, Optional


def generate_recommendations(
    *,
    market: Dict,
    portfolio: Optional[Dict] = None,
    costs: Optional[Dict] = None,
) -> List[str]:
    """Produce human-readable bullet recommendations from signals.

    - market: output of signals/market.compute_market_signals
    - portfolio: output of portfolio.compute_portfolio_metrics (the dict of DataFrames)
    - costs: output of costs.optimizer.compute_input_cost_signals
    """
    bullets: List[str] = []

    # Market timing
    m = market.get("metrics", {}) if market else {}
    pctl = m.get("percentile_36m")
    mom3 = m.get("momentum_3m")
    if m.get("sell_zone"):
        bullets.append("Price is in the top 15% of the last 3 years with positive momentum and MA alignment — consider selling a tranche.")
    elif m.get("caution_zone"):
        bullets.append("Price is in the bottom 15% — avoid forced sales; review cash buffers or hedges.")
    elif pctl is not None and mom3 is not None:
        direction = "improving" if mom3 > 0 else "softening"
        bullets.append(f"Three-month momentum is {direction}; current 36m percentile ≈ {pctl:.2f}.")

    # Portfolio
    if portfolio:
        insights = portfolio.get("insights", {})
        top_ret = insights.get("top_return")
        lowest_vol = insights.get("lowest_vol")
        bullets.append(
            f"Portfolio est. return ≈ {insights.get('portfolio_return', 0):.2%}, vol ≈ {insights.get('portfolio_vol', 0):.2%}."
        )
        if top_ret and lowest_vol and top_ret != lowest_vol:
            bullets.append(
                f"Consider tilting toward {lowest_vol} for stability; {top_ret} currently leads on annualized return."
            )
        inv = insights.get("inverse_vol_weights", {})
        if inv:
            bullets.append("Inverse-volatility weights (suggested baseline): " + ", ".join(f"{k}:{v:.0%}" for k, v in inv.items()))

    # Costs
    if costs and isinstance(costs.get("summary"), list):
        hot = [row for row in costs["summary"] if row.get("recommendation") == "Buy now"]
        delay = [row for row in costs["summary"] if row.get("recommendation") == "Delay purchase"]
        if hot:
            names = ", ".join(sorted({r["input"] for r in hot}))
            bullets.append(f"Input costs favorable — consider buying: {names}.")
        if delay:
            names = ", ".join(sorted({r["input"] for r in delay}))
            bullets.append(f"Input costs elevated — consider delaying: {names}.")

    if not bullets:
        bullets.append("No strong signals detected — maintain baseline plan and monitor weekly.")
    return bullets

