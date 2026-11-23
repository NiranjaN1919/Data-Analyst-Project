"""Builds 04_Dashboard/financial_dashboard.html (single self-contained file) from the analysis outputs.
Usage: python3 build_dashboard.py <project_root>"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
D1, D3 = os.path.join(ROOT, "01_Data"), os.path.join(ROOT, "03_Models")
R = json.load(open(os.path.join(D3, "results.json")))
m = pd.read_csv(os.path.join(D1, "sp500_monthly_prices.csv"))
months = sorted(m.month.unique())
px = m.pivot(index="month", columns="ticker", values="adj_close").reindex(months)
tv = m.pivot(index="month", columns="ticker", values="traded_value_musd").reindex(months)
tick = sorted(px.columns)
st = pd.read_csv(os.path.join(D1, "ticker_summary.csv"), index_col=0)
rank = st.avg_dollar_vol_musd.rank(ascending=False, method="first").astype(int)
mk = pd.read_csv(os.path.join(D1, "market_index_daily.csv"), index_col=0, parse_dates=True)
mkt_m = mk.index_level.resample("ME").last()
mvol = mk.daily_return.resample("ME").std() * np.sqrt(252)
mkt_m.index = mkt_m.index.strftime("%Y-%m"); mvol.index = mvol.index.strftime("%Y-%m")
assert list(mkt_m.index) == months, "month mismatch"
nn = lambda a, d: [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), d) for v in a]
ft = pd.read_csv(os.path.join(D1, "forecast_table.csv"))
fc = {}
for name, g in ft.groupby("series"):
    a = g[g.model == "ARIMA"].sort_values("month")
    h = g[g.model == "Holt (damped)"].sort_values("month")
    fc[name] = {"arima": nn(a.forecast, 4), "lo80": nn(a.lo80, 4), "hi80": nn(a.hi80, 4), "lo95": nn(a.lo95, 4), "hi95": nn(a.hi95, 4),
                "holt": nn(h.forecast, 4) if len(h) else None,
                "order": str(tuple(R["forecast"][name].get("arima_order", R["forecast"][name].get("order"))))}
fmonths = sorted(ft.month.unique())
bp = {(r["model"], r["horizon_months"]): r["rmse_vs_naive"] for r in R["bt"]["price"]}
bv = {(r["model"], r["horizon_months"]): r["rmse_vs_naive"] for r in R["bt"]["vol"]}
asof = pd.Timestamp(R["today"])
data = {"months": months, "tick": tick, "px": [nn(px[t], 3) for t in tick], "tv": [nn(tv[t].round(0), 0) for t in tick],
        "rank": [int(rank[t]) for t in tick], "focus": R["focus"], "mkt": nn(mkt_m, 3), "mvol": nn(mvol, 5), "fc": fc, "fmonths": fmonths,
        "asof_label": asof.strftime("%d %b %Y").lstrip("0"), "n_tick": len(tick),
        "bt": {"n": R["bt"]["n_origins"], "price12": bp[("ARIMA", 12)], "vol12": bv[("ARIMA", 12)]}}
html = open(os.path.join(HERE, "dashboard_template.html")).read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
os.makedirs(os.path.join(ROOT, "04_Dashboard"), exist_ok=True)
out = os.path.join(ROOT, "04_Dashboard", "financial_dashboard.html")
open(out, "w").write(html)
print("saved", out, f"{len(html) / 1e6:.2f} MB")
