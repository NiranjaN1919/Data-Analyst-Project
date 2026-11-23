"""
Project 3 - Financial Data Analysis and Forecasting (S&P 500 daily prices, 2015-2026)

Usage:  python3 financial_analysis.py <project_root> <path to sp500-stock-prices.csv.gz>
Needs:  pandas, numpy, scipy, matplotlib, seaborn  (+ ts_models.py in this folder; statsmodels optional, not required)

Cells (marked "# %%") run top to bottom and also work as a notebook:
  0 setup   1 cleaning   2 features, market index, monthly tables   3 trend analysis + charts
  4 forecast back-test   5 final forecasts + charts   6 save results
Outputs go to 01_Data, 02_Charts, 03_Models.  The back-test (cell 4) takes a few minutes; its result is cached
in 03_Models/forecast_backtest.csv (delete the file to recompute).
"""
# %% 0. Setup
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ts_models import ARIMA, HoltWinters, auto_arima, naive_forecast, drift_forecast  # noqa: E402

ROOT = sys.argv[1]
RAW = sys.argv[2]
D1, D2, D3 = (os.path.join(ROOT, d) for d in ("01_Data", "02_Charts", "03_Models"))
for d in (D1, D2, D3):
    os.makedirs(d, exist_ok=True)

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]      # fixed categorical order
INK, MUTED, GRID, SURF = "#2b2a28", "#6b6a66", "#e4e3df", "#fcfcfb"
MKT_COL = "#3d3c3a"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 100,
})
R = {}                       # everything the report needs goes here (saved as results.json)
LOG = []                     # data cleaning log


def save(fig, name):
    fig.savefig(os.path.join(D2, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def cleanlog(step, finding, action):
    LOG.append({"Step": step, "Finding": finding, "Action": action})


pct = lambda v, d=1: f"{v * 100:.{d}f}%"

# %% 1. Data cleaning
raw = pd.read_csv(RAW, parse_dates=["date"])
n0 = len(raw)
cleanlog("Load raw file", f"{n0:,} rows x {raw.shape[1]} columns ({raw.ticker.nunique()} tickers, {raw.date.nunique():,} trading days, "
         f"{raw.date.min():%Y-%m-%d} to {raw.date.max():%Y-%m-%d})", "None")
cleanlog("Variable types", "date read as text; prices decimal, volume integer", "date converted to datetime")
cleanlog("Missing values", f"{int(raw.isna().sum().sum())} missing cells in any column", "Nothing to impute")
d_key = int(raw.duplicated(["date", "ticker"]).sum())
cleanlog("Duplicates", f"{d_key} duplicated (date, ticker) rows", "None removed" if d_key == 0 else "Removed")
raw = raw.sort_values(["ticker", "date"]).drop_duplicates(["date", "ticker"]).reset_index(drop=True)
nonpos = int((raw[["open", "high", "low", "close", "adj_close"]] <= 0).sum().sum())
cleanlog("Non-positive prices", f"{nonpos} zero or negative prices", "None to fix")
bad_hl = int((raw.high < raw.low).sum())
bad_c = int(((raw.close > raw.high * 1.0001) | (raw.close < raw.low * 0.9999)).sum())
bad_o = (raw.open > raw.high * 1.0001) | (raw.open < raw.low * 0.9999)
cleanlog("Price consistency (OHLC)", f"{bad_hl} rows with high < low; {bad_c} with close outside the day's range; {int(bad_o.sum())} with open outside the range",
         "Open clipped into [low, high]" if bad_o.sum() else "None")
raw.loc[bad_o, "open"] = raw.loc[bad_o, "open"].clip(raw.loc[bad_o, "low"], raw.loc[bad_o, "high"])
ratio = raw.adj_close / raw.close
cleanlog("Adjusted close", f"adj_close/close ranges {ratio.min():.2f}-{ratio.max():.2f} (never above 1) and moves only at dividend dates",
         "adj_close used for returns and indices (dividends reinvested); close and volume used for traded value")

# placeholder history: a few tickers carry stale prices with zero volume until their US listing date
zc = (raw.volume == 0).groupby(raw.ticker).sum()
heavy = zc[zc > 100].index.tolist()
cut = {t: raw[(raw.ticker == t) & (raw.volume == 0)].date.max() for t in heavy}
lead = pd.Series(False, index=raw.index)
for t, d in cut.items():
    lead |= (raw.ticker == t) & (raw.date <= d)
cleanlog("Zero volume (pre-listing)", f"{int(lead.sum()):,} rows on {len(heavy)} tickers have zero volume and stale prices up to the US listing date "
         f"({', '.join(f'{t} to {d:%Y-%m-%d}' for t, d in cut.items())})",
         "Removed: these prices come from before the stock traded in the US, so returns and volume would be wrong")
raw = raw[~lead].copy()
zv = raw.volume == 0
cleanlog("Zero volume (other)", f"{int(zv.sum())} remaining zero-volume rows on {raw[zv].ticker.nunique()} tickers", "Volume set to missing; prices kept")
raw.loc[zv, "volume"] = np.nan

adj = raw.pivot(index="date", columns="ticker", values="adj_close")
close = raw.pivot(index="date", columns="ticker", values="close")
vol = raw.pivot(index="date", columns="ticker", values="volume")
dvol = close * vol                                              # traded value, USD
ret = adj.pct_change()
first = adj.apply(lambda c: c.first_valid_index())
last = adj.apply(lambda c: c.last_valid_index())
inner = sum(int(adj[t].loc[first[t]:last[t]].isna().sum()) for t in adj.columns)
cleanlog("Gaps inside a history", f"{inner} missing trading days inside any ticker's history; every ticker ends on {adj.index[-1]:%Y-%m-%d}", "None to fill")
ndays = adj.notna().sum()
cleanlog("Short histories", f"{int((ndays < 252).sum())} tickers have under one year of data and {int((ndays < 2900).sum())} have less than the full period (new listings)",
         "Kept in the data; excluded from performance rankings when under 2 years")
big = ret.abs() > 0.30
rev = big & ((adj.shift(-1) / adj.shift(1) - 1).abs() < 0.05)
cleanlog("Extreme daily moves", f"{int((ret.abs() > 0.4).sum().sum())} daily moves above 40% and {int(rev.sum().sum())} one-day spike(s) that reverse the next day",
         "Large moves that persist are kept (real events such as March 2020 or earnings); a spike that reverses next day is treated as a data error and repaired")
spk = rev.stack()
spk = spk[spk].index.tolist()
for dt, tk in spk:
    adj.loc[dt, tk] = np.nan
adj = adj.interpolate(limit=1, limit_area="inside")             # one-day repair only
ret = adj.pct_change()
cleanlog("Variable transformation", "prices are levels; analysis needs returns, an index, monthly series",
         "Built daily returns, equal-weight market index, month-end prices, monthly returns and traded value")
raw_n = len(raw)
R["cleaning"] = {"rows_raw": n0, "rows_clean": raw_n, "tickers": int(adj.shape[1]), "days": int(adj.shape[0]),
                 "date_min": f"{adj.index.min():%Y-%m-%d}", "date_max": f"{adj.index.max():%Y-%m-%d}",
                 "lead_rows": int(lead.sum()), "spikes_fixed": [f"{d:%Y-%m-%d} {t}" for d, t in spk]}
pd.DataFrame(LOG).to_csv(os.path.join(D1, "data_cleaning_log.csv"), index=False)

# data-quality chart: coverage of tickers over time + short-history tickers
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
cov = adj.notna().sum(axis=1)
ax[0].fill_between(cov.index, cov.values, color=PAL[0], alpha=0.25)
ax[0].plot(cov.index, cov.values, color=PAL[0], lw=1.6)
ax[0].set_ylim(400, 520)
ax[0].set_title("Stocks with data on each day")
ax[0].set_ylabel("number of tickers")
ax[0].text(cov.index[40], 508, "503 tickers today", color=MUTED, fontsize=8.5)
yrs = (ndays / 252)
ax[1].hist(yrs, bins=[0, 1, 2, 3, 5, 8, 10, 12], color=PAL[0], edgecolor=SURF, linewidth=1.5)
ax[1].set_title("Years of history per stock")
ax[1].set_xlabel("years")
ax[1].set_ylabel("tickers")
fig.tight_layout()
save(fig, "01_data_coverage.png")

# %% 2. Features: market index, per-stock statistics, monthly tables
TD = 252
mkt_ret = ret.mean(axis=1)                                      # equal-weight, daily rebalanced
mkt_ret.iloc[0] = 0.0
mkt = 100 * (1 + mkt_ret).cumprod()
mkt_dd = mkt / mkt.cummax() - 1
mkt_vol60 = mkt_ret.rolling(60).std() * np.sqrt(TD)
n_last = (adj.index[-1] - adj.index[0]).days / 365.25

# per-stock summary
dd_all = adj / adj.cummax() - 1
mkt_var = mkt_ret.var()
rows = []
for t in adj.columns:
    s = adj[t].dropna()
    r_ = ret[t].dropna()
    yrs_ = (s.index[-1] - s.index[0]).days / 365.25
    ann_vol = r_.std() * np.sqrt(TD)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs_) - 1 if yrs_ > 0.5 else np.nan
    both = pd.concat([r_, mkt_ret], axis=1, join="inner").dropna()
    rows.append({"ticker": t, "first_date": s.index[0].date(), "n_days": len(s), "years": round(yrs_, 2), "last_price": s.iloc[-1],
                 "total_return": s.iloc[-1] / s.iloc[0] - 1, "cagr": cagr, "ann_vol": ann_vol,
                 "sharpe": (r_.mean() * TD) / ann_vol if ann_vol > 0 else np.nan, "max_drawdown": dd_all[t].min(),
                 "beta": both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var() if len(both) > 60 else np.nan,
                 "ret_1y": s.iloc[-1] / s.iloc[max(-TD - 1, -len(s))] - 1 if len(s) > TD else np.nan,
                 "avg_dollar_vol_musd": dvol[t].iloc[-TD:].mean() / 1e6})
stats_df = pd.DataFrame(rows).set_index("ticker")
stats_df["history_ok"] = stats_df.years >= 2
stats_df["full_history"] = stats_df.n_days >= 2900
focus_pool = stats_df[stats_df.full_history].avg_dollar_vol_musd.sort_values(ascending=False)
FOCUS = focus_pool.index[:5].tolist()
PRIMARY = FOCUS[0]
stats_df["focus"] = stats_df.index.isin(FOCUS)
stats_df.round(4).to_csv(os.path.join(D1, "ticker_summary.csv"))
FCOL = dict(zip(FOCUS, PAL))

# monthly tables
adjm = adj.resample("ME").last()
retm = adjm.pct_change()
dvm = dvol.resample("ME").sum(min_count=1) / 1e6
mkt_m = mkt.resample("ME").last()
mkt_mret = mkt_m.pct_change()
mkt_mvol = mkt_ret.resample("ME").std() * np.sqrt(TD)
LAST_FULL = pd.Timestamp("2026-10-31")                          # November 2026 is a partial month
long = adjm.stack().rename("adj_close").rename_axis(["month", "ticker"]).to_frame()
long = long.join(retm.stack().rename("monthly_return").rename_axis(["month", "ticker"]))
long = long.join(dvm.stack().rename("traded_value_musd").rename_axis(["month", "ticker"])).reset_index()
long.columns = ["month", "ticker", "adj_close", "monthly_return", "traded_value_musd"]
long = long.dropna(subset=["adj_close"])
long["month"] = long.month.dt.strftime("%Y-%m")
long.round({"adj_close": 4, "monthly_return": 5, "traded_value_musd": 3}).to_csv(os.path.join(D1, "sp500_monthly_prices.csv"), index=False)

# daily files for the market and the focus stocks
sma = lambda s, n: s.rolling(n).mean()
mdf = pd.DataFrame({"index_level": mkt, "daily_return": mkt_ret, "drawdown": mkt_dd, "vol_60d_annualised": mkt_vol60,
                    "stocks_up_share": (ret > 0).sum(axis=1) / ret.notna().sum(axis=1)})
mdf.round(5).to_csv(os.path.join(D1, "market_index_daily.csv"))
fd = []
for t in FOCUS:
    x = raw[raw.ticker == t].set_index("date")[["open", "high", "low", "close", "adj_close", "volume"]].copy()
    x["daily_return"] = ret[t].reindex(x.index)
    x["sma_50"] = sma(x.adj_close, 50)
    x["sma_200"] = sma(x.adj_close, 200)
    x["drawdown"] = dd_all[t].reindex(x.index)
    x.insert(0, "ticker", t)
    fd.append(x.reset_index())
pd.concat(fd).round(4).to_csv(os.path.join(D1, "focus_stocks_daily.csv"), index=False)

# calendar-year table
yr_ret = lambda s: s.resample("YE").last().pct_change()
yearly = pd.DataFrame({"Market (equal-weight)": mkt.resample("YE").last().pct_change()})
first_year = mkt.iloc[0] / 100 - 1
yearly.iloc[0, 0] = mkt.resample("YE").last().iloc[0] / 100 - 1
for t in FOCUS:
    a = adj[t].resample("YE").last()
    a0 = adj[t].dropna().iloc[0]
    y = a.pct_change()
    y.iloc[0] = a.iloc[0] / a0 - 1
    yearly[t] = y
yearly.index = yearly.index.year
yearly.round(4).to_csv(os.path.join(D1, "yearly_returns.csv"), index_label="year")

R["market"] = {
    "total_return": float(mkt.iloc[-1] / 100 - 1), "cagr": float((mkt.iloc[-1] / 100) ** (1 / n_last) - 1),
    "ann_vol": float(mkt_ret.std() * np.sqrt(TD)), "sharpe": float(mkt_ret.mean() * TD / (mkt_ret.std() * np.sqrt(TD))),
    "max_dd": float(mkt_dd.min()), "max_dd_date": f"{mkt_dd.idxmin():%Y-%m-%d}", "last": float(mkt.iloc[-1]),
    "ytd": float(mkt.iloc[-1] / mkt[:f"{adj.index[-1].year - 1}-12-31"].iloc[-1] - 1),
    "ret_1y": float(mkt.iloc[-1] / mkt.iloc[-TD - 1] - 1), "years": float(n_last)}
pk = mkt[:mkt_dd.idxmin()].idxmax()
R["market"]["max_dd_peak"] = f"{pk:%Y-%m-%d}"
R["market"]["dd_recovery_days"] = int((mkt[mkt_dd.idxmin():][mkt[mkt_dd.idxmin():] >= mkt[pk]].index[0] - mkt_dd.idxmin()).days) if (mkt[mkt_dd.idxmin():] >= mkt[pk]).any() else None
R["focus"] = FOCUS
R["primary"] = PRIMARY
R["yearly"] = yearly.round(4).to_dict()
R["today"] = f"{adj.index[-1]:%Y-%m-%d}"
R["partial_month"] = f"{adj.index[-1]:%B %Y}"

# %% 3. Trend analysis (EDA) and charts
ok = stats_df[stats_df.history_ok]
R["stocks_ok"] = int(len(ok))
R["stocks_up_share_total"] = float((ok.total_return > 0).mean())
R["stocks_beat_mkt"] = float((ok.cagr > R["market"]["cagr"]).mean())
R["median_cagr"] = float(ok.cagr.median())
R["median_vol"] = float(ok.ann_vol.median())
R["top_perf"] = ok.sort_values("total_return", ascending=False).head(8)[["total_return", "cagr", "ann_vol"]].round(3).reset_index().to_dict("records")
R["focus_stats"] = stats_df.loc[FOCUS, ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "beta", "ret_1y", "avg_dollar_vol_musd", "last_price"]].round(4).reset_index().to_dict("records")

# Chart 02 - market index and drawdown
fig, ax = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
ax[0].plot(mkt.index, mkt.values, color=MKT_COL, lw=1.8)
ax[0].plot(mkt.index, mkt.rolling(200).mean(), color=PAL[3], lw=1.2, label="200-day average")
ax[0].set_title("Equal-weight S&P 500 index (rebased to 100 on 2 Jan 2015, dividends reinvested)")
ax[0].set_ylabel("index level")
for lab, dt in (("COVID-19 crash", "2020-03-23"), ("2022 bear market", "2022-10-12")):
    d_ = mkt.index[mkt.index.get_indexer([pd.Timestamp(dt)], method="nearest")[0]]
    ax[0].annotate(lab, (d_, mkt[d_]), xytext=(0, 38), textcoords="offset points", fontsize=8.5, color=MUTED, ha="center",
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax[0].legend(loc="upper left")
ax[1].fill_between(mkt_dd.index, mkt_dd.values * 100, 0, color=PAL[1], alpha=0.35)
ax[1].plot(mkt_dd.index, mkt_dd.values * 100, color=PAL[1], lw=1)
ax[1].set_ylabel("drawdown (%)")
ax[1].set_title("Fall from the previous peak", fontsize=10)
fig.tight_layout()
save(fig, "02_market_index_drawdown.png")

# Chart 03 - focus stocks vs market (rebased, log scale)
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(mkt.index, mkt.values, color=MKT_COL, lw=2, ls="--", label="Market (equal-weight)")
for t in FOCUS:
    s = adj[t].dropna()
    ax.plot(s.index, 100 * s / s.iloc[0], color=FCOL[t], lw=1.5, label=t)
ends = sorted([(np.log(100 * adj[t].dropna().iloc[-1] / adj[t].dropna().iloc[0]), t) for t in FOCUS])
ylab, gap = [], 0.17                                              # spread end labels so they do not overlap
for v, t in ends:
    ylab.append((max(v, ylab[-1][0] + gap) if ylab else v, t))
for v, t in ylab:
    ax.text(adj.index[-1] + pd.Timedelta(days=25), np.exp(v), t, color=FCOL[t], fontsize=8.5, va="center")
ax.set_yscale("log")
ax.set_yticks([100, 200, 500, 1000, 2000, 5000])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_title("The five most-traded long-history stocks against the market (log scale, rebased to 100)")
ax.set_ylabel("value of 100 invested in Jan 2015")
ax.legend(loc="upper left", ncol=3)
ax.set_xlim(mkt.index[0], mkt.index[-1] + pd.Timedelta(days=170))
fig.tight_layout()
save(fig, "03_focus_vs_market.png")

# Chart 04 - calendar-year performance, market vs focus
yp = yearly.copy() * 100
fig, ax = plt.subplots(figsize=(11, 4.6))
cols = ["Market (equal-weight)"] + FOCUS
w = 0.8 / len(cols)
for i, c in enumerate(cols):
    ax.bar(np.arange(len(yp)) + (i - 2.5) * w, yp[c].values, w * 0.92, color=MKT_COL if i == 0 else PAL[i - 1], label=c)
ax.set_xticks(np.arange(len(yp)))
ax.set_xticklabels([f"{y}{' YTD' if y == adj.index[-1].year else ''}" for y in yp.index])
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylabel("calendar-year return (%)")
ax.set_title("Market performance comparison by year")
ax.legend(ncol=6, loc="upper left")
ax.set_ylim(top=max(yp.max().max() * 1.12, 20))
fig.tight_layout()
save(fig, "04_yearly_performance.png")

# Chart 05 - 60-day volatility of the market and stocks moving up together
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(mkt_vol60.index, mkt_vol60.values * 100, color=PAL[0], lw=1.5)
ax.set_title("Market volatility: 60-day annualised standard deviation of daily returns")
ax.set_ylabel("%")
ax.axhline(mkt_ret.std() * np.sqrt(TD) * 100, color=MUTED, lw=0.9, ls=":")
ax.text(mkt_vol60.index[70], mkt_ret.std() * np.sqrt(TD) * 100 + 1, "full-period average", color=MUTED, fontsize=8.5)
fig.tight_layout()
save(fig, "05_market_volatility.png")
R["vol"] = {"avg": float(mkt_ret.std() * np.sqrt(TD)), "max60": float(mkt_vol60.max()), "max60_date": f"{mkt_vol60.idxmax():%Y-%m-%d}",
            "now60": float(mkt_vol60.iloc[-1]), "median60": float(mkt_vol60.median())}

# Chart 06 - primary stock price with moving averages
p_ = adj[PRIMARY].dropna()
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(p_.index, p_.values, color=FCOL[PRIMARY], lw=1.3, label=f"{PRIMARY} adjusted close")
ax.plot(p_.index, sma(p_, 50), color=INK, lw=1, label="50-day average")
ax.plot(p_.index, sma(p_, 200), color=PAL[3], lw=1.3, label="200-day average")
ax.set_yscale("log")
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_yticks([5, 10, 20, 50, 100, 200])
ax.set_title(f"{PRIMARY}: price trend with 50- and 200-day moving averages (log scale)")
ax.set_ylabel("USD")
ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "06_primary_stock_trend.png")
above200 = {t: float((adj[t] > sma(adj[t], 200))[sma(adj[t], 200).notna()].mean()) for t in FOCUS}
R["above200"] = above200
R["mkt_above200"] = float((mkt > mkt.rolling(200).mean())[mkt.rolling(200).mean().notna()].mean())

# Chart 07 - monthly return distribution and calendar-month seasonality (market)
mr = mkt_mret.dropna()
mr_full = mr[mr.index <= LAST_FULL]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
ax[0].hist(mr_full * 100, bins=24, color=PAL[0], edgecolor=SURF, linewidth=1.2)
ax[0].axvline(mr_full.mean() * 100, color=INK, lw=1.2)
ax[0].set_title("Monthly market return distribution")
ax[0].set_xlabel("monthly return (%)")
ax[0].set_ylabel("months")
cm = mr_full.groupby(mr_full.index.month).agg(["mean", "std", "count"])
cm["se"] = cm["std"] / np.sqrt(cm["count"])
mn = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ax[1].bar(range(12), cm["mean"] * 100, yerr=cm["se"] * 100 * 1.96, color=[PAL[0] if v >= 0 else PAL[1] for v in cm["mean"]],
          error_kw=dict(ecolor=MUTED, lw=1, capsize=3), width=0.7)
ax[1].set_xticks(range(12))
ax[1].set_xticklabels(mn)
ax[1].axhline(0, color=INK, lw=0.8)
ax[1].set_title("Average return by calendar month (bars = 95% interval)")
ax[1].set_ylabel("%")
fig.tight_layout()
save(fig, "07_monthly_distribution_seasonality.png")
kw = stats.kruskal(*[g.values for _, g in mr_full.groupby(mr_full.index.month)])
R["monthly"] = {"mean": float(mr_full.mean()), "median": float(mr_full.median()), "sd": float(mr_full.std()), "pos_share": float((mr_full > 0).mean()),
                "best": float(mr_full.max()), "best_month": f"{mr_full.idxmax():%b %Y}", "worst": float(mr_full.min()), "worst_month": f"{mr_full.idxmin():%b %Y}",
                "skew": float(stats.skew(mr_full)), "n": int(len(mr_full)), "kruskal_p": float(kw.pvalue),
                "best_cal": mn[int(cm["mean"].idxmax()) - 1], "best_cal_mean": float(cm["mean"].max()),
                "worst_cal": mn[int(cm["mean"].idxmin()) - 1], "worst_cal_mean": float(cm["mean"].min()),
                "cal_sig": [mn[int(m) - 1] for m in cm.index if abs(cm.loc[m, "mean"]) > 1.96 * cm.loc[m, "se"]]}

# Chart 08 - traded value: concentration and trend
dv_last = dvol.iloc[-TD:].mean().sort_values(ascending=False)
top10 = dv_last.head(10) / dv_last.sum()
tot_m = dvm.sum(axis=1)[:LAST_FULL] / 1e6                         # USD trillions per month
fig, ax = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1.1, 1.3]})
ax[0].barh(top10.index[::-1], top10.values[::-1] * 100, color=PAL[0], height=0.7)
for i, v in enumerate(top10.values[::-1] * 100):
    ax[0].text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=8.5, color=MUTED)
ax[0].set_title("Share of all traded value, last 12 months")
ax[0].set_xlabel("% of total traded value (top 10 stocks)")
ax[0].grid(axis="y", visible=False)
ax[1].bar(tot_m.index, tot_m.values, width=22, color=PAL[0])
ax[1].set_title("Monthly traded value, all 503 stocks")
ax[1].set_ylabel("USD trillion per month")
fig.tight_layout()
save(fig, "08_traded_value.png")
R["traded"] = {"top10_share": float(top10.sum()), "top10": {k: float(v) for k, v in top10.items()}, "dollar_per_day_bn": float(dv_last.sum() / 1e9),
               "mo_last": float(tot_m.iloc[-1]), "mo_2019": float(tot_m["2019"].mean()), "mo_last12": float(tot_m.iloc[-12:].mean())}

# Chart 09 - risk and return of every stock
fig, ax = plt.subplots(figsize=(8.6, 5.2))
o = ok.dropna(subset=["cagr"])
ax.scatter(o.ann_vol * 100, o.cagr * 100, s=12, color="#b8b7b2", alpha=0.7, label="other stocks")
for t in FOCUS:
    ax.scatter(o.loc[t, "ann_vol"] * 100, o.loc[t, "cagr"] * 100, s=60, color=FCOL[t], edgecolor=SURF, linewidth=1.5, zorder=5, label=t)
ax.scatter(R["market"]["ann_vol"] * 100, R["market"]["cagr"] * 100, s=90, marker="D", color=MKT_COL, edgecolor=SURF, linewidth=1.5, zorder=6, label="Market")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel("annualised volatility (%)")
ax.set_ylabel("compound annual return (%)")
ax.set_title("Risk and return, all stocks with 2+ years of data")
ax.legend(ncol=2, loc="upper right")
fig.tight_layout()
save(fig, "09_risk_return.png")
corr_ = ret[FOCUS].corr()
R["corr_focus"] = corr_.round(3).to_dict()
# Chart 10 - correlations of the focus stocks
fig, ax = plt.subplots(figsize=(5.4, 4.3))
sns.heatmap(corr_, annot=True, fmt=".2f", cmap=sns.light_palette(PAL[0], as_cmap=True), vmin=0, vmax=1.7, cbar_kws={"label": "daily return correlation"},
            linewidths=2, linecolor=SURF, ax=ax, annot_kws={"size": 9, "color": INK})
ax.collections[0].colorbar.set_ticks([0, 0.5, 1.0])
ax.set_title("How closely the five stocks move together")
ax.set_xlabel("")
ax.set_ylabel("")
fig.tight_layout()
save(fig, "10_focus_correlation.png")

# %% 4. Forecast back-test (rolling origin, monthly data)
# Question: can any model forecast better than "tomorrow looks like today" (the naive model)?
H = (1, 3, 6, 12)
MIN_TRAIN = 84                                                   # first forecast origin after 7 years of months
top10 = focus_pool.index[:10].tolist()
SERIES = {"Market": np.log(mkt_m[:LAST_FULL].values)}
for t in top10:
    SERIES[t] = np.log(adjm[t][:LAST_FULL].values)
MDATES = mkt_m[:LAST_FULL].index
VOLS = np.log(mkt_mvol[:LAST_FULL].values)
BT_FILE = os.path.join(D3, "forecast_backtest.csv")


def run_backtest():
    out = []
    jobs = [(n, y, "price") for n, y in SERIES.items()] + [("Market volatility", VOLS, "volatility")]
    for name, y, kind in jobs:
        for o in range(MIN_TRAIN - 1, len(y) - 1):
            tr = y[:o + 1]
            fc = {}
            if kind == "price":
                fc["Naive"] = naive_forecast(tr, 12)
                fc["Drift"] = drift_forecast(tr, 12)
                fc["ARIMA"] = auto_arima(tr, d=1).forecast(12)["mean"]
            else:
                fc["Naive"] = naive_forecast(tr, 12)
                fc["Mean"] = np.repeat(tr.mean(), 12)
                fc["ARIMA"] = auto_arima(tr, d=0, grid=[(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]).forecast(12)["mean"]
            fc["Holt (damped)"] = HoltWinters(seasonal=False).fit(tr).forecast(12)["mean"]
            fc["Holt-Winters"] = HoltWinters(seasonal=True, m=12).fit(tr).forecast(12)["mean"]
            for mname, f in fc.items():
                for h in H:
                    if o + h < len(y):
                        out.append((name, kind, mname, h, MDATES[o].strftime("%Y-%m"), f[h - 1], y[o + h], tr[-1]))
        print("back-test done:", name, flush=True)
    return pd.DataFrame(out, columns=["series", "kind", "model", "h", "origin", "forecast", "actual", "last"])


if os.path.exists(BT_FILE):
    bt = pd.read_csv(BT_FILE)
else:
    bt = run_backtest()
    bt.round(6).to_csv(BT_FILE, index=False)
bt["err"] = bt.forecast - bt.actual


def block_boot_ratio(sub, model, h, nb=1000, block=12, seed=42):
    """RMSE ratio vs naive with a moving-block bootstrap over origin months (pooled across series)."""
    m = sub[(sub.model == model) & (sub.h == h)].pivot(index="origin", columns="series", values="err").sort_index()
    nv = sub[(sub.model == "Naive") & (sub.h == h)].pivot(index="origin", columns="series", values="err").sort_index()
    sm, sn = (m ** 2).values, (nv ** 2).values
    n = len(sm)
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(n / block))
    rat = []
    for _ in range(nb):
        starts = rng.integers(0, n - block + 1, nblk)
        idx = np.concatenate([np.arange(s_, s_ + block) for s_ in starts])[:n]
        rat.append(np.sqrt(np.nanmean(sm[idx]) / np.nanmean(sn[idx])))
    return float(np.sqrt(np.nanmean(sm) / np.nanmean(sn))), float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))


def metrics(sub, kind):
    rows = []
    base = sub[sub.model == "Naive"].set_index(["series", "origin", "h"]).err.abs()
    for (model, h), g in sub.groupby(["model", "h"]):
        g = g.set_index(["series", "origin", "h"])
        rmse = float(np.sqrt((g.err ** 2).mean()))
        mae = float(g.err.abs().mean())
        dirn = float((np.sign(g.forecast - g["last"]) == np.sign(g.actual - g["last"])).mean())
        beat = float((g.err.abs() < base.reindex(g.index)).mean()) if model != "Naive" else np.nan
        if model != "Naive":
            ratio, lo, hi = block_boot_ratio(sub, model, h)
        else:
            ratio, lo, hi = 1.0, 1.0, 1.0
        rows.append({"kind": kind, "model": model, "horizon_months": h, "rmse_log": rmse, "mae_log": mae, "rmse_vs_naive": ratio, "ratio_ci_low": lo,
                     "ratio_ci_high": hi, "share_beating_naive": beat, "direction_hit": dirn, "n": len(g)})
    return pd.DataFrame(rows)


mp = metrics(bt[bt.kind == "price"], "price")
mv = metrics(bt[bt.kind == "volatility"], "volatility")
mp.round(4).to_csv(os.path.join(D3, "model_metrics_price.csv"), index=False)
mv.round(4).to_csv(os.path.join(D3, "model_metrics_volatility.csv"), index=False)

# market index in index points (easy to read)
bm = bt[(bt.series == "Market")].copy()
bm["f_lvl"], bm["a_lvl"] = np.exp(bm.forecast), np.exp(bm.actual)
bm["e"] = bm.f_lvl - bm.a_lvl
mm = bm.groupby(["model", "h"]).apply(lambda g: pd.Series({"rmse_points": np.sqrt((g.e ** 2).mean()), "mae_points": g.e.abs().mean(),
                                                           "mape_pct": (g.e.abs() / g.a_lvl).mean() * 100,
                                                           "direction_hit": (np.sign(g.forecast - g["last"]) == np.sign(g.actual - g["last"])).mean()})).reset_index()
mm.round(3).to_csv(os.path.join(D3, "model_metrics_market_index.csv"), index=False)
R["bt"] = {"n_origins": int(bt[bt.series == "Market"].origin.nunique()), "first_origin": bt.origin.min(), "last_origin": bt.origin.max(),
           "series_price": list(SERIES.keys()), "price": mp.round(4).to_dict("records"), "vol": mv.round(4).to_dict("records"), "market": mm.round(3).to_dict("records")}

# chart 11 - back-test results
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, (mdf_, mods, ttl) in zip(ax, ((mp, ["Drift", "ARIMA", "Holt (damped)", "Holt-Winters"], "Price forecasts (market + 10 stocks)"),
                                     (mv, ["Mean", "ARIMA", "Holt (damped)", "Holt-Winters"], "Market volatility forecasts"))):
    w = 0.2
    for i, mname in enumerate(mods):
        d = mdf_[mdf_.model == mname].set_index("horizon_months").reindex(H)
        a.bar(np.arange(len(H)) + (i - 1.5) * w, d.rmse_vs_naive, w * 0.9, color=PAL[i], label=mname,
              yerr=[d.rmse_vs_naive - d.ratio_ci_low, d.ratio_ci_high - d.rmse_vs_naive], error_kw=dict(ecolor=MUTED, lw=0.9, capsize=2))
    a.axhline(1, color=INK, lw=1.2)
    a.set_xticks(range(len(H)))
    a.set_xticklabels([f"{h} month{'s' if h > 1 else ''} ahead" for h in H])
    a.set_ylabel("error vs naive (1.0 = same; lower is better)")
    a.set_title(ttl)
    a.legend(ncol=2, loc="upper left", fontsize=8.5)
    a.set_ylim(0, max(1.45, mdf_.ratio_ci_high.max() * 1.05))
fig.tight_layout()
save(fig, "11_forecast_backtest.png")

# %% 5. Final forecasts (fit on all months to Oct 2026)
FH = 12
fut = pd.date_range(LAST_FULL + pd.offsets.MonthEnd(1), periods=FH, freq="ME")
rows, fits = [], {}
for name in ["Market"] + FOCUS:
    y = SERIES[name] if name in SERIES else np.log(adjm[name][:LAST_FULL].values)
    am = auto_arima(y, d=1)
    f = am.forecast(FH, (0.8, 0.95))
    hw = HoltWinters(seasonal=False).fit(y).forecast(FH, (0.8, 0.95))
    fits[name] = {"y": y, "arima": f, "holt": hw, "order": (am.p, am.d, am.q), "drift": am.par_[0], "sigma": float(np.sqrt(am.sigma2_))}
    for mname, ff in (("ARIMA", f), ("Holt (damped)", hw)):
        for k in range(FH):
            rows.append({"series": name, "model": mname, "month": fut[k].strftime("%Y-%m"), "forecast": float(np.exp(ff["mean"][k])),
                         "lo80": float(np.exp(ff[0.8][0][k])), "hi80": float(np.exp(ff[0.8][1][k])),
                         "lo95": float(np.exp(ff[0.95][0][k])), "hi95": float(np.exp(ff[0.95][1][k]))})
    for k in range(FH):
        rows.append({"series": name, "model": "Naive", "month": fut[k].strftime("%Y-%m"), "forecast": float(np.exp(y[-1])), "lo80": np.nan, "hi80": np.nan, "lo95": np.nan, "hi95": np.nan})
# volatility forecast
am_v = auto_arima(VOLS, d=0, grid=[(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)])
fv = am_v.forecast(FH, (0.8, 0.95))
fits["Market volatility"] = {"y": VOLS, "arima": fv, "order": (am_v.p, 0, am_v.q)}
for k in range(FH):
    rows.append({"series": "Market volatility", "model": "ARIMA", "month": fut[k].strftime("%Y-%m"), "forecast": float(np.exp(fv["mean"][k])),
                 "lo80": float(np.exp(fv[0.8][0][k])), "hi80": float(np.exp(fv[0.8][1][k])), "lo95": float(np.exp(fv[0.95][0][k])), "hi95": float(np.exp(fv[0.95][1][k]))})
ftab = pd.DataFrame(rows)
ftab.round(4).to_csv(os.path.join(D1, "forecast_table.csv"), index=False)

summ = {}
for name in ["Market"] + FOCUS:
    f = fits[name]
    last_v = float(np.exp(f["y"][-1]))
    a = f["arima"]
    summ[name] = {"last": last_v, "arima_order": list(f["order"]), "drift_pct_month": float(f["drift"]), "monthly_sigma": f["sigma"],
                  "fc12": float(np.exp(a["mean"][-1])), "chg12": float(np.exp(a["mean"][-1]) / last_v - 1),
                  "lo80": float(np.exp(a[0.8][0][-1]) / last_v - 1), "hi80": float(np.exp(a[0.8][1][-1]) / last_v - 1),
                  "lo95": float(np.exp(a[0.95][0][-1]) / last_v - 1), "hi95": float(np.exp(a[0.95][1][-1]) / last_v - 1),
                  "holt12": float(np.exp(f["holt"]["mean"][-1]) / last_v - 1)}
fvm = fits["Market volatility"]
summ["Market volatility"] = {"last": float(np.exp(VOLS[-1])), "order": list(fvm["order"]), "mean_hist": float(np.exp(VOLS.mean())),
                             "fc1": float(np.exp(fvm["arima"]["mean"][0])), "fc12": float(np.exp(fvm["arima"]["mean"][-1])),
                             "lo80_12": float(np.exp(fvm["arima"][0.8][0][-1])), "hi80_12": float(np.exp(fvm["arima"][0.8][1][-1]))}
R["forecast"] = summ
R["forecast_months"] = [fut[0].strftime("%b %Y"), fut[-1].strftime("%b %Y")]

# chart 12 - market fan chart
f = fits["Market"]
lvl = np.exp(f["y"])
hist_idx = MDATES
cut = len(hist_idx) - 72
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(hist_idx[cut:], lvl[cut:], color=MKT_COL, lw=1.8, label="Market index (month-end)")
for lv, al in ((0.95, 0.15), (0.8, 0.28)):
    ax.fill_between(fut, np.exp(f["arima"][lv][0]), np.exp(f["arima"][lv][1]), color=PAL[0], alpha=al, lw=0, label=f"ARIMA {int(lv * 100)}% range")
ax.plot(fut, np.exp(f["arima"]["mean"]), color=PAL[0], lw=2, label=f"ARIMA{tuple(f['order'])} forecast")
ax.plot(fut, np.exp(f["holt"]["mean"]), color=PAL[1], lw=1.8, ls="--", label="Holt (damped) forecast")
ax.plot(fut, np.repeat(lvl[-1], FH), color=MUTED, lw=1.2, ls=":", label="Naive (no change)")
ax.set_title(f"Market index: 12-month forecast ({fut[0]:%b %Y} to {fut[-1]:%b %Y})")
ax.set_ylabel("index level")
ax.legend(loc="upper left", ncol=2, fontsize=8.5)
fig.tight_layout()
save(fig, "12_market_forecast.png")

# chart 13 - five stocks
fig, axs = plt.subplots(2, 3, figsize=(11, 6.2))
for a, t in zip(axs.ravel(), FOCUS):
    f = fits[t]
    l_ = np.exp(f["y"])
    a.plot(hist_idx[cut:], l_[cut:], color=FCOL[t], lw=1.6)
    a.fill_between(fut, np.exp(f["arima"][0.95][0]), np.exp(f["arima"][0.95][1]), color=FCOL[t], alpha=0.14, lw=0)
    a.fill_between(fut, np.exp(f["arima"][0.8][0]), np.exp(f["arima"][0.8][1]), color=FCOL[t], alpha=0.26, lw=0)
    a.plot(fut, np.exp(f["arima"]["mean"]), color=INK, lw=1.4)
    a.set_title(f"{t}: {summ[t]['chg12'] * 100:+.0f}% central, 80% range {summ[t]['lo80'] * 100:+.0f}% to {summ[t]['hi80'] * 100:+.0f}%", fontsize=9.5)
    a.tick_params(labelsize=8)
    a.xaxis.set_major_locator(mdates.YearLocator(2))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
a_ = axs.ravel()[5]
a_.axis("off")
a_.text(0, 0.7, "Line = central ARIMA forecast\nDark band = 80% range\nLight band = 95% range\n\nThe range widens fast:\nafter a year it is wider\nthan the central move.", fontsize=9.5, color=MUTED, va="top")
fig.suptitle("12-month forecasts for the five most-traded stocks (adjusted close, USD)", x=0.01, ha="left", fontweight="bold", fontsize=11.5)
fig.tight_layout()
save(fig, "13_stock_forecasts.png")

# chart 14 - volatility forecast
fv_ = fits["Market volatility"]
fig, ax = plt.subplots(figsize=(11, 3.9))
vv = np.exp(VOLS) * 100
ax.plot(MDATES[cut:], vv[cut:], color=PAL[0], lw=1.6, label="Monthly market volatility (annualised)")
for lv, al in ((0.95, 0.14), (0.8, 0.26)):
    ax.fill_between(fut, np.exp(fv_["arima"][lv][0]) * 100, np.exp(fv_["arima"][lv][1]) * 100, color=PAL[1], alpha=al, lw=0, label=f"{int(lv * 100)}% range")
ax.plot(fut, np.exp(fv_["arima"]["mean"]) * 100, color=PAL[1], lw=2, label="ARIMA forecast")
ax.axhline(np.exp(VOLS.mean()) * 100, color=MUTED, lw=1, ls=":", label=f"long-run level ({np.exp(VOLS.mean()) * 100:.0f}%)")
ax.set_title("Market volatility is forecastable: it drifts back to its long-run level")
ax.set_ylabel("%")
ax.legend(loc="upper right", ncol=2, fontsize=8.5)
fig.tight_layout()
save(fig, "14_volatility_forecast.png")

# %% 6. Save results
class NpEnc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        return super().default(o)


json.dump(R, open(os.path.join(D3, "results.json"), "w"), indent=1, cls=NpEnc)
print("done:", R["today"], "focus", FOCUS)
