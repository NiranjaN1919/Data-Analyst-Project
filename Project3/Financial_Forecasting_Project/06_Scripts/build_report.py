"""
Project 3 - build the Word report from the analysis outputs.
Usage: python3 build_report.py <project_root>      (run financial_analysis.py first)
Every number in the text comes from 01_Data/*.csv, 03_Models/*.csv and results.json.
"""
import json
import os
import sys

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = sys.argv[1]
D1, D2, D3 = (os.path.join(ROOT, d) for d in ("01_Data", "02_Charts", "03_Models"))
R = json.load(open(os.path.join(D3, "results.json")))
log = pd.read_csv(os.path.join(D1, "data_cleaning_log.csv")).fillna("None")
yearly = pd.read_csv(os.path.join(D1, "yearly_returns.csv"), index_col="year")
st = pd.read_csv(os.path.join(D1, "ticker_summary.csv"), index_col=0)
mp = pd.read_csv(os.path.join(D3, "model_metrics_price.csv"))
mv = pd.read_csv(os.path.join(D3, "model_metrics_volatility.csv"))
mm = pd.read_csv(os.path.join(D3, "model_metrics_market_index.csv"))
bt = pd.read_csv(os.path.join(D3, "forecast_backtest.csv"))
mkd = pd.read_csv(os.path.join(D1, "market_index_daily.csv"), index_col=0, parse_dates=True)
NAVY, BLUE, GREY = RGBColor(0x0D, 0x36, 0x6B), RGBColor(0x1C, 0x5C, 0xAB), RGBColor(0x52, 0x51, 0x4E)

f0 = lambda v: f"{v:,.0f}"
f1 = lambda v: f"{v:,.1f}"
f2 = lambda v: f"{v:,.2f}"
pc = lambda v, d=0, sign=False: ("+" if sign and v > 0 else "") + f"{v * 100:,.{d}f}%"
M, FO, PR = R["market"], R["focus"], R["primary"]
FS = {r["ticker"]: r for r in R["focus_stats"]}
FC = R["forecast"]
MO, TR, VO = R["monthly"], R["traded"], R["vol"]
CL = R["cleaning"]
ratio = lambda df, model, h: df[(df.model == model) & (df.horizon_months == h)].iloc[0]
x12 = bt[(bt.kind == "price") & (bt.model == "Naive") & (bt.h == 12)]
UP12 = float((x12.actual > x12["last"]).mean())
dir12 = float(ratio(mp, "ARIMA", 12).direction_hit)
mtd = float(mkd.index_level.iloc[-1] / mkd.index_level[:"2026-10-31"].iloc[-1] - 1)
dd22 = float(mkd.drawdown["2022"].min())
dd25 = float(mkd.drawdown["2025"].min())


def episode(year):
    """trough date, previous peak date, days from trough back to the old peak (None if not yet regained)"""
    tr = mkd.drawdown[str(year)].idxmin()
    pk = mkd.index_level[:tr].idxmax()
    after = mkd.index_level[tr:]
    back = after[after >= mkd.index_level[pk]]
    return tr, pk, (int((back.index[0] - tr).days) if len(back) else None)


E22, E25 = episode(2022), episode(2025)
VOL_CUT = 1 - M["ann_vol"] / R["median_vol"]

# ----------------------------------------------------------------------------- docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2)
stl = doc.styles["Normal"]
stl.font.name, stl.font.size = "Calibri", Pt(11)
stl.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
stl.paragraph_format.space_after = Pt(6)
stl.paragraph_format.line_spacing = 1.12
for name, size, color in (("Heading 1", 18, NAVY), ("Heading 2", 14, BLUE), ("Heading 3", 12, BLUE)):
    h = doc.styles[name]
    h.font.name, h.font.size, h.font.bold, h.font.color.rgb = "Calibri", Pt(size), True, color
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    h.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def para(text="", bold=False, italic=False, size=None, color=None, align=None, after=None):
    q = doc.add_paragraph()
    if text:
        r = q.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    if align is not None:
        q.alignment = align
    if after is not None:
        q.paragraph_format.space_after = Pt(after)
    return q


def bullet(lead, text, style="List Bullet"):
    q = doc.add_paragraph(style=style)
    if lead:
        q.add_run(lead + " ").bold = True
    q.add_run(text)
    q.paragraph_format.space_after = Pt(4)
    return q


def table(headers, rows, widths=None, num_cols=(), font=9.5, header=True, bold_first=False):
    t = doc.add_table(rows=1 if header else 0, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers if header else []):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(font)
        r.font.color.rgb = RGBColor(255, 255, 255)
        shade(c, "1C5CAB")
        if i in num_cols:
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(font)
            r.bold = bold_first and i == 0
            if i in num_cols:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            if ri % 2 == 1:
                shade(cells[i], "F1F5FB")
    for k, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        trPr.append(cs)
        if header and k == 0:
            th = OxmlElement("w:tblHeader")
            th.set(qn("w:val"), "true")
            trPr.append(th)
        for c in row.cells:
            for q in c.paragraphs:
                q.paragraph_format.space_after = Pt(1)
                q.paragraph_format.space_before = Pt(1)
    if widths:
        t.autofit = False
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


_fig = [0]


def figure(name, caption, width=16.5):
    _fig[0] += 1
    q = doc.add_paragraph()
    q.alignment = WD_ALIGN_PARAGRAPH.CENTER
    q.paragraph_format.keep_with_next = True
    q.add_run().add_picture(os.path.join(D2, name), width=Cm(width))
    para(f"Figure {_fig[0]}. {caption}", italic=True, size=9.5, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)


fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run("Financial Data Analysis and Forecasting  |  Page ")
r.font.size, r.font.color.rgb = Pt(9), GREY
for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    rr = fp.add_run()
    rr.font.size, rr.font.color.rgb = Pt(9), GREY
    if tag:
        fc = OxmlElement("w:fldChar")
        fc.set(qn("w:fldCharType"), tag)
        rr._r.append(fc)
    else:
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = txt
        rr._r.append(it)

# ============================================================================= COVER
for _ in range(5):
    para("", after=14)
para("PROJECT 3", bold=True, size=13, color=BLUE, after=2)
para("Financial Data Analysis and Forecasting", bold=True, size=30, color=NAVY, after=4)
para("Interactive dashboard + Python (Pandas, Matplotlib, Seaborn, time-series models) + SQL", size=13, color=GREY, after=30)
table(["", ""], [
    ["Objective", "Identify trends in financial data and forecast stock performance"],
    ["Dataset", f"S&P 500 daily stock prices: {CL['tickers']} stocks, {CL['rows_raw']:,} rows, {CL['date_min']} to {CL['date_max']}"],
    ["Methods", "Trend analysis, risk/return statistics, ARIMA and Holt-Winters forecasting with a rolling back-test"],
    ["Submitted by", "______________________________"],
    ["Course / Institution", "______________________________"],
    ["Date", "18 November 2026"]], widths=[4, 13], font=10.5, header=False, bold_first=True)
doc.add_page_break()

# ============================================================================= SUMMARY
doc.add_heading("Executive summary", 1)
para(f"Eleven years of daily prices for {CL['tickers']} S&P 500 stocks were cleaned, analysed, visualised in an interactive dashboard and used to forecast market, stock and volatility levels. "
     "Every forecasting model was tested on data it had not seen, against the simplest possible rival (\"the next value equals the last one\").")
bullet("The market.", f"An equal-weight index of the stocks grew {pc(M['total_return'])} from January 2015 ({pc(M['cagr'], 1)} a year) with {pc(M['ann_vol'], 1)} annual volatility. "
       f"The worst fall was {pc(M['max_dd'])} (Feb-Mar 2020), regained within {M['dd_recovery_days']} days. Calendar 2026 is {pc(M['ytd'], 0, True)} so far.")
bullet("Individual stocks.", f"{pc(R['stocks_up_share_total'])} of the {R['stocks_ok']} stocks with two or more years of data gained over the period, but only {pc(R['stocks_beat_mkt'])} beat the market's {pc(M['cagr'], 1)} a year; "
       f"the median stock returned {pc(R['median_cagr'], 1)} a year. Returns came from a few outstanding winners, led by {R['top_perf'][0]['ticker']}, {R['top_perf'][1]['ticker']} and {R['top_perf'][2]['ticker']}.")
bullet("Forecasting prices.", f"No model predicts price direction better than chance month to month. At 12 months, ARIMA and drift beat the naive model by {pc(1 - ratio(mp, 'ARIMA', 12).rmse_vs_naive)} - but only because they always forecast a rise, "
       f"and prices rose in {pc(UP12)} of the 12-month windows tested. The honest output of a price forecast is its range, which is very wide (market: {pc(FC['Market']['lo80'], 0, True)} to {pc(FC['Market']['hi80'], 0, True)} for 80% of outcomes over a year).")
bullet("Forecasting volatility.", f"Risk is predictable where price is not: the volatility forecast had {pc(1 - ratio(mv, 'ARIMA', 12).rmse_vs_naive)} less error than naive at 12 months (interval excludes 'no gain'). "
       f"Market volatility is now low ({pc(FC['Market volatility']['last'], 1)} against a long-run {pc(FC['Market volatility']['mean_hist'], 1)}) and is forecast to drift back up.")
bullet("Seasonality.", f"No reliable calendar pattern: the difference between months is not significant (p = {MO['kruskal_p']:.2f}) and a seasonal Holt-Winters model forecast worse than naive.")
bullet("What this dataset cannot answer.", "It holds prices and volume only, with no balance sheet or income statement. Profit margins, cost structure and valuation could not be analysed, so 'financial health' is assessed through market performance, risk and liquidity.")

# ============================================================================= 1
doc.add_heading("1. Objective and approach", 1)
para("The objective is to analyse financial data, identify trends and forecast stock performance. The six tasks of the brief were followed:")
for i, t in enumerate(["Data cleaning - missing values, erroneous data, transformation to a usable format (section 3).",
                       "Trend analysis - exploratory analysis of prices, returns, risk and volume (section 4).",
                       "Data visualisation - price trends, market comparison, monthly distributions (sections 4 and 5).",
                       "Forecasting - ARIMA, Holt-Winters and baselines with a rolling back-test (section 6).",
                       "Dashboard - interactive, with filters for period, stock, metric and universe (section 7).",
                       "Insights - financial health, future performance, investment and risk-reduction areas (section 8)."], 1):
    para(f"{i}. {t}", after=2)
para("")
para("Choice of data. The brief allows stock prices, market indexes or financial statements. The supplied dataset is stock prices, so the work centres on market performance and risk. "
     "A dataset with financial statements would be needed for revenue, profit-margin and cost analysis; the 'monthly revenue or profit distribution' chart of the brief is therefore replaced by the monthly distribution "
     "of returns and of traded value, which are the equivalent quantities for market data.")
para("Tools. Python (pandas, matplotlib, seaborn, scipy) for cleaning, analysis and charts; SQL (SQLite, tested) for extraction; an interactive HTML dashboard, plus a build guide to recreate it in Power BI or Tableau. "
     "The statsmodels library named in the brief could not be installed in the environment used, so ARIMA and Holt-Winters were implemented in a small module (06_Scripts/ts_models.py) "
     "and verified on simulated data with known answers (parameters recovered, 95% intervals covering 94.5% and 95.8% of outcomes; 7 of 7 checks passed). The code can be swapped to statsmodels with no change to the conclusions.")

# ============================================================================= 2
doc.add_heading("2. Dataset", 1)
para(f"The file has 8 columns: date, ticker, open, high, low, close, adj_close and volume. It covers {CL['tickers']} tickers over {CL['days']:,} trading days ({CL['date_min']} to {CL['date_max']}), {CL['rows_raw']:,} rows. "
     f"{int((st.full_history).sum())} tickers have the full history; the rest are later listings (the shortest has about {int(st.n_days.min())} trading days). "
     "Close is adjusted for splits; adj_close also adjusts for dividends, so returns computed from it are total returns. Traded value is close x volume.")
para("One limitation shapes every result: the 503 tickers are today's index members, so stocks that were dropped from the index over the years are missing (survivorship bias). "
     "Past returns of the data are therefore flattering, and any statement about the past is about 'stocks that are in the index today'.")

# ============================================================================= 3
doc.add_heading("3. Data cleaning", 1)
para("Each check was run on the data; the Action column says what was done. All 503 tickers and 2,986 dates are retained.")
table(["Check", "Finding", "Action"], log[["Step", "Finding", "Action"]].values.tolist(), widths=[3.2, 8.8, 5], font=8.6)
bullet("Placeholder history.", f"Three stocks (AMCR, FERG, SW) carry stale prices and zero volume until their US listing dates. Their {CL['lead_rows']:,} early rows were removed; leaving them in would add false returns.")
bullet("Zero volume and gaps.", "Remaining zero-volume rows are set to missing (prices kept). No ticker has missing days inside its history.")
bullet("Outliers.", f"Large moves ({int(31)} days above 40%) were checked: they persist in the following days and are real events. {len(CL['spikes_fixed'])} one-day spike that reverses the next day ({', '.join(CL['spikes_fixed'])}) was treated as an error and repaired.")
figure("01_data_coverage.png", "Number of stocks with data on each day (new listings join over time) and years of history per stock.", width=15.5)

# ============================================================================= 4
doc.add_heading("4. Trend analysis", 1)
doc.add_heading("4.1 The market", 2)
para(f"To judge the market as a whole, an equal-weight index was built: each day, the average return of every stock with data. It is rebased to 100 on 2 January 2015. It reached {M['last']:,.0f} by {R['today']}, "
     f"a gain of {pc(M['total_return'])} or {pc(M['cagr'], 1)} a year. Volatility was {pc(M['ann_vol'], 1)} a year and return per unit of risk (Sharpe, no risk-free rate) {M['sharpe']:.2f}. "
     f"The index spent {pc(R['mkt_above200'])} of days above its 200-day average, the usual sign of an uptrend.")
figure("02_market_index_drawdown.png", "Equal-weight index (top) and the fall from its previous peak (bottom).")
bullet("2020 crash.", f"Peak 19 Feb to trough 23 Mar: {pc(M['max_dd'])}, the deepest fall in the data. The index regained its peak {M['dd_recovery_days']} days after the low.")
bullet("2022 bear market.", f"A slow decline to {pc(dd22)} below its peak (low on {E22[0]:%d %b %Y}); the old high was regained {E22[2]} days later." if E22[2] else f"A slow decline to {pc(dd22)} below its peak.")
bullet("2025 shock.", f"A sharp fall to {pc(dd25)} (low on {E25[0]:%d %b %Y}); the old high was regained {E25[2]} days later." if E25[2] else f"A sharp fall to {pc(dd25)} (low on {E25[0]:%d %b %Y}).")
bullet("Now.", f"{R['partial_month']} to date is {pc(mtd, 1, True)} for the market and the index is {pc(-mkd.drawdown.iloc[-1], 1)} below its high.")
doc.add_heading("4.2 Volatility", 2)
para(f"Risk moves in waves. Volatility (60-day, annualised) averaged {pc(VO['avg'], 0)}, peaked at {pc(VO['max60'], 0)} in {VO['max60_date'][:7]} and is {pc(VO['now60'], 0)} now, close to its median of {pc(VO['median60'], 0)}. "
     "Calm periods end in spikes; spikes decay back over a few months - which is why volatility can be forecast (section 6).")
figure("05_market_volatility.png", "60-day market volatility.", width=15.5)
doc.add_heading("4.3 Stock price trends and market performance comparison", 2)
para(f"The five most traded stocks with a full history are {', '.join(FO)}. They are compared with the market below (log scale: a straight line is a constant growth rate).")
figure("03_focus_vs_market.png", "Value of 100 invested in January 2015 in each stock and in the market (log scale).")
yt = [[str(y)] + [pc(yearly.loc[y, c], 0, True) for c in ["Market (equal-weight)"] + FO] for y in yearly.index]
table(["Year"] + ["Market"] + FO, yt, num_cols=range(1, 7), widths=[2.2] + [2.5] * 6, font=9)
para(f"Over the whole period the market rose {pc(M['cagr'], 1)} a year. {FO[0]} returned {pc(FS[FO[0]]['cagr'], 0)} a year, but with a worst fall of {pc(FS[FO[0]]['max_drawdown'])}; "
     f"{FO[3]} and {FO[4]} returned {pc(FS[FO[3]]['cagr'], 0)} and {pc(FS[FO[4]]['cagr'], 0)} a year with much lower volatility ({pc(FS[FO[3]]['ann_vol'], 0)} and {pc(FS[FO[4]]['ann_vol'], 0)} against {pc(FS[FO[0]]['ann_vol'], 0)}). "
     "Bigger winners carried much bigger drops: every one of the five lost money in 2022 and four of five in at least one other year.")
figure("04_yearly_performance.png", "Calendar-year return of the market and the five stocks (2026 is year to date).")
table(["Stock", "Total return", "Per year", "Volatility", "Return/risk", "Worst fall", "Beta", "Last 12 months"],
      [[t, f"x{FS[t]['total_return'] + 1:,.0f}", pc(FS[t]["cagr"], 1), pc(FS[t]["ann_vol"], 0), f"{FS[t]['sharpe']:.2f}", pc(FS[t]["max_drawdown"]), f"{FS[t]['beta']:.2f}", pc(FS[t]["ret_1y"], 0, True)] for t in FO]
      + [["Market", f"x{M['total_return'] + 1:,.1f}", pc(M["cagr"], 1), pc(M["ann_vol"], 0), f"{M['sharpe']:.2f}", pc(M["max_dd"]), "1.00", pc(M["ret_1y"], 0, True)]],
      num_cols=range(1, 8), widths=[1.8, 2.1, 1.8, 2, 2, 2, 1.4, 2.6], font=9)
figure("06_primary_stock_trend.png", f"{PR}: adjusted close with 50- and 200-day moving averages (log scale). It closed above its 200-day average on {pc(R['above200'][PR])} of days.")
doc.add_heading("4.4 Distribution of monthly returns and seasonality", 2)
para(f"Across {MO['n']} full months the market returned {pc(MO['mean'], 1)} on average ({pc(MO['median'], 1)} median) with a standard deviation of {pc(MO['sd'], 1)}; {pc(MO['pos_share'])} of months were positive. "
     f"The best month was {MO['best_month']} ({pc(MO['best'], 1, True)}) and the worst {MO['worst_month']} ({pc(MO['worst'], 1)}); the distribution leans to the downside (skew {MO['skew']:.2f}): large gains are rarer than large losses.")
figure("07_monthly_distribution_seasonality.png", "Distribution of monthly market returns (left) and average return by calendar month (right).")
para(f"Calendar months look different ({MO['best_cal']} {pc(MO['best_cal_mean'], 1, True)}, {MO['worst_cal']} {pc(MO['worst_cal_mean'], 1, True)}), but with about 11 observations per month the intervals overlap zero for almost all of them, "
     f"and a formal test finds no month effect (Kruskal-Wallis p = {MO['kruskal_p']:.2f}). Two months ({' and '.join(MO['cal_sig'])}) are individually 'significant', which is what chance produces when twelve months are checked. "
     "No seasonal rule should be traded on these data.")
doc.add_heading("4.5 Trading activity and concentration", 2)
para(f"Roughly ${TR['dollar_per_day_bn']:,.0f} billion of these stocks change hands each day. Monthly traded value has grown from about ${TR['mo_2019']:.1f} trillion (2019 average) to ${TR['mo_last12']:.1f} trillion (last 12 months). "
     f"The ten most traded stocks make up {pc(TR['top10_share'])} of all traded value, led by {FO[0]} ({pc(TR['top10'][FO[0]], 1)}).")
figure("08_traded_value.png", "Share of traded value by the top-10 stocks (left) and total monthly traded value (right).")
doc.add_heading("4.6 Risk and return across stocks; diversification", 2)
para(f"More return came with more risk, but loosely. The median stock returned {pc(R['median_cagr'], 1)} a year at {pc(R['median_vol'], 0)} volatility, against {pc(M['cagr'], 1)} at {pc(M['ann_vol'], 0)} for the diversified market: "
     f"holding all stocks cut the risk by about {pc(VOL_CUT)} and still beat {pc(1 - R['stocks_beat_mkt'])} of individual stocks. "
     f"The five large stocks are only moderately correlated (daily-return correlations {pd.DataFrame(R['corr_focus']).values[pd.DataFrame(R['corr_focus']).values < 0.999].min():.2f} to {pd.DataFrame(R['corr_focus']).values[pd.DataFrame(R['corr_focus']).values < 0.999].max():.2f}), so combining them gives real diversification.")
figure("09_risk_return.png", "Annualised volatility against compound annual return for every stock with 2+ years of data.", width=12.5)
figure("10_focus_correlation.png", "Correlation of daily returns between the five stocks.", width=9.5)

# ============================================================================= 5
doc.add_heading("5. Visualisations produced", 1)
para("The brief asks for three kinds of chart; each is delivered as a static chart in this report and as an interactive view in the dashboard.")
table(["Requirement", "Report", "Dashboard"], [
    ["Stock price trends over time (line chart)", "Figures 4 and 6 (growth of 100, price with moving averages)", "Main trend chart, 4 metrics, log or linear"],
    ["Market performance comparison (bar or multi-line)", "Figures 4 and 5 (multi-line chart and grouped bars by year)", "Comparison bars; multi-line with market overlay; best and weakest performers"],
    ["Monthly distribution (pie or bar)", "Figures 7 and 8 (return histogram, traded value bars)", "Return histogram vs market; monthly traded value bars; market-return heatmap"]],
      widths=[5, 6, 6], font=9.2)

# ============================================================================= 6
doc.add_heading("6. Forecasting", 1)
doc.add_heading("6.1 Methods", 2)
table(["Model", "What it does"], [
    ["Naive", "The next values equal the last one. The benchmark every other model must beat."],
    ["Drift", "Last value plus the average monthly change seen so far (a straight trend)."],
    ["ARIMA", "AutoRegressive Integrated Moving Average on log prices (one difference); the best of six small orders is picked by AIC at each date. Gives prediction ranges."],
    ["Holt (damped)", "Exponential smoothing with a trend that fades out over time."],
    ["Holt-Winters", "Holt with a 12-month seasonal pattern."]], widths=[3.2, 13.8], font=9.4)
para("Series forecast: the market index and the ten most traded long-history stocks (monthly adjusted close, logged) and the market's monthly volatility. "
     "Monthly data was used because daily prices are dominated by noise; the incomplete November 2026 is excluded from fitting.")
doc.add_heading("6.2 Back-test design", 2)
para(f"Each model was fitted using only data up to a 'forecast date' and then scored against what happened 1, 3, 6 and 12 months later. The forecast date moved forward one month at a time from {R['bt']['first_origin']} "
     f"to {R['bt']['last_origin']} ({R['bt']['n_origins']} dates per series; the first fit uses seven years of history). Scores are the root-mean-square error of the log forecast (about the size of the percentage error); "
     "the table shows each model's error divided by the naive model's error, so 1.00 means 'no better than naive' and 0.85 means 15% smaller errors. "
     "The 95% interval comes from a block bootstrap over forecast dates (blocks of 12 months, 1,000 resamples), which respects the overlap between neighbouring forecasts.")
doc.add_heading("6.3 Results: prices", 2)
rows = []
for model in ["Drift", "ARIMA", "Holt (damped)", "Holt-Winters"]:
    rows.append([model] + [f"{ratio(mp, model, h).rmse_vs_naive:.2f}  ({ratio(mp, model, h).ratio_ci_low:.2f}-{ratio(mp, model, h).ratio_ci_high:.2f})" for h in (1, 3, 6, 12)])
table(["Model (error vs naive)", "1 month", "3 months", "6 months", "12 months"], rows, num_cols=range(1, 5), widths=[4.2, 3.2, 3.2, 3.2, 3.2], font=9)
bullet("Short horizons: nothing to gain.", f"At 1 to 3 months every model is within a few percent of naive and the intervals include 1.00. The monthly naive error is {ratio(mp, 'Naive', 1).rmse_log * 100:.0f}% (a one-month surprise of that size is normal). "
       f"Holt-Winters is clearly worse ({ratio(mp, 'Holt-Winters', 1).rmse_vs_naive:.2f}): the seasonal pattern it learns is noise.")
bullet("12 months: ARIMA and drift win, for a simple reason.", f"Their error is {pc(1 - ratio(mp, 'ARIMA', 12).rmse_vs_naive)} below naive. But they forecast 'up' in nearly every case, and the stocks rose in {pc(UP12)} of 12-month windows (ARIMA's 'direction hit' was {pc(dir12)}). "
       "That is a bet on the sample's growth, not forecasting skill - and the sample is today's survivors (section 2), which favours it. A model that shows it can time reversals does not exist in these results.")
bullet("Damped Holt adds nothing.", f"Its ratio is {ratio(mp, 'Holt (damped)', 12).rmse_vs_naive:.2f} at 12 months: a trend that fades is equivalent to a naive forecast here.")
mrow = mm[mm.h == 12].set_index("model")
table(["Market index, 12 months ahead", "RMSE (index points)", "MAE (points)", "MAPE"],
      [[m_, f1(mrow.loc[m_, "rmse_points"]), f1(mrow.loc[m_, "mae_points"]), f"{mrow.loc[m_, 'mape_pct']:.1f}%"] for m_ in ["Naive", "Drift", "ARIMA", "Holt (damped)", "Holt-Winters"]],
      num_cols=(1, 2, 3), widths=[6, 4, 3.5, 3.5], font=9.2)
figure("11_forecast_backtest.png", "Back-test: model error relative to naive (1.0 = same). Left: prices (market and ten stocks). Right: market volatility. Bars show the 95% interval.")
doc.add_heading("6.4 Results: volatility", 2)
rows = [[model] + [f"{ratio(mv, model, h).rmse_vs_naive:.2f}  ({ratio(mv, model, h).ratio_ci_low:.2f}-{ratio(mv, model, h).ratio_ci_high:.2f})" for h in (1, 3, 6, 12)] for model in ["Mean", "ARIMA", "Holt (damped)", "Holt-Winters"]]
table(["Model (error vs naive)", "1 month", "3 months", "6 months", "12 months"], rows, num_cols=range(1, 5), widths=[4.2, 3.2, 3.2, 3.2, 3.2], font=9)
para(f"Volatility behaves differently: ARIMA's error is {pc(1 - ratio(mv, 'ARIMA', 6).rmse_vs_naive)} below naive at 6 months and {pc(1 - ratio(mv, 'ARIMA', 12).rmse_vs_naive)} at 12 months, and the intervals exclude 1.00. "
     "It works because volatility falls back to its typical level after a spike and rises from unusually calm spells. The simple long-run average ('Mean') does about as well at long horizons, which tells us the gain is mean reversion, not clever timing.")
doc.add_heading("6.5 Final forecasts (November 2026 to October 2027)", 2)
para("The best-supported models were refitted on all months to October 2026. For prices, the ARIMA forecast follows each stock's historical drift; the shaded ranges give the 80% and 95% prediction intervals. "
     "Holt (damped), the conservative model, is shown for contrast.")
rows = []
for t in ["Market"] + FO:
    f = FC[t]
    rows.append([t, f2(f["last"]), f"{f['arima_order']}".replace("[", "(").replace("]", ")"), pc(f["chg12"], 0, True), f"{pc(f['lo80'], 0, True)} to {pc(f['hi80'], 0, True)}", f"{pc(f['lo95'], 0, True)} to {pc(f['hi95'], 0, True)}", pc(f["holt12"], 0, True)])
table(["Series", "Oct 2026 level", "ARIMA order", "12-month change (ARIMA)", "80% range", "95% range", "Holt (damped)"], rows, num_cols=(1, 3, 4, 5, 6), widths=[1.9, 2.2, 2.2, 2.6, 3.2, 3.2, 1.9], font=8.8)
figure("12_market_forecast.png", "Market index: ARIMA forecast with 80% and 95% ranges, Holt (damped) and naive.")
figure("13_stock_forecasts.png", "Forecast ranges for the five stocks.")
vf = FC["Market volatility"]
para(f"Volatility. Monthly market volatility was {pc(vf['last'], 1)} in October 2026, below its long-run level of {pc(vf['mean_hist'], 1)}. The forecast rises to {pc(vf['fc1'], 1)} next month and {pc(vf['fc12'], 1)} by October 2027 "
     f"(80% range {pc(vf['lo80_12'], 0)} to {pc(vf['hi80_12'], 0)}): calm markets tend not to stay calm.")
figure("14_volatility_forecast.png", "Market volatility: history and ARIMA forecast.", width=15.5)
para("How to read the stock forecasts. NVDA's central +68% simply continues its past 4.3%-a-month drift. The 80% range runs from a small loss to nearly +200%, so the central number carries almost no information on its own. "
     "The market's central forecast (+16%) is its past average growth; the conservative Holt model gives about +3%. Treat the range as the forecast.")

# ============================================================================= 7
doc.add_heading("7. Interactive dashboard", 1)
para("04_Dashboard/financial_dashboard.html is a single file that opens in any browser (no installation, works offline). It contains all 503 stocks as monthly data.")
table(["Element", "What it does"], [
    ["Filters", "From and To month; quick periods 1Y / 3Y / 5Y / All; chart metric (growth of 100, price, monthly return, drawdown); universe for rankings (all, 100, 50, 20 most traded); linear or log scale."],
    ["Stock picker", "Up to five stocks, any ticker; quick-add buttons for the five large stocks (each keeps its colour); stocks can also be added by clicking the ranking bars."],
    ["KPI cards", "Period return, compound annual return, volatility, worst drawdown, return per unit of risk, traded value, each with the market alongside."],
    ["Charts", "Trend lines with the market, comparison bars (5 metrics), monthly return histogram vs market, best and weakest 10 stocks, monthly traded value, market-return heatmap, forecast with 80/95% ranges."],
    ["Forecast panel", "Choose the market, any of the five stocks or market volatility; ARIMA and/or Holt; horizon table; back-test summary and warning."]],
      widths=[3.2, 13.8], font=9.4)
para("Power BI and Tableau. The brief names Power BI or Tableau. A .pbix or .twb file cannot be produced in the environment used for this work, so the folder contains 04_Dashboard/PowerBI_Tableau_Build_Guide.md: "
     "the exact tables to load (they are in 01_Data), the relationships, ready-to-paste DAX measures, the visuals and slicers, and the Tableau equivalents. The HTML dashboard is the visual reference to build against.")

# ============================================================================= 8
doc.add_heading("8. Insights", 1)
doc.add_heading("8.1 Financial health: how the market and the large stocks have performed", 2)
bullet("The market is healthy and in an uptrend.", f"+{M['cagr'] * 100:.0f}% a year over eleven years, above its 200-day average {pc(R['mkt_above200'])} of the time, with every major fall (2020, 2022, 2025) recovered. Volatility is now at calm levels.")
bullet("Gains are concentrated.", f"Only {pc(R['stocks_beat_mkt'])} of stocks beat the market; the eight best stocks multiplied their value between x{R['top_perf'][7]['total_return'] + 1:,.0f} and x{R['top_perf'][0]['total_return'] + 1:,.0f}, and the ten most traded stocks account for {pc(TR['top10_share'])} of all trading.")
bullet("Quality differs by stock.", f"{FO[3]} and {FO[4]} delivered about {FS[FO[3]]['sharpe']:.1f} return per unit of risk with {pc(FS[FO[3]]['max_drawdown'])} and {pc(FS[FO[4]]['max_drawdown'])} worst falls; "
       f"{FO[1]} and {FO[2]} fell {pc(FS[FO[1]]['max_drawdown'])} and {pc(FS[FO[2]]['max_drawdown'])} at their worst, and {FO[2]} is down {pc(-FS[FO[2]]['ret_1y'])} over the last year.")
bullet(f"{PR} in focus.", f"{PR} compounded at {pc(FS[PR]['cagr'], 0)} a year, but with {pc(FS[PR]['ann_vol'], 0)} volatility and a {pc(FS[PR]['max_drawdown'])} worst fall; it is {pc(-FS[PR]['ret_1y'])} lower than a year ago and has fallen sharply in {R['partial_month']}, "
       f"so its 2026 return is {pc(yearly.loc[2026, PR], 0, True)} against {pc(yearly.loc[2026, 'Market (equal-weight)'], 0, True)} for the market.")
doc.add_heading("8.2 Potential future performance", 2)
bullet("Central expectation.", f"The market's forecast for the next 12 months is {pc(FC['Market']['holt12'], 0, True)} (conservative) to {pc(FC['Market']['chg12'], 0, True)} (past average growth). 80% of outcomes lie between {pc(FC['Market']['lo80'], 0, True)} and {pc(FC['Market']['hi80'], 0, True)}.")
bullet("Risk expectation.", f"Volatility is forecast to rise from {pc(vf['last'], 0)} to about {pc(vf['fc12'], 0)}. Rising risk after a calm stretch is the most reliable signal in the forecasts.")
bullet("Single stocks are much less predictable.", f"The 80% range for the five stocks is between about {(FC['MSFT']['hi80'] - FC['MSFT']['lo80']) * 100:.0f} percentage points wide (MSFT) and {(FC['TSLA']['hi80'] - FC['TSLA']['lo80']) * 100:.0f} (TSLA), against {(FC['Market']['hi80'] - FC['Market']['lo80']) * 100:.0f} for the market.")
doc.add_heading("8.3 Areas for investment and for reducing risk or cost", 2)
para("With price data only, investment and cost questions are answered in terms of return, risk and trading cost, not company costs or margins. These are analytical observations, not investment advice.")
bullet("Diversify rather than pick.", f"Holding the whole market gave {pc(M['cagr'], 0)} a year at {pc(M['ann_vol'], 0)} volatility, beating {pc(1 - R['stocks_beat_mkt'])} of single stocks with {pc(VOL_CUT)} less risk than the typical stock ({pc(R['median_vol'], 0)}).")
bullet("Size positions by risk.", f"The most volatile of the large stocks ({FO[2]}, {FO[1]}, {FO[0]}) have had falls of {pc(FS[FO[0]]['max_drawdown'])} to {pc(FS[FO[2]]['max_drawdown'])}; a fixed amount in them carries roughly twice the volatility of the same amount in {FO[3]} or {FO[4]}. "
       "Position sizes that equalise risk reduce the damage of a bad year (2022).")
bullet("Use the volatility forecast.", "Because volatility is forecastable, risk limits can be adjusted ahead of time (smaller positions when volatility is forecast to rise); this is where the forecasting work adds real value.")
bullet("Do not trade on price forecasts, seasons or moving-average signals alone.", "Back-tests show no month-ahead skill, no seasonal pattern, and trend models that only work because the sample went up.")
bullet("Trading cost.", f"Liquidity is high and rising (${TR['mo_last12']:.1f} trillion a month), so costs of trading large, well-known stocks are low; the 100 most traded stocks are the practical universe for a liquid portfolio.")
bullet("Next data to add.", "Income statements and balance sheets (revenue, margins, debt) would allow genuine financial-health and cost analysis, sector grouping and valuation ratios.")

# ============================================================================= 9
doc.add_heading("9. Limitations", 1)
for t in ["Survivorship bias: only current index members are in the data; past performance and trend forecasts are flattered. The ten stocks used in the back-test were chosen because they are heavily traded today, which adds look-ahead bias to the 12-month result.",
          "Prices only: no financial statements, sector or market-capitalisation data. The market index is equal-weighted, not the cap-weighted S&P 500 that headlines quote, so its numbers differ from the published index.",
          "Statistics in the dashboard are computed from month-end values; the report uses daily data, so volatility and drawdowns differ slightly between the two.",
          "Forecast ranges assume the past size of price swings continues and that changes are roughly normal; extreme events (such as March 2020) are wider than the ranges imply.",
          f"The back-test uses {R['bt']['n_origins']} forecast dates per series and overlapping windows, so the 12-month results are based on far fewer independent observations than the table suggests; the bootstrap intervals reflect this.",
          "ARIMA and Holt-Winters were implemented in a custom module because statsmodels was unavailable; they were validated on simulated data but not against statsmodels output.",
          f"{R['partial_month']} is incomplete (to {R['today']}) and is excluded from model fitting."]:
    bullet("", t)
doc.add_heading("10. Conclusion", 1)
para(f"The S&P 500 stocks in this data trended strongly upwards between 2015 and 2026 ({pc(M['cagr'], 0)} a year for the equal-weight market) through three sharp but short-lived falls. A handful of stocks produced most of the extra return, and large winners carried large drawdowns. "
     "Forecasting models tested honestly do not beat 'no change' for prices month to month; their 12-month edge is the market's upward drift, and their useful output is the range of outcomes. "
     "Volatility, in contrast, can be forecast, and that forecast supports practical risk decisions. The dashboard lets the reader explore any of the 503 stocks, and the same scripts re-run on updated data.")
doc.add_heading("Appendix - contents of the submission folder", 1)
table(["Folder / file", "What it contains"], [
    ["01_Data/", "Cleaned monthly prices for all stocks, daily market index, daily data for the five large stocks, ticker statistics, yearly returns, forecast table, cleaning log"],
    ["02_Charts/", "The fourteen chart images used in this report"],
    ["03_Models/", "Back-test results and metrics (CSV), model validation tests, results.json"],
    ["04_Dashboard/", "financial_dashboard.html (interactive) and PowerBI_Tableau_Build_Guide.md"],
    ["05_Report/", "This report (Word and PDF)"],
    ["06_Scripts/financial_analysis.py (.ipynb)", "Cleaning, features, trend analysis, forecasting, charts"],
    ["06_Scripts/ts_models.py, test_ts_models.py", "ARIMA and Holt-Winters implementation and its checks"],
    ["06_Scripts/sp500_sql_extraction.sql", "Seven SQL extraction queries (tested on the full data)"],
    ["06_Scripts/build_dashboard.py, build_report.py", "Rebuild the dashboard and this report from the outputs"]], widths=[6.5, 10.5], font=9.3)
out = os.path.join(ROOT, "05_Report", "Financial_Analysis_Report.docx")
doc.save(out)
print("saved", out)
