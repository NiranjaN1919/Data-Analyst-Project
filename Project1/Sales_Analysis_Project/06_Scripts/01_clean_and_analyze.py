"""
Sales Analysis Dashboard - Step 1: data cleaning, aggregation, charts.

Usage:  python3 01_clean_and_analyze.py <project_root>
Reads   <root>/01_Data/ecommerce_sales_raw.csv
Writes  <root>/01_Data/ecommerce_sales_cleaned.csv
        <root>/01_Data/data_cleaning_log.csv
        <root>/02_Aggregations/sales_aggregations.xlsx
        <root>/03_Charts/*.png
        <root>/04_Dashboard/dashboard_data.json   (feeds the HTML dashboard)
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

ROOT = sys.argv[1]
RAW = os.path.join(ROOT, "01_Data", "ecommerce_sales_raw.csv")
OUT_DATA = os.path.join(ROOT, "01_Data")
OUT_AGG = os.path.join(ROOT, "02_Aggregations")
OUT_CH = os.path.join(ROOT, "03_Charts")
OUT_DASH = os.path.join(ROOT, "04_Dashboard")

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TIER_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue ramp
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": False, "font.family": "DejaVu Sans",
})


def short(v, d=1):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.{d + 1}f}M"
    if a >= 1e3:
        return f"{v / 1e3:.{d}f}K"
    return f"{v:.0f}"


KFMT = FuncFormatter(lambda v, _: short(v, 0))


def save(fig, name):
    fig.savefig(os.path.join(OUT_CH, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ================================================================ 1. CLEAN
log = []
raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
log.append(("Load raw file", f"{len(raw):,} rows x {raw.shape[1]} columns", "None"))

blank = raw.apply(lambda r: "".join(r).strip() == "", axis=1)
df = raw[~blank].copy()
log.append(("Fully empty rows", f"{int(blank.sum()):,} rows contain no data at all (padding after row {len(df):,})",
            f"Removed -> {len(df):,} real orders remain"))

df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
miss = int(df.isna().sum().sum())
log.append(("Missing values (real rows)", f"{miss} missing cells across all 17 columns", "Nothing to impute or drop"))

for c in ["price", "discount", "total_amount", "shipping_cost", "profit_margin"]:
    df[c] = pd.to_numeric(df[c], errors="raise")
for c in ["quantity", "delivery_time_days", "customer_age"]:
    df[c] = pd.to_numeric(df[c], errors="raise").astype(int)
for c in ["order_id", "customer_id", "product_id", "category", "payment_method", "region", "returned",
          "customer_gender"]:
    df[c] = df[c].str.strip()
log.append(("Data types", "All numeric columns parsed; text columns trimmed of stray spaces",
            "Quantity, delivery days and age cast to integer"))

df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="raise")
log.append(("Date format", "order_date stored as text dd-mm-yyyy; 0 unparseable values",
            "Converted to a true date (yyyy-mm-dd) so Power BI treats it as a Date"))

dups_full = int(df.duplicated().sum())
dups_id = int(df["order_id"].duplicated().sum())
df = df.drop_duplicates().drop_duplicates("order_id")
log.append(("Duplicates", f"{dups_full} fully duplicated rows, {dups_id} repeated order_id", "Nothing removed"))

calc = (df.price * df.quantity * (1 - df.discount)).round(2)
mism = int(((calc - df.total_amount).abs() > 0.05).sum())
log.append(("Consistency: total_amount", f"{mism} orders where total_amount != price x quantity x (1 - discount)",
            "Totals reconcile exactly"))

bad = int(((df.price <= 0) | (df.quantity <= 0) | (df.shipping_cost < 0) | (~df.discount.between(0, 1))).sum())
log.append(("Consistency: ranges", f"{bad} rows with non-positive price/quantity, negative shipping or invalid discount",
            "None found"))

_g = df.groupby("product_id")
_rep = int((_g.size() > 1).sum())
_multi_cat = int((_g.category.nunique() > 1).sum())
_multi_price = int((_g.price.nunique() > 1).sum())
log.append(("Consistency: product_id", f"{_rep} product IDs appear in more than one order, always at different prices; "
            f"{_multi_cat} of them are listed under different categories (product_id is not a reliable unique key)",
            "Kept all rows; products are identified by product_id + category in rankings"))

cats = {c: sorted(df[c].unique()) for c in ["category", "region", "payment_method", "returned", "customer_gender"]}
log.append(("Consistency: labels", "Category (7), region (5), payment method (6), returned (2), gender (3) - no spelling or case variants",
            "None needed"))

neg = int((df.profit_margin < 0).sum())
log.append(("Negative profit", f"{neg} orders ({neg / len(df):.1%}) have profit_margin < 0 (loss-making after discount/shipping)",
            "Kept - genuine business losses, analysed in the report"))
p99 = df.total_amount.quantile(0.99)
hi = int((df.total_amount > p99).sum())
log.append(("Outliers", f"{hi} orders above the 99th percentile (> {p99:,.0f})",
            "Kept - high-ticket orders are real, flagged with high_value_order"))

# derived columns
df["year"] = df.order_date.dt.year
df["quarter"] = "Q" + df.order_date.dt.quarter.astype(str)
df["year_quarter"] = df.year.astype(str) + "-" + df.quarter
df["year_month"] = df.order_date.dt.strftime("%Y-%m")
df["month_name"] = df.order_date.dt.strftime("%b")
df["gross_amount"] = (df.price * df.quantity).round(2)
df["discount_amount"] = (df.gross_amount - df.total_amount).round(2)
df["is_returned"] = (df.returned == "Yes").astype(int)
df["profit_pct"] = (df.profit_margin / df.total_amount * 100).round(2)
df["age_group"] = pd.cut(df.customer_age, [17, 24, 34, 44, 54, 64, 120],
                         labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(str)
df["high_value_order"] = (df.total_amount > p99).astype(int)
df = df.sort_values("order_date").reset_index(drop=True)
log.append(("Derived columns", "year, quarter, year_quarter, year_month, month_name, gross_amount, discount_amount, "
            "is_returned, profit_pct, age_group, high_value_order", "Added for aggregation and Power BI slicers"))

out = df.copy()
out["order_date"] = out.order_date.dt.strftime("%Y-%m-%d")
out.to_csv(os.path.join(OUT_DATA, "ecommerce_sales_cleaned.csv"), index=False)
pd.DataFrame(log, columns=["Step", "Finding", "Action"]).to_csv(os.path.join(OUT_DATA, "data_cleaning_log.csv"), index=False)

D0, D1 = df.order_date.min(), df.order_date.max()
print(f"Cleaned: {len(df):,} orders, {D0.date()} -> {D1.date()}")

# ============================================================ 2. AGGREGATE


def agg(g, keys):
    r = g.groupby(keys, observed=True).agg(
        total_sales=("total_amount", "sum"), avg_sales=("total_amount", "mean"), orders=("order_id", "count"),
        units_sold=("quantity", "sum"), total_profit=("profit_margin", "sum"),
        return_rate=("is_returned", "mean"), avg_delivery_days=("delivery_time_days", "mean")).reset_index()
    r["sales_share"] = r.total_sales / r.total_sales.sum()
    r["profit_pct_of_sales"] = r.total_profit / r.total_sales
    r["net_sales_after_returns"] = g[g.is_returned == 0].groupby(keys, observed=True).total_amount.sum().reindex(
        pd.MultiIndex.from_frame(r[keys]) if len(keys) > 1 else r[keys[0]]).values
    return r


by_region = agg(df, ["region"]).sort_values("total_sales", ascending=False)
by_cat = agg(df, ["category"]).sort_values("total_sales", ascending=False)
by_year = agg(df, ["year"])
by_quarter = agg(df, ["year_quarter"])
by_month = agg(df, ["year_month"])
by_pay = agg(df, ["payment_method"]).sort_values("total_sales", ascending=False)
by_age = agg(df, ["age_group"])
by_gender = agg(df, ["customer_gender"])
reg_cat = agg(df, ["region", "category"])
reg_year = agg(df, ["region", "year"])
cat_year = agg(df, ["category", "year"])
reg_month = agg(df, ["region", "year_month"])
cat_month = agg(df, ["category", "year_month"])

# 12-month windows (calendar years are partial: data runs 12 Sep 2023 - 11 Sep 2025)
mid = pd.Timestamp("2024-09-12")
df["period12"] = np.where(df.order_date < mid, "Year 1 (12 Sep 23 - 11 Sep 24)", "Year 2 (12 Sep 24 - 11 Sep 25)")
by_p12 = agg(df, ["period12"])
reg_p12 = df.pivot_table(index="region", columns="period12", values="total_amount", aggfunc="sum")
reg_p12.columns = ["Year1", "Year2"]
reg_p12["growth"] = reg_p12.Year2 / reg_p12.Year1 - 1
cat_p12 = df.pivot_table(index="category", columns="period12", values="total_amount", aggfunc="sum")
cat_p12.columns = ["Year1", "Year2"]
cat_p12["growth"] = cat_p12.Year2 / cat_p12.Year1 - 1

# top products (product_id is almost unique -> ranking = biggest single purchases)
prod = df.groupby(["product_id", "category"]).agg(total_sales=("total_amount", "sum"), orders=("order_id", "count"),
                                                  units=("quantity", "sum"), profit=("profit_margin", "sum")).reset_index()
prod = prod.sort_values("total_sales", ascending=False)
top_prod = prod.head(15).reset_index(drop=True)
top_prod.insert(0, "rank", range(1, len(top_prod) + 1))

# customer segmentation by total spend
cust = df.groupby("customer_id").agg(total_spend=("total_amount", "sum"), orders=("order_id", "count"),
                                     age=("customer_age", "mean"), profit=("profit_margin", "sum"),
                                     region=("region", "first")).reset_index()
q50, q80, q95 = cust.total_spend.quantile([.5, .8, .95]).values
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
cust["segment"] = pd.cut(cust.total_spend, [-np.inf, q50, q80, q95, np.inf], labels=TIERS).astype(str)
seg = cust.groupby("segment").agg(customers=("customer_id", "count"), total_spend=("total_spend", "sum"),
                                  avg_spend=("total_spend", "mean"), avg_orders=("orders", "mean"),
                                  avg_age=("age", "mean"), profit=("profit", "sum")).reindex(TIERS).reset_index()
seg["customer_share"] = seg.customers / seg.customers.sum()
seg["sales_share"] = seg.total_spend / seg.total_spend.sum()
seg["rule"] = [f"<= {q50:,.0f} (bottom 50%)", f"{q50:,.0f} - {q80:,.0f} (next 30%)",
               f"{q80:,.0f} - {q95:,.0f} (next 15%)", f"> {q95:,.0f} (top 5%)"]
df = df.merge(cust[["customer_id", "segment"]], on="customer_id", how="left")

# KPI summary
total_sales = df.total_amount.sum()
kpi = pd.DataFrame([
    ("Period covered", f"{D0:%d %b %Y} - {D1:%d %b %Y}"),
    ("Total orders", len(df)),
    ("Unique customers", df.customer_id.nunique()),
    ("Total sales", round(total_sales, 2)),
    ("Average sales per order", round(df.total_amount.mean(), 2)),
    ("Total profit", round(df.profit_margin.sum(), 2)),
    ("Profit as % of sales", round(df.profit_margin.sum() / total_sales * 100, 2)),
    ("Total units sold", int(df.quantity.sum())),
    ("Return rate (orders)", round(df.is_returned.mean() * 100, 2)),
    ("Sales value of returned orders", round(df[df.is_returned == 1].total_amount.sum(), 2)),
    ("Average delivery days", round(df.delivery_time_days.mean(), 2)),
    ("Average discount", round(df.discount.mean() * 100, 2)),
], columns=["Metric", "Value"])

# ------------------------------------------------------------ Excel export
xlsx = os.path.join(OUT_AGG, "sales_aggregations.xlsx")
sheets = {
    "KPI_Summary": kpi, "By_Region": by_region, "By_Category": by_cat, "By_Year": by_year,
    "By_Quarter": by_quarter, "By_Month": by_month, "By_Payment": by_pay, "By_AgeGroup": by_age,
    "By_Gender": by_gender, "Region_x_Category": reg_cat, "Region_x_Year": reg_year,
    "Category_x_Year": cat_year, "Region_x_Month": reg_month, "Category_x_Month": cat_month,
    "Top_Products": top_prod, "Customer_Segments": seg, "Customers": cust.sort_values("total_spend", ascending=False),
    "Growth_Region_12m": reg_p12.reset_index(), "Growth_Category_12m": cat_p12.reset_index(),
}
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    for name, d in sheets.items():
        d = d.copy()
        for c in d.columns:
            if d[c].dtype.kind == "f":
                d[c] = d[c].round(4)
        d.to_excel(xw, sheet_name=name, index=False)
        ws = xw.sheets[name]
        from openpyxl.styles import Alignment, Font, PatternFill
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="1C5CAB")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for i, col in enumerate(d.columns, 1):
            w = max(len(str(col)), *(len(str(v)) for v in d[col].head(200))) + 2
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(w, 11), 34)
        ws.freeze_panes = "A2"
print("Excel written")

# ============================================================ 3. CHARTS
# 3.1 monthly trend
m = by_month.copy()
m["date"] = pd.to_datetime(m.year_month + "-01")
m["partial"] = m.year_month.isin([D0.strftime("%Y-%m"), D1.strftime("%Y-%m")])
m["ma3"] = m.total_sales.rolling(3, center=True).mean()
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(m.date, m.total_sales, color=BLUE, lw=2, label="Monthly sales", zorder=3)
ax.plot(m.date, m.ma3, color=INK2, lw=1.6, ls="--", label="3-month average", zorder=2)
ax.scatter(m.date[~m.partial], m.total_sales[~m.partial], s=34, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4)
ax.scatter(m.date[m.partial], m.total_sales[m.partial], s=40, facecolor=SURFACE, edgecolor=BLUE, lw=1.6, zorder=4,
           label="Partial month")
pk = m[~m.partial].total_sales.idxmax()
lo = m[~m.partial].total_sales.idxmin()
for i, dy in ((pk, 12), (lo, -18)):
    ax.annotate(f"{m.year_month[i]}: {short(m.total_sales[i])}", (m.date[i], m.total_sales[i]), xytext=(0, dy),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.yaxis.set_major_formatter(KFMT)
ax.set_ylim(0, m.total_sales.max() * 1.18)
ax.grid(axis="y", color=GRID, lw=.8)
ax.set_axisbelow(True)
ax.set_title("Sales trend over time (monthly total sales)")
ax.legend(frameon=False, loc="upper left", ncol=3)
fig.text(0.01, -0.02, f"Data runs {D0:%d %b %Y} - {D1:%d %b %Y}, so the first and last months are partial (hollow markers).",
         fontsize=8.5, color=INK2)
save(fig, "01_sales_trend_monthly.png")

# 3.2 category bar
c = by_cat.sort_values("total_sales")
fig, ax = plt.subplots(figsize=(9, 4.8))
b = ax.barh(c.category, c.total_sales, color=BLUE, height=.62)
for r, v, s in zip(b, c.total_sales, c.sales_share):
    ax.text(v + c.total_sales.max() * .012, r.get_y() + r.get_height() / 2, f"{short(v)}  ({s:.1%})", va="center", fontsize=9.5)
ax.set_xlim(0, c.total_sales.max() * 1.22)
ax.xaxis.set_major_formatter(KFMT)
ax.grid(axis="x", color=GRID, lw=.8)
ax.set_axisbelow(True)
ax.set_title("Sales distribution across product categories")
save(fig, "02_sales_by_category.png")

# 3.3 region bar
r_ = by_region.sort_values("total_sales")
fig, ax = plt.subplots(figsize=(9, 4))
b = ax.barh(r_.region, r_.total_sales, color=BLUE, height=.6)
for r, v, s in zip(b, r_.total_sales, r_.sales_share):
    ax.text(v + r_.total_sales.max() * .012, r.get_y() + r.get_height() / 2, f"{short(v)}  ({s:.1%})", va="center", fontsize=9.5)
ax.set_xlim(0, r_.total_sales.max() * 1.22)
ax.xaxis.set_major_formatter(KFMT)
ax.grid(axis="x", color=GRID, lw=.8)
ax.set_axisbelow(True)
ax.set_title("Total sales by region")
save(fig, "03_sales_by_region.png")

# 3.4 top products table
t = top_prod.head(10)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.axis("off")
cells = [[int(r["rank"]), r.product_id, r.category, f"{r.total_sales:,.2f}", int(r.units), f"{r.profit:,.2f}"]
         for _, r in t.iterrows()]
tb = ax.table(cellText=cells, colLabels=["#", "Product ID", "Category", "Total sales", "Units", "Profit"],
              loc="center", cellLoc="left", colWidths=[.06, .2, .2, .2, .1, .2])
tb.auto_set_font_size(False)
tb.set_fontsize(10)
tb.scale(1, 1.55)
for (i, j), cell in tb.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.set_facecolor(SURFACE)
    if i == 0:
        cell.set_facecolor("#1c5cab")
        cell.set_text_props(color="white", weight="bold")
    if j in (3, 4, 5) and i > 0:
        cell._loc = "right"
        cell.get_text().set_ha("right")
ax.set_title("Top 10 products by total sales", loc="left")
save(fig, "04_top_10_products_table.png")

# 3.5 region x category heatmap
hm = reg_cat.pivot(index="region", columns="category", values="total_sales").loc[by_region.region, by_cat.category]
fig, ax = plt.subplots(figsize=(10, 4.6))
im = ax.imshow(hm.values, cmap=SEQ, aspect="auto")
ax.set_xticks(range(hm.shape[1]), hm.columns)
ax.set_yticks(range(hm.shape[0]), hm.index)
mx = hm.values.max()
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        ax.text(j, i, short(v), ha="center", va="center", fontsize=9.5, color="white" if v > mx * .45 else INK)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-.5, hm.shape[1]), minor=True)
ax.set_yticks(np.arange(-.5, hm.shape[0]), minor=True)
ax.grid(which="minor", color=SURFACE, lw=2)
ax.tick_params(which="minor", length=0)
ax.set_title("Heatmap: total sales by region and product category")
save(fig, "05_heatmap_region_category.png")

# 3.6 customer segmentation (scatter + share bars)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1]})
for tname, col in zip(TIERS, TIER_COLORS):
    s = cust[cust.segment == tname]
    a1.scatter(s.age, s.total_spend, s=22, color=col, alpha=.85, edgecolor=SURFACE, lw=.6, label=tname)
a1.set_yscale("log")
a1.set_xlabel("Customer age")
a1.set_ylabel("Total spend per customer (log scale)")
a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: short(v, 0)))
a1.grid(color=GRID, lw=.6)
a1.set_axisbelow(True)
a1.legend(title="Segment (by total spend)", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14),
          ncol=4, fontsize=9, title_fontsize=9, markerscale=1.4)
a1.set_title("Customer segmentation by total spend")
x = np.arange(len(TIERS))
w = .38
b1 = a2.bar(x - w / 2, seg.customer_share * 100, w, color="#86b6ef", label="% of customers")
b2 = a2.bar(x + w / 2, seg.sales_share * 100, w, color="#0d366b", label="% of sales")
for bars in (b1, b2):
    for r in bars:
        a2.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.0f}%", ha="center", fontsize=9)
a2.set_xticks(x, TIERS)
a2.set_ylim(0, max(seg.customer_share.max(), seg.sales_share.max()) * 100 * 1.2)
a2.grid(axis="y", color=GRID, lw=.6)
a2.set_axisbelow(True)
a2.legend(frameon=False, loc="upper left")
a2.set_title("Share of customers vs share of sales")
fig.tight_layout()
save(fig, "06_customer_segmentation.png")

# 3.7 quarterly sales
q = by_quarter.copy()
fig, ax = plt.subplots(figsize=(11, 4.6))
cols = [BLUE if not (k in (f"{D0.year}-Q{(D0.month - 1) // 3 + 1}", f"{D1.year}-Q{(D1.month - 1) // 3 + 1}")) else "#86b6ef"
        for k in q.year_quarter]
b = ax.bar(q.year_quarter, q.total_sales, color=cols, width=.62)
for r, v, n in zip(b, q.total_sales, q.orders):
    ax.text(r.get_x() + r.get_width() / 2, v + q.total_sales.max() * .01, short(v), ha="center", fontsize=9)
ax.yaxis.set_major_formatter(KFMT)
ax.grid(axis="y", color=GRID, lw=.8)
ax.set_axisbelow(True)
ax.set_ylim(0, q.total_sales.max() * 1.15)
ax.set_title("Total sales by quarter")
fig.text(0.01, -0.02, "Lighter bars = partial quarters at the start and end of the data.", fontsize=8.5, color=INK2)
save(fig, "07_sales_by_quarter.png")
print("Charts written")

# ======================================================= 4. DASHBOARD DATA
cat_l, reg_l, pay_l = sorted(df.category.unique()), sorted(df.region.unique()), sorted(df.payment_method.unique())
months = sorted(df.year_month.unique())
rec = []
for r in df.itertuples():
    rec.append([months.index(r.year_month), cat_l.index(r.category), reg_l.index(r.region), pay_l.index(r.payment_method),
                r.product_id, r.customer_id, round(r.total_amount, 2), round(r.profit_margin, 2), int(r.quantity),
                int(r.is_returned), int(r.customer_age)])
json.dump({"months": months, "categories": cat_l, "regions": reg_l, "payments": pay_l, "tiers": TIERS,
           "thresholds": [float(q50), float(q80), float(q95)], "rows": rec},
          open(os.path.join(OUT_DASH, "dashboard_data.json"), "w"), separators=(",", ":"))

# ================================================== 5. INSIGHT NUMBERS
ins = {
    "n_orders": len(df), "n_customers": int(df.customer_id.nunique()), "total_sales": total_sales,
    "aov": df.total_amount.mean(), "profit": df.profit_margin.sum(), "profit_pct": df.profit_margin.sum() / total_sales,
    "return_rate": df.is_returned.mean(), "avg_delivery": df.delivery_time_days.mean(),
    "by_region": by_region[["region", "total_sales", "sales_share", "avg_sales", "orders", "total_profit", "profit_pct_of_sales", "return_rate"]].to_dict("records"),
    "by_cat": by_cat[["category", "total_sales", "sales_share", "avg_sales", "orders", "total_profit", "profit_pct_of_sales", "return_rate", "units_sold"]].to_dict("records"),
    "by_year": by_year[["year", "total_sales", "orders"]].to_dict("records"),
    "p12": by_p12[["period12", "total_sales", "orders", "avg_sales", "total_profit"]].to_dict("records"),
    "reg_growth": reg_p12.reset_index().to_dict("records"), "cat_growth": cat_p12.reset_index().to_dict("records"),
    "best_month": m[~m.partial].sort_values("total_sales").iloc[-1][["year_month", "total_sales"]].tolist(),
    "worst_month": m[~m.partial].sort_values("total_sales").iloc[0][["year_month", "total_sales"]].tolist(),
    "by_pay": by_pay[["payment_method", "total_sales", "sales_share", "orders", "return_rate"]].to_dict("records"),
    "by_age": by_age[["age_group", "total_sales", "avg_sales", "orders"]].to_dict("records"),
    "by_gender": by_gender[["customer_gender", "total_sales", "avg_sales", "orders"]].to_dict("records"),
    "segments": seg.to_dict("records"), "top10": top_prod.head(10).to_dict("records"),
    "top10_share": top_prod.head(10).total_sales.sum() / total_sales, "neg_profit_orders": neg,
    "neg_profit_share": neg / len(df), "p99": p99, "high_value_n": hi,
    "high_value_share": df[df.high_value_order == 1].total_amount.sum() / total_sales,
    "repeat_customers": int((cust.orders > 1).sum()), "repeat_share": float((cust.orders > 1).mean()),
    "repeat_sales_share": float(cust[cust.orders > 1].total_spend.sum() / total_sales),
    "discount_profit": df.groupby("discount").agg(orders=("order_id", "count"), avg_profit=("profit_margin", "mean"),
                                                   avg_profit_pct=("profit_pct", "mean")).reset_index().to_dict("records"),
    "delivery_return": df.groupby(pd.cut(df.delivery_time_days, [0, 4, 6, 20], labels=["3-4d", "5-6d", "7d+"]), observed=True)
    .is_returned.mean().to_dict() and {str(k): float(v) for k, v in df.groupby(pd.cut(df.delivery_time_days, [0, 4, 6, 20], labels=["3-4d", "5-6d", "7d+"]), observed=True).is_returned.mean().items()},
    "reg_cat_top": reg_cat.sort_values("total_sales", ascending=False).head(5)[["region", "category", "total_sales"]].to_dict("records"),
    "reg_cat_bottom": reg_cat.sort_values("total_sales").head(5)[["region", "category", "total_sales"]].to_dict("records"),
}
json.dump(ins, open(os.path.join(ROOT, "06_Scripts", "_insights.json"), "w"), indent=1, default=str)
print(json.dumps(ins, indent=1, default=str))
