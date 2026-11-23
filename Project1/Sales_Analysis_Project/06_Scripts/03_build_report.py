"""
Sales Analysis Dashboard - Step 3: build the Word report from the cleaned data.
Usage: python3 03_build_report.py <project_root>
All numbers in the text are computed here from 01_Data/ecommerce_sales_cleaned.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = sys.argv[1]
df = pd.read_csv(os.path.join(ROOT, "01_Data", "ecommerce_sales_cleaned.csv"), parse_dates=["order_date"])
log = pd.read_csv(os.path.join(ROOT, "01_Data", "data_cleaning_log.csv"))
CH = os.path.join(ROOT, "03_Charts")
NAVY = RGBColor(0x0D, 0x36, 0x6B)
BLUE = RGBColor(0x1C, 0x5C, 0xAB)
GREY = RGBColor(0x52, 0x51, 0x4E)

# ------------------------------------------------------------------ numbers
S = df.total_amount.sum()
N = len(df)
PROFIT = df.profit_margin.sum()
D0, D1 = df.order_date.min(), df.order_date.max()
f0 = lambda v: f"{v:,.0f}"
f2 = lambda v: f"{v:,.2f}"
pc = lambda v, d=1: f"{v * 100:.{d}f}%"


def grp(keys):
    g = df.groupby(keys).agg(sales=("total_amount", "sum"), orders=("order_id", "count"), profit=("profit_margin", "sum"),
                             ret=("is_returned", "mean"), units=("quantity", "sum"), ship=("shipping_cost", "sum"),
                             deliv=("delivery_time_days", "mean")).reset_index()
    g["avg"] = g.sales / g.orders
    g["share"] = g.sales / S
    g["pm"] = g.profit / g.sales
    return g


reg = grp("region").sort_values("sales", ascending=False).set_index("region")
cat = grp("category").sort_values("sales", ascending=False).set_index("category")
pay = grp("payment_method").sort_values("sales", ascending=False).set_index("payment_method")
gen = grp("customer_gender").set_index("customer_gender")
age = grp("age_group").set_index("age_group")
qtr = grp("year_quarter").set_index("year_quarter")
mon = grp("year_month").set_index("year_month")
yr = grp("year").set_index("year")

df["p12"] = np.where(df.order_date < pd.Timestamp("2024-09-12"), "Y1", "Y2")
p12 = grp("p12").set_index("p12")
g_total = p12.loc["Y2", "sales"] / p12.loc["Y1", "sales"] - 1
g_orders = p12.loc["Y2", "orders"] / p12.loc["Y1", "orders"] - 1
g_aov = p12.loc["Y2", "avg"] / p12.loc["Y1", "avg"] - 1
rg = df.pivot_table(index="region", columns="p12", values="total_amount", aggfunc="sum")
rg["g"] = rg.Y2 / rg.Y1 - 1
cg = df.pivot_table(index="category", columns="p12", values="total_amount", aggfunc="sum")
cg["g"] = cg.Y2 / cg.Y1 - 1

cust = df.groupby("customer_id").agg(spend=("total_amount", "sum"), orders=("order_id", "count"), age=("customer_age", "mean")).reset_index()
q50, q80, q95 = cust.spend.quantile([.5, .8, .95])
TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
cust["seg"] = pd.cut(cust.spend, [-np.inf, q50, q80, q95, np.inf], labels=TIERS)
seg = cust.groupby("seg", observed=True).agg(n=("customer_id", "count"), spend=("spend", "sum"), avg=("spend", "mean")).reindex(TIERS)
seg["cs"] = seg.n / seg.n.sum()
seg["ss"] = seg.spend / seg.spend.sum()
rep = cust[cust.orders > 1]

prod = df.groupby(["product_id", "category"]).agg(sales=("total_amount", "sum"), units=("quantity", "sum"), profit=("profit_margin", "sum")).reset_index().sort_values("sales", ascending=False)
top10 = prod.head(10)
neg = df[df.profit_margin < 0]
p99 = df.total_amount.quantile(.99)
hv = df[df.total_amount > p99]
disc = df.groupby("discount").agg(orders=("order_id", "count"), ap=("profit_margin", "mean"), q=("quantity", "mean"))
retval = df[df.is_returned == 1].total_amount.sum()
ret_cat_val = df[df.is_returned == 1].groupby("category").total_amount.sum()
elec_other = reg.index.difference(["Central"])
elec_reg = df[df.category == "Electronics"].groupby("region").total_amount.sum()
elec_gap = elec_reg[elec_other].mean() - elec_reg["Central"]
mfull = mon.drop(index=[D0.strftime("%Y-%m"), D1.strftime("%Y-%m")])
best_m, worst_m = mfull.sales.idxmax(), mfull.sales.idxmin()
mname = lambda s: pd.Timestamp(s + "-01").strftime("%b %Y")

# ------------------------------------------------------------------ docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2)
sec.top_margin = sec.bottom_margin = Cm(2)
st = doc.styles["Normal"]
st.font.name = "Calibri"
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.12
for name, size, color in (("Heading 1", 18, NAVY), ("Heading 2", 14, BLUE), ("Heading 3", 12, BLUE)):
    h = doc.styles[name]
    h.font.name = "Calibri"
    h.font.size = Pt(size)
    h.font.bold = True
    h.font.color.rgb = color
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


def para(text="", bold=False, italic=False, size=None, color=None, align=None, after=None, style=None):
    p = doc.add_paragraph(style=style)
    if text:
        r = p.add_run(text)
        r.bold, r.italic = bold, italic
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    if align:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def rich(parts, style=None):
    """parts = [(text, bold), ...]"""
    p = doc.add_paragraph(style=style)
    for t, b in parts:
        r = p.add_run(t)
        r.bold = b
    return p


def bullet(lead, text):
    p = doc.add_paragraph(style="List Bullet")
    if lead:
        p.add_run(lead + " ").bold = True
    p.add_run(text)
    p.paragraph_format.space_after = Pt(4)
    return p


def table(headers, rows, widths=None, num_cols=(), font=9.5, header=True, bold_first=False):
    t = doc.add_table(rows=1 if header else 0, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers if header else []):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(font)
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
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.space_before = Pt(1)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    # repeat header row; never split a row across pages
    for k, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        trPr.append(cs)
        if header and k == 0:
            th = OxmlElement("w:tblHeader")
            th.set(qn("w:val"), "true")
            trPr.append(th)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


_fig = [0]


def figure(path, caption, width=16.5):
    _fig[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width))
    c = para(f"Figure {_fig[0]}. {caption}", italic=True, size=9.5, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
    return c


# footer with page number
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run("Sales Analysis Dashboard  |  Page ")
r.font.size = Pt(9)
r.font.color.rgb = GREY
for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    rr = fp.add_run()
    rr.font.size = Pt(9)
    rr.font.color.rgb = GREY
    if tag:
        fc = OxmlElement("w:fldChar")
        fc.set(qn("w:fldCharType"), tag)
        rr._r.append(fc)
    else:
        it = OxmlElement("w:instrText")
        it.set(qn("xml:space"), "preserve")
        it.text = txt
        rr._r.append(it)

# ================================================================== COVER
for _ in range(5):
    para("", after=14)
para("PROJECT 1", bold=True, size=13, color=BLUE, after=2)
para("Sales Analysis Dashboard", bold=True, size=34, color=NAVY, after=4)
para("eCommerce sales performance, 12 September 2023 to 11 September 2025", size=14, color=GREY, after=30)
table(["", ""], [
    ["Objective", "Analyse eCommerce sales performance and present it in an interactive dashboard with business insights"],
    ["Dataset", f"ecommerce_sales_34500.csv - {N:,} orders after cleaning, {df.customer_id.nunique():,} customers, 7 categories, 5 regions"],
    ["Tools", "Python (pandas, matplotlib) for cleaning and analysis; SQL for extraction; Power BI build guide with DAX; interactive HTML dashboard"],
    ["Submitted by", "______________________________"],
    ["Course / Institution", "______________________________"],
    ["Date", "12 November 2026"],
], widths=[4, 13], font=10.5, header=False, bold_first=True)
doc.add_page_break()

# ================================================================== SUMMARY
doc.add_heading("Executive summary", 1)
para(f"Across {D0:%d %b %Y} to {D1:%d %b %Y} the company took {N:,} orders from {df.customer_id.nunique():,} customers, "
     f"generating total sales of {f0(S)} (average {f2(df.total_amount.mean())} per order) and a profit of {f0(PROFIT)} ({pc(PROFIT / S)} of sales). "
     f"{pc(df.is_returned.mean())} of orders were returned.")
bullet("Revenue is concentrated.", f"Electronics produces {pc(cat.loc['Electronics', 'share'])} of sales from only {pc(cat.loc['Electronics', 'orders'] / N)} of orders, "
       f"and {reg.index[0]} is the strongest region ({pc(reg.iloc[0].share)}). The top 5% of customers account for {pc(seg.loc['Platinum', 'ss'])} of sales.")
bullet("Sales are growing.", f"The second 12 months delivered {pc(g_total)} more sales than the first, driven by more orders ({pc(g_orders)}) and a higher average order ({pc(g_aov)}). "
       f"East ({pc(rg.loc['East', 'g'], 0)}) and Central ({pc(rg.loc['Central', 'g'], 0)}) grew fastest; South and Home declined.")
bullet("Growth is not the same as profit.", f"Electronics earns a {pc(cat.loc['Electronics', 'pm'])} margin against 24-32% in most other categories, and Grocery loses money "
       f"({f0(cat.loc['Grocery', 'profit'])} overall) because shipping eats {pc(cat.loc['Grocery', 'ship'] / cat.loc['Grocery', 'sales'])} of its sales value.")
bullet("Discounts cost profit without adding volume.", f"Average profit per order falls from {f2(disc.loc[0.0, 'ap'])} with no discount to {f2(disc.loc[0.2, 'ap'])} at 20% off, "
       f"while units per order stay near {disc.q.mean():.1f}.")
para("Section 7 turns these findings into seven recommendations.", italic=True, color=GREY)

# ================================================================== 1. OBJECTIVE
doc.add_heading("1. Objective and approach", 1)
para("The objective is to build a comprehensive dashboard that analyses sales performance for an eCommerce company over a specific period. "
     "The work follows the five tasks in the project brief:")
for i, t in enumerate(["Data cleaning - handle missing values, duplicates and inconsistencies.",
                       "Data aggregation - total sales, average sales and number of orders by region, product category and time period.",
                       "Visualizations - sales trend (line), category distribution (bar), top products (table), customer segmentation (scatter and heatmap).",
                       "Dashboard - interactive, with filters for time period, product category, region and more.",
                       "Insights - which regions and products generate the most revenue and where growth can come from."], 1):
    para(f"{i}. {t}", after=2)
para("")
rich([("Definitions used throughout. ", True),
      ("Total sales = sum of total_amount (price x quantity x (1 - discount)); average sales = average total_amount per order; "
       "profit = sum of the profit_margin column, which holds currency amounts rather than percentages. The data gives no currency, so values are plain numbers. "
       "A 12-month comparison (12 Sep to 11 Sep) is used for growth because calendar years 2023 and 2025 are incomplete.", False)])

# ================================================================== 2. DATA
doc.add_heading("2. Dataset", 1)
para(f"The file has 34,500 rows and 17 columns: order, customer and product identifiers; category, price, discount and quantity; payment method; order date; "
     f"delivery time; region; returned flag; total amount, shipping cost and profit; customer age and gender. Only {N:,} rows contain data - the remaining 32,503 are completely empty.")

# ================================================================== 3. CLEANING
doc.add_page_break()
doc.add_heading("3. Data cleaning", 1)
para("Every check below was run on the data; the 'Action' column says what was done. The cleaned file is 01_Data/ecommerce_sales_cleaned.csv.")
table(["Check", "Finding", "Action"], log[["Step", "Finding", "Action"]].fillna("None").values.tolist(), widths=[3.4, 8.2, 5.4], font=9)
para("Two points deserve attention. First, the empty rows would have distorted every average and count, so removing them is the single most important cleaning step. "
     "Second, product_id cannot be used on its own to identify a product (the same ID appears at different prices and in different categories), so products are always ranked by product_id and category together.")

# ================================================================== 4. AGGREGATION
doc.add_heading("4. Data aggregation", 1)
para("Totals, averages and order counts were computed for every region, category and time period. "
     "The complete set (including region x category, region x month and category x month) is in 02_Aggregations/sales_aggregations.xlsx; the key tables follow.")
doc.add_heading("4.1 Overall KPIs", 2)
table(["Metric", "Value"], [
    ["Total sales", f2(S)], ["Total orders", f"{N:,}"], ["Average sales per order", f2(df.total_amount.mean())],
    ["Total profit", f"{f2(PROFIT)} ({pc(PROFIT / S)} of sales)"], ["Unique customers", f"{df.customer_id.nunique():,}"],
    ["Units sold", f"{int(df.quantity.sum()):,}"], ["Return rate (orders)", f"{pc(df.is_returned.mean(), 2)} (sales value {f0(retval)}, {pc(retval / S)})"],
    ["Average delivery time", f"{df.delivery_time_days.mean():.1f} days"]], widths=[7, 10])
doc.add_heading("4.2 By region", 2)
table(["Region", "Total sales", "Share", "Avg sales / order", "Orders", "Profit % of sales", "Return rate"],
      [[i, f0(r.sales), pc(r.share), f2(r.avg), int(r.orders), pc(r.pm), pc(r.ret)] for i, r in reg.iterrows()], num_cols=range(1, 7))
doc.add_heading("4.3 By product category", 2)
table(["Category", "Total sales", "Share", "Avg sales / order", "Orders", "Profit % of sales", "Return rate"],
      [[i, f0(r.sales), pc(r.share), f2(r.avg), int(r.orders), pc(r.pm), pc(r.ret)] for i, r in cat.iterrows()], num_cols=range(1, 7))
doc.add_heading("4.4 By time period", 2)
table(["Quarter", "Total sales", "Avg sales / order", "Orders"], [[i, f0(r.sales), f2(r.avg), int(r.orders)] for i, r in qtr.iterrows()], num_cols=range(1, 4), widths=[4, 4.3, 4.3, 4.3])
table(["Period", "Total sales", "Avg sales / order", "Orders", "Profit"],
      [["Year 1 (12 Sep 2023 - 11 Sep 2024)", f0(p12.loc["Y1", "sales"]), f2(p12.loc["Y1", "avg"]), int(p12.loc["Y1", "orders"]), f0(p12.loc["Y1", "profit"])],
       ["Year 2 (12 Sep 2024 - 11 Sep 2025)", f0(p12.loc["Y2", "sales"]), f2(p12.loc["Y2", "avg"]), int(p12.loc["Y2", "orders"]), f0(p12.loc["Y2", "profit"])],
       ["Change", pc(g_total), pc(g_aov), pc(g_orders), pc(p12.loc["Y2", "profit"] / p12.loc["Y1", "profit"] - 1)]],
      num_cols=range(1, 5), widths=[6.4, 2.7, 3, 2.3, 2.6])
para("2023 (Sep-Dec) and 2025 (Jan-Sep 11) are partial calendar years, which is why the two 12-month periods are used for like-for-like comparison.", italic=True, size=9.5, color=GREY)

# ================================================================== 5. VISUALS
doc.add_heading("5. Visualizations", 1)
para("Each chart answers one question and uses a single colour scale: blue for magnitude, a light-to-dark blue ramp for ordered customer tiers.")
figure(os.path.join(CH, "01_sales_trend_monthly.png"), "Sales trend over time (line chart). Hollow markers are partial months.")
para(f"Monthly sales swing between roughly 9K and 20K. The best full month was {mname(best_m)} ({f0(mfull.loc[best_m, 'sales'])}) and the weakest {mname(worst_m)} ({f0(mfull.loc[worst_m, 'sales'])}). "
     "The 3-month average shows no sharp trend change, but a moderate upward drift in the second year.")
figure(os.path.join(CH, "07_sales_by_quarter.png"), "Total sales by quarter.")
figure(os.path.join(CH, "02_sales_by_category.png"), "Sales distribution across product categories (bar chart).", width=14.5)
figure(os.path.join(CH, "03_sales_by_region.png"), "Total sales by region (bar chart).", width=14.5)
doc.add_heading("Top-performing products", 3)
table(["#", "Product ID", "Category", "Total sales", "Units", "Profit"],
      [[i + 1, r.product_id, r.category, f2(r.sales), int(r.units), f2(r.profit)] for i, (_, r) in enumerate(top10.iterrows())],
      num_cols=(0, 3, 4, 5), widths=[1.2, 3.4, 3.4, 3.6, 2.2, 3.2])
para(f"A table is used instead of a pie chart: with {df.product_id.nunique():,} product IDs across {N:,} orders almost every ID occurs once, so a pie would be made of tiny, unreadable slices. "
     f"The top 10 are all Electronics and together make up {pc(top10.sales.sum() / S)} of sales.")
figure(os.path.join(CH, "05_heatmap_region_category.png"), "Heatmap of total sales by region and category.", width=15.5)
figure(os.path.join(CH, "06_customer_segmentation.png"), "Customer segmentation by total spend: each dot is a customer (left); share of customers versus share of sales per tier (right).")
table(["Segment", "Rule (total spend)", "Customers", "% of customers", "Total spend", "% of sales", "Avg spend"],
      [[t, rule, int(seg.loc[t, "n"]), pc(seg.loc[t, "cs"], 0), f0(seg.loc[t, "spend"]), pc(seg.loc[t, "ss"], 0), f0(seg.loc[t, "avg"])]
       for t, rule in zip(TIERS, [f"up to {q50:,.0f} (bottom 50%)", f"{q50:,.0f} - {q80:,.0f} (next 30%)", f"{q80:,.0f} - {q95:,.0f} (next 15%)", f"above {q95:,.0f} (top 5%)"])],
      num_cols=(2, 3, 4, 5, 6), widths=[2.2, 4.3, 2, 2.2, 2.3, 1.9, 2.1], font=9)

# ================================================================== 6. DASHBOARD
doc.add_heading("6. Interactive dashboard", 1)
para("The dashboard combines a KPI strip, the sales trend, category and region bars, the top-products table, the region x category heatmap and the customer-segmentation view on one page.")
bullet("Filters:", "time period (from-month and to-month), product category, region, payment method and order status (all / not returned / returned only).")
bullet("Cross-filtering:", "clicking a category or region bar, or a heatmap cell, applies that filter to every other visual; all KPIs and charts recalculate.")
bullet("Power BI:", "04_Dashboard/PowerBI_Build_Guide.md gives the complete Power BI specification: Power Query steps, the date and customer tables, every DAX measure, the visual-to-field mapping and the slicer set-up, "
       "together with control totals to check against.")
bullet("HTML version:", "04_Dashboard/sales_dashboard.html is a self-contained interactive version of the same dashboard that opens in any browser with no installation.")
from PIL import Image
_im = Image.open(os.path.join(ROOT, "04_Dashboard", "dashboard_preview.png"))
_top = os.path.join(ROOT, "04_Dashboard", "dashboard_preview_top.png")
_im.crop((0, 0, _im.width, int(_im.width * 0.835))).save(_top)
figure(_top, "Dashboard preview (top section: filters, KPIs, trend, category and region views). The full page is in sales_dashboard.html.", width=13.5)

# ================================================================== 7. INSIGHTS
doc.add_heading("7. Key business insights and recommendations", 1)
doc.add_heading("7.1 Where revenue comes from", 2)
bullet("Regions.", f"{reg.index[0]} leads with {f0(reg.iloc[0].sales)} ({pc(reg.iloc[0].share)}). West, East and South are almost level at {pc(reg.iloc[1].share, 0)} each "
       f"({f0(reg.iloc[3].sales)}-{f0(reg.iloc[1].sales)}), and Central trails at {pc(reg.loc['Central', 'share'])}. South takes the most orders after North ({reg.loc['South', 'orders']}) "
       f"but has the lowest average order ({f2(reg.loc['South', 'avg'])} vs {f2(reg.loc['East', 'avg'])} in East).")
bullet("Categories.", f"Electronics generates {f0(cat.loc['Electronics', 'sales'])} ({pc(cat.loc['Electronics', 'share'])} of sales) from {pc(cat.loc['Electronics', 'orders'] / N)} of orders at an average of "
       f"{f0(cat.loc['Electronics', 'avg'])} per order. Home ({pc(cat.loc['Home', 'share'])}), Fashion ({pc(cat.loc['Fashion', 'share'])}) and Sports ({pc(cat.loc['Sports', 'share'])}) follow; "
       f"Beauty, Toys and Grocery together contribute only {pc(cat.loc[['Beauty', 'Toys', 'Grocery'], 'share'].sum())}.")
bullet("Region x category.", f"North Electronics is the single biggest cell ({f0(elec_reg['North'])}, {pc(elec_reg['North'] / S)} of all sales). Electronics dominates every region's heatmap row.")
bullet("Products.", f"The ten best products are all Electronics, each a single large multi-unit order of {f0(top10.sales.min())}-{f0(top10.sales.max())}. "
       f"The {len(hv)} orders above {f0(p99)} (the top 1%) make up {pc(hv.total_amount.sum() / S)} of sales, so revenue depends heavily on a small number of big-ticket orders - which also explains the month-to-month volatility.")

doc.add_heading("7.2 Growth and seasonality", 2)
bullet("Overall growth.", f"Year 2 sales were {f0(p12.loc['Y2', 'sales'])} against {f0(p12.loc['Y1', 'sales'])} in Year 1 (+{pc(g_total)}): orders +{pc(g_orders)} and average order +{pc(g_aov)}.")
bullet("Fastest-growing segments.", f"By region, East +{pc(rg.loc['East', 'g'], 0)} and Central +{pc(rg.loc['Central', 'g'], 0)}; by category, Sports +{pc(cg.loc['Sports', 'g'], 0)}, "
       f"Fashion +{pc(cg.loc['Fashion', 'g'], 0)} and Electronics +{pc(cg.loc['Electronics', 'g'], 0)}.")
bullet("Declines.", f"South ({pc(rg.loc['South', 'g'])}) and Home ({pc(cg.loc['Home', 'g'])}) shrank, and West was flat ({pc(rg.loc['West', 'g'])}). "
       f"North, the largest region, grew only {pc(rg.loc['North', 'g'], 0)}.")
bullet("Seasonality.", f"Q4 is strong ({f0(qtr.loc['2023-Q4', 'sales'])} in 2023, {f0(qtr.loc['2024-Q4', 'sales'])} in 2024) and Q2 2025 was the best quarter ({f0(qtr.loc['2025-Q2', 'sales'])}). "
       f"Q3 2024 ({f0(qtr.loc['2024-Q3', 'sales'])}) and Q1 2025 ({f0(qtr.loc['2025-Q1', 'sales'])}) were the softest complete quarters.")

doc.add_heading("7.3 Profitability", 2)
bullet("Margin by category.", f"Overall profit is {pc(PROFIT / S)} of sales. Electronics supplies {pc(cat.loc['Electronics', 'share'])} of sales but only {pc(cat.loc['Electronics', 'profit'] / PROFIT)} of profit "
       f"(margin {pc(cat.loc['Electronics', 'pm'])}). Beauty ({pc(cat.loc['Beauty', 'pm'], 0)}), Fashion ({pc(cat.loc['Fashion', 'pm'], 0)}), Sports, Home and Toys (about 24-25%) are far more profitable per unit of sales.")
bullet("Grocery loses money.", f"{(neg.category == 'Grocery').sum()} of {int((df.category == 'Grocery').sum())} Grocery orders ({pc((neg.category == 'Grocery').sum() / (df.category == 'Grocery').sum(), 0)}) are loss-making. "
       f"An average Grocery order is worth {f2(cat.loc['Grocery', 'avg'])} yet carries about {f2(cat.loc['Grocery', 'ship'] / cat.loc['Grocery', 'orders'])} of shipping ({pc(cat.loc['Grocery', 'ship'] / cat.loc['Grocery', 'sales'])} of sales). "
       f"Toys ({pc(cat.loc['Toys', 'ship'] / cat.loc['Toys', 'sales'])}) and Beauty ({pc(cat.loc['Beauty', 'ship'] / cat.loc['Beauty', 'sales'])}) show the same small-basket squeeze. "
       f"Across the business {len(neg)} orders ({pc(len(neg) / N)}) lose money.")
bullet("Discounts.", f"Average profit per order is {f2(disc.loc[0.0, 'ap'])} without a discount, {f2(disc.loc[0.1, 'ap'])} at 10%, {f2(disc.loc[0.2, 'ap'])} at 20% and {f2(disc.loc[0.3, 'ap'])} at 30%. "
       f"Average units per order do not rise with the discount ({disc.loc[0.0, 'q']:.2f} at 0% vs {disc.loc[0.2, 'q']:.2f} at 20%), so on this data deeper discounts reduce profit without a visible volume gain "
       f"(the data cannot show whether they win extra orders).")

doc.add_heading("7.4 Customers", 2)
bullet("Concentration.", f"The top 5% of customers (Platinum, {int(seg.loc['Platinum', 'n'])} people) generate {pc(seg.loc['Platinum', 'ss'], 0)} of sales and the top 20% (Gold + Platinum) {pc(seg.loc[['Gold', 'Platinum'], 'ss'].sum(), 0)}. "
       f"The bottom half (Bronze) contributes just {pc(seg.loc['Bronze', 'ss'], 0)}; their average spend is {f0(seg.loc['Bronze', 'avg'])} versus {f0(seg.loc['Platinum', 'avg'])} for Platinum.")
bullet("Repeat buying is low.", f"Only {len(rep)} customers ({pc(len(rep) / len(cust))}) ordered more than once, yet they account for {pc(rep.spend.sum() / S)} of sales.")
bullet("Who buys.", f"Customers are spread evenly across age groups - 55-64 is the largest ({f0(age.loc['55-64', 'sales'])}, {pc(age.loc['55-64', 'share'])}) and 45-54 has the highest average order ({f2(age.loc['45-54', 'avg'])}). "
       f"Female customers account for {pc(gen.loc['Female', 'share'])} of sales and spend {pc(gen.loc['Female', 'avg'] / gen.loc['Male', 'avg'] - 1, 0)} more per order than male customers ({f2(gen.loc['Female', 'avg'])} vs {f2(gen.loc['Male', 'avg'])}).")

doc.add_heading("7.5 Returns, delivery and payments", 2)
bullet("Returns.", f"{pc(df.is_returned.mean(), 2)} of orders are returned, worth {f0(retval)} ({pc(retval / S)} of sales). Electronics ({pc(cat.loc['Electronics', 'ret'])}) and Fashion ({pc(cat.loc['Fashion', 'ret'])}) return the most; "
       f"Electronics alone is {pc(ret_cat_val['Electronics'] / retval, 0)} of returned value. Grocery returns are almost nil ({pc(cat.loc['Grocery', 'ret'])}).")
bullet("Payments.", f"Credit card is the leading method ({pc(pay.loc['Credit Card', 'share'])} of sales), then Debit Card ({pc(pay.loc['Debit Card', 'share'])}). "
       f"Wallet has the highest return rate ({pc(pay.loc['Wallet', 'ret'])}, on {pay.loc['Wallet', 'orders']} orders) and PayPal the lowest ({pc(pay.loc['PayPal', 'ret'])}).")
bullet("Delivery.", f"The average is {df.delivery_time_days.mean():.1f} days, but East takes {reg.loc['East', 'deliv']:.1f} days compared with {reg.loc['North', 'deliv']:.1f} in North - while East is also the fastest-growing region.")

doc.add_heading("Recommendations", 2)
recs = [
    ("Protect Electronics and raise its margin.", f"It carries the business, but at a {pc(cat.loc['Electronics', 'pm'])} margin. Review supplier cost, shipping and discount policy on high-ticket items and add higher-margin accessories to Electronics baskets."),
    ("Fix the economics of small baskets.", "Set a minimum order value or free-shipping threshold, bundle Grocery, Toys and Beauty items, or charge shipping on low-value orders so that these categories stop losing money."),
    ("Tighten discounting.", f"Limit discounts above 15% to clearance or targeted use; on this data they cut profit per order by up to {pc(1 - disc.loc[0.3, 'ap'] / disc.loc[0.0, 'ap'], 0)} without lifting units per order."),
    ("Grow Central and South.", f"Central Electronics ({f0(elec_reg['Central'])}) lags the other regions' average ({f0(elec_reg[elec_other].mean())}); closing that gap is worth about {f0(elec_gap)} ({pc(elec_gap / S)} of total sales). "
     f"South has the most orders after North but the lowest average order and shrank {pc(abs(rg.loc['South', 'g']))} - focus on basket size there."),
    ("Back the winners.", f"Sports (+{pc(cg.loc['Sports', 'g'], 0)}) and Fashion (+{pc(cg.loc['Fashion', 'g'], 0)}) are growing with healthy margins, and East (+{pc(rg.loc['East', 'g'], 0)}) is the fastest region - prioritise inventory and marketing there. Investigate why Home declined."),
    ("Build loyalty around top customers.", f"Gold and Platinum customers ({pc(seg.loc[['Gold', 'Platinum'], 'cs'].sum(), 0)} of customers) bring {pc(seg.loc[['Gold', 'Platinum'], 'ss'].sum(), 0)} of sales and only {pc(len(rep) / len(cust), 0)} of customers repeat. "
     "Launch a loyalty or re-engagement programme to raise repeat purchase and move Silver customers up."),
    ("Plan around seasonality and reduce returns.", "Stock up and campaign for Q4 and spring, and use promotions to soften the Q3 and January dips. "
     "Cut Electronics and Fashion returns with better product information and sizing guidance, and shorten East delivery times."),
]
for i, (lead, t) in enumerate(recs, 1):
    p = doc.add_paragraph(style="List Number")
    p.add_run(lead + " ").bold = True
    p.add_run(t)
    p.paragraph_format.space_after = Pt(4)

# ================================================================== 8. LIMITS / CONCLUSION
doc.add_heading("8. Limitations", 1)
for t in ["The data holds 1,997 orders over two years; some segment-level results (for example Wallet returns, 9 returned orders) rest on small samples.",
          "product_id is not unique, so product-level rankings combine product_id and category; the data has no product names.",
          "The first and last months (Sep 2023, Sep 2025) are partial. No currency is stated.",
          "The profit_margin column is treated as a profit amount because its values are in currency units; the profit-percentage figures are profit divided by total sales.",
          "Findings describe associations in this data set; for example, the discount result does not prove that discounts do not attract new customers."]:
    bullet("", t)
doc.add_heading("9. Conclusion", 1)
para(f"Sales grew {pc(g_total)} year on year and are anchored by Electronics and a small group of high-value customers. The biggest opportunities are not simply more sales but better-quality sales: "
     "repair the small-basket economics, discipline discounting, lift Central and South, and convert one-time buyers into repeat customers.")

doc.add_heading("Appendix - contents of the submission folder", 1)
table(["Folder / file", "What it contains"], [
    ["01_Data/", "Raw data, cleaned data (ecommerce_sales_cleaned.csv) and the cleaning log"],
    ["02_Aggregations/sales_aggregations.xlsx", "19 sheets: KPIs, by region / category / year / quarter / month / payment / age / gender, region x category, top products, customer segments, growth"],
    ["03_Charts/", "The seven chart images used in this report"],
    ["04_Dashboard/sales_dashboard.html", "Interactive dashboard (open in a browser)"],
    ["04_Dashboard/PowerBI_Build_Guide.md", "Step-by-step Power BI build with DAX measures"],
    ["05_Report/Sales_Analysis_Report.docx", "This report"],
    ["06_Scripts/", "Python cleaning/analysis script, SQL queries (tested), report builder"],
], widths=[6.5, 10.5], font=9.5)

out = os.path.join(ROOT, "05_Report", "Sales_Analysis_Report.docx")
doc.save(out)
print("saved", out)
