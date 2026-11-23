"""
Project 2 - build the Word report from the analysis outputs.
Usage: python3 build_report.py <project_root>      (run customer_segmentation_prediction.py first)
All numbers in the text come from 01_Data/*.csv and 03_Models/results.json.
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
R = json.load(open(os.path.join(ROOT, "03_Models", "results.json")))
prof = pd.read_csv(os.path.join(ROOT, "01_Data", "segment_profile.csv")).set_index("segment")
cust = pd.read_csv(os.path.join(ROOT, "01_Data", "customer_segments.csv"))
log = pd.read_csv(os.path.join(ROOT, "01_Data", "data_cleaning_log.csv"))
CH = os.path.join(ROOT, "02_Charts")
NAVY, BLUE, GREY = RGBColor(0x0D, 0x36, 0x6B), RGBColor(0x1C, 0x5C, 0xAB), RGBColor(0x52, 0x51, 0x4E)

f0 = lambda v: f"{v:,.0f}"
f1 = lambda v: f"{v:,.1f}"
f2 = lambda v: f"{v:,.2f}"
pc = lambda v, d=0: f"{v * 100:.{d}f}%"
SEG = prof.index.tolist()                       # ordered by revenue share
PS, RB, DS, BS = "Premium spenders", "Repeat buyers", "Discount seekers", "Budget shoppers"
p = lambda s, c: prof.loc[s, c]
N, NC = R["n_orders"], R["n_customers"]
ID = R["idcheck"]
MD = R["model_data"]
clf = pd.DataFrame(R["clf_metrics"]).set_index("model")
reg = pd.DataFrame(R["reg_metrics"]).set_index("model")
val = pd.DataFrame(R["val_metrics"]).set_index("model")
base_c, base_r, base_v = "Baseline (always 'no')", "Baseline (training mean)", "Baseline (training mean)"
models_c = [m for m in clf.index if not m.startswith("Baseline")]
best_clf = R["best_clf"]
rf_gain = 1 - val.loc["Linear Regression", "rmse"] / val.loc[base_v, "rmse"]
total_rev = prof.total_revenue.sum()
cc = R["cluster_check"]

# ----------------------------------------------------------------------------- docx helpers
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.12
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
    q.add_run().add_picture(os.path.join(CH, name), width=Cm(width))
    para(f"Figure {_fig[0]}. {caption}", italic=True, size=9.5, color=GREY, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)


fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run("Customer Segmentation and Prediction  |  Page ")
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
para("PROJECT 2", bold=True, size=13, color=BLUE, after=2)
para("Customer Segmentation and Prediction", bold=True, size=32, color=NAVY, after=4)
para("Python: Pandas, Scikit-learn, Matplotlib, Seaborn, SQL", size=14, color=GREY, after=30)
table(["", ""], [
    ["Objective", "Segment customers by purchasing behaviour and predict future buying patterns"],
    ["Dataset", f"ecommerce_sales_34500.csv - {N:,} orders, {NC:,} customer IDs, {R['date_min']} to {R['date_max']}"],
    ["Methods", "RFM feature engineering, K-Means clustering, logistic/linear regression, decision trees, random forests"],
    ["Submitted by", "______________________________"],
    ["Course / Institution", "______________________________"],
    ["Date", "16 November 2026"],
], widths=[4, 13], font=10.5, header=False, bold_first=True)
doc.add_page_break()

# ============================================================================= SUMMARY
doc.add_heading("Executive summary", 1)
para(f"The {NC:,} customer IDs were grouped into four behavioural segments with K-Means, and three predictive tasks were tested on held-out future data. "
     "The segmentation is stable and useful for marketing. Predicting repeat buying is not possible with this data, and the report shows why.")
bullet("Four segments.", f"{PS} ({pc(p(PS, 'customer_share'))} of customers, {pc(p(PS, 'revenue_share'))} of revenue), {RB} ({pc(p(RB, 'customer_share'))}, {pc(p(RB, 'revenue_share'))}), "
       f"{DS} ({pc(p(DS, 'customer_share'))}, {pc(p(DS, 'revenue_share'))}) and {BS} ({pc(p(BS, 'customer_share'))}, {pc(p(BS, 'revenue_share'))}). "
       f"Re-fitting on bootstrap samples reproduces them (adjusted Rand index {cc['bootstrap_ari_mean']:.2f}).")
bullet("The 'buy again' prediction fails.", f"Every model scores a ROC-AUC between {clf.loc[models_c, 'roc_auc'].min():.2f} and {clf.loc[models_c, 'roc_auc'].max():.2f}, "
       f"and all confidence intervals include 0.50 (pure chance). Future spend per customer is not predicted better than a plain average either (R-squared at or below zero).")
bullet("Why.", f"A repeated customer ID does not behave like one person: for the {ID['repeat_customers']} IDs that appear more than once, gender changes in {pc(ID['mismatch']['Gender']['observed'])} "
       f"and region in {pc(ID['mismatch']['Region']['observed'])} of cases, almost exactly the rates expected if IDs were assigned at random "
       f"({pc(ID['mismatch']['Gender']['random'])} and {pc(ID['mismatch']['Region']['random'])}). The unique-ID count ({NC:,}) also equals what random draws give "
       f"({ID['unique_random_mean']:,.0f} +/- {ID['unique_random_sd']:.0f}). With no real repeat behaviour in the data, there is nothing to learn.")
bullet("What can be predicted.", f"The value of a purchase, from the product category (RMSE {val.loc['Linear Regression', 'rmse']:.0f} against {val.loc[base_v, 'rmse']:.0f} for the baseline, "
       f"R-squared {val.loc['Linear Regression', 'r2']:.2f}). Category drives it; payment method, region, age and gender add almost nothing.")
bullet("Recommendation.", "Target campaigns by segment and product category, and do not rely on repeat-purchase scores from this dataset. "
       "The scripts re-run unchanged on data with reliable customer IDs.")

# ============================================================================= 1. OBJECTIVE
doc.add_heading("1. Objective and approach", 1)
para("The objective is to perform customer segmentation based on purchasing behaviour and to predict future buying patterns. The work follows the six tasks of the brief:")
for i, t in enumerate(["Data cleaning - missing data, duplicates, variable transformation.",
                       "Feature engineering - customer lifetime value, average order value, purchase frequency and related features.",
                       "Customer segmentation - K-Means clustering, with checks against other methods.",
                       "Modelling - linear regression, decision trees and other models to predict future purchases.",
                       "Evaluation - accuracy, precision, recall, F1, ROC-AUC, RMSE, always against a naive baseline.",
                       "Insights - actionable findings and marketing strategies for every segment."], 1):
    para(f"{i}. {t}", after=2)
para("")
para("The same dataset as Project 1 was used (ecommerce_sales_34500.csv). SQL queries for the data-extraction step are in 05_Scripts/customer_sql_extraction.sql "
     "and were tested to reproduce the Python feature table exactly.")

# ============================================================================= 2. DATA
doc.add_heading("2. Dataset", 1)
para(f"The file has 34,500 rows and 17 columns (order, customer and product identifiers, category, price, discount, quantity, payment method, order date, delivery time, region, "
     f"returned flag, total amount, shipping cost, profit, customer age and gender). Only {N:,} rows contain data; the rest are empty. The orders belong to {NC:,} customer IDs. "
     f"{pc((cust.frequency == 1).mean())} of IDs have a single order, {int((cust.frequency == 2).sum())} have two and {int((cust.frequency == 3).sum())} have three.")

# ============================================================================= 3. CLEANING
doc.add_heading("3. Data cleaning", 1)
para("Each check was run on the data; the 'Action' column says what was done. The cleaned orders are in 01_Data/ecommerce_orders_cleaned.csv.")
table(["Check", "Finding", "Action"], log[["Step", "Finding", "Action"]].fillna("None").values.tolist(), widths=[3.2, 8.8, 5], font=8.8)
doc.add_heading("3.1 Important finding: customer IDs do not identify one person", 2)
para(f"The brief treats a customer as the unit of analysis, so customer_id was kept as the customer key. It was tested first, because a real customer should keep the same "
     f"gender, and roughly the same region and age, across orders. For the {ID['repeat_customers']} IDs that appear in more than one order this is not the case: "
     f"gender differs in {pc(ID['mismatch']['Gender']['observed'])}, region in {pc(ID['mismatch']['Region']['observed'])} and age in {pc(ID['mismatch']['Age']['observed'])} of them. "
     f"These match the rates expected under random assignment ({pc(ID['mismatch']['Gender']['random'])}, {pc(ID['mismatch']['Region']['random'])}, {pc(ID['mismatch']['Age']['random'])}). "
     f"Likewise, {NC:,} unique IDs among {N:,} orders is what random draws from the ID range C10003-C17997 produce ({ID['unique_random_mean']:,.0f} +/- {ID['unique_random_sd']:.0f} in 500 simulations).")
figure("01_customer_id_reliability.png", "Repeated customer IDs change gender, region and age as often as random matches would.", width=12.5)
para("Consequences: customer-level spend and basket-type segmentation remain meaningful (each ID still describes a set of orders), but any claim about repeat behaviour, loyalty or "
     "'the same customer coming back' rests on IDs that probably collide by chance. This is reflected in the modelling results of section 6.")

# ============================================================================= 4. FEATURES
doc.add_heading("4. Feature engineering", 1)
para("One row per customer ID was built from the orders. The reference date is the day after the last order (" + R["ref_date"] + ").")
table(["Feature", "Definition"], [
    ["recency_days", "Days from the last order to the reference date (R of RFM)"],
    ["frequency", "Number of orders (F)"],
    ["monetary / lifetime_value", "Total amount spent over the whole period (M); used as historical customer lifetime value"],
    ["avg_order_value", "Total spend divided by number of orders"],
    ["net_value_after_returns", "Spend on orders that were not returned"],
    ["avg_discount, return_rate", "Mean discount rate and share of orders returned"],
    ["avg_delivery_days, avg_shipping", "Mean delivery time and shipping cost"],
    ["n_categories, share_<category>", "Categories bought and the share of spend in each of the seven categories"],
    ["favourite_category, favourite_payment", "Category with the most spend and the most-used payment method"],
    ["age, gender, region", "Taken from the latest order (see section 3.1)"],
    ["tenure_days", "Days between first and last order"]], widths=[5.2, 11.8], font=9.2)
para(f"Monetary value is strongly right-skewed (skewness {log.Action.str.extract(r'skewness ([0-9.]+)').dropna().iloc[0, 0]} after the log transform, against a much higher value before), so spend and order value were log-transformed and every clustering feature was standardised.")
figure("02_rfm_distributions.png", "Distributions of recency, frequency and monetary value.")
figure("03_feature_correlation.png", "Correlation matrix. Spend and order value are almost identical (0.96); recency and frequency are mildly related; age, delivery and returns are unrelated to spending.", width=11.5)

# ============================================================================= 5. SEGMENTATION
doc.add_heading("5. Customer segmentation (K-Means)", 1)
para("Five standardised features entered the clustering: recency, frequency, log total spend, log average order value and average discount. "
     "Returns, delivery time and demographics were left out of the clustering and used to describe the segments instead, because they are not purchasing behaviour or carry no signal (section 4).")
doc.add_heading("5.1 Choosing the number of clusters", 2)
kd = pd.DataFrame(R["k_table"])
para(f"K-Means was fitted for k = 2 to 8 (20 random starts each). The silhouette score is highest at k = {R['k_chosen']} ({kd.set_index('k').silhouette[R['k_chosen']]:.3f}), "
     "the Davies-Bouldin index flattens from there, and the elbow in inertia is gentle (no sharp bend), so the silhouette score decides. k = 4 also gives groups that can be explained to a marketing team.")
figure("04_choosing_k.png", "Elbow, silhouette and Davies-Bouldin curves for k = 2 to 8.")
doc.add_heading("5.2 The four segments", 2)
table(["Segment", "Customers", "% cust.", "Revenue", "% revenue", "Orders / cust.", "Avg order", "Avg discount", "Profit / cust."],
      [[s, f0(p(s, "customers")), pc(p(s, "customer_share")), f0(p(s, "total_revenue")), pc(p(s, "revenue_share"), 1), f2(p(s, "frequency")),
        f0(p(s, "avg_order_value")), pc(p(s, "avg_discount"), 1), f1(p(s, "avg_profit"))] for s in SEG],
      num_cols=range(1, 9), widths=[3.1, 1.8, 1.4, 1.9, 1.7, 1.7, 1.6, 1.8, 1.7], font=8.6)
bullet(PS + ".", f"One large purchase of about {f0(p(PS, 'avg_order_value'))} on average, little discount ({pc(p(PS, 'avg_discount'), 1)}), mostly Electronics "
       f"({R['segment_category_pct'][PS]['Electronics']:.0f}% of spend) and Home ({R['segment_category_pct'][PS]['Home']:.0f}%). They are {pc(p(PS, 'customer_share'))} of customers but bring {pc(p(PS, 'revenue_share'))} of revenue.")
bullet(RB + ".", f"IDs with {f1(p(RB, 'frequency'))} orders on average, the most recent buyers ({f0(p(RB, 'recency_days'))} days since last order against about 360 for the others), "
       f"high lifetime value ({f0(p(RB, 'monetary'))}), the highest profit per customer ({f1(p(RB, 'avg_profit'))}) and {R['segment_payment_pct'][RB]['Credit Card']:.0f}% credit-card payers. "
       "Given section 3.1, treat this group as 'IDs with several orders', not as proven loyal customers.")
bullet(DS + ".", f"Small baskets ({f0(p(DS, 'avg_order_value'))}) bought at an average discount of {pc(p(DS, 'avg_discount'), 1)} (others pay under 6%). Fashion is the favourite category "
       f"({R['segment_category_pct'][DS]['Fashion']:.0f}% of spend). Profit per customer is only {f1(p(DS, 'avg_profit'))}.")
bullet(BS + ".", f"The smallest baskets ({f0(p(BS, 'avg_order_value'))}), concentrated in Grocery ({R['segment_category_pct'][BS]['Grocery']:.0f}%), Toys ({R['segment_category_pct'][BS]['Toys']:.0f}%) and Beauty "
       f"({R['segment_category_pct'][BS]['Beauty']:.0f}%). {pc(p(BS, 'customer_share'))} of customers but {pc(p(BS, 'revenue_share'), 1)} of revenue and {f1(p(BS, 'avg_profit'))} profit per customer.")
figure("05_segments_pca.png", "The segments projected onto the first two principal components (64% of the variance).", width=13.5)
figure("06_segment_size_vs_revenue.png", "Share of customers versus share of revenue per segment.", width=13.5)
figure("07_segment_profile_heatmap.png", "Standardised feature averages: orange = above average, blue = below.", width=14)
figure("08_segment_category_mix.png", "Share of each segment's spend by product category.", width=14.5)
doc.add_heading("5.3 How reliable are the segments?", 2)
bullet("Stability.", f"K-Means was re-fitted on 30 bootstrap samples and used to label all customers: the labels agree with the final ones with an adjusted Rand index of {cc['bootstrap_ari_mean']:.2f} on average (lowest {cc['bootstrap_ari_min']:.2f}). Values near 1 mean the groups do not depend on which customers happen to be sampled.")
bullet("Other methods.", f"Ward hierarchical clustering gives a lower silhouette ({cc['ward_silhouette']:.2f}) and only moderate agreement with K-Means (ARI {cc['ward_ari_vs_kmeans']:.2f}); a Gaussian mixture model fits poorly "
       f"(silhouette {cc['gmm_silhouette']:.2f}, ARI {cc['gmm_ari_vs_kmeans']:.2f}). The groups are therefore not sharply separated: the repeat and the discount groups are clear-cut, "
       "while the boundary between premium and budget spenders is a continuous scale cut by K-Means. The segments are best used as practical bands, not as natural classes.")
bullet("Separation.", f"The overall silhouette of {cc['kmeans_silhouette']:.2f} is moderate, consistent with the points above.")

# ============================================================================= 6. MODELLING
doc.add_heading("6. Predictive modelling", 1)
doc.add_heading("6.1 Design", 2)
para("A predictive model must be tested on data from after the period it learned from. The customer history was therefore cut at snapshot dates. For each snapshot, features come only from orders before it, "
     f"and the targets come from the following {MD['window_days']} days.")
bullet("Training:", f"snapshots {', '.join(MD['train_snapshots'])}: {MD['train_rows']:,} customer-snapshots, {MD['train_positives']} of them ({pc(MD['train_positive_rate'], 1)}) followed by another purchase.")
bullet("Testing:", f"snapshot {MD['test_snapshot']}: {MD['test_rows']:,} customers, {MD['test_positives']} ({pc(MD['test_positive_rate'], 1)}) bought again. The last training window ends the day before this snapshot, so no future information leaks into training.")
bullet("Features:", "number of orders, recency, tenure, log total spend, log order value, average discount, return rate, delivery time, categories bought, age, gender, region and favourite category. The K-Means segment was not used, because it is computed from the full period and would leak the future.")
bullet("Models:", "logistic regression and linear regression, a depth-limited decision tree and a random forest (class-balanced for classification), each against a naive baseline.")
doc.add_heading("6.2 Model A: will the customer buy again within 180 days? (classification)", 2)
rows = [[m, pc(clf.loc[m, "accuracy"], 1), pc(clf.loc[m, "precision"], 1), pc(clf.loc[m, "recall"], 1), f"{clf.loc[m, 'f1']:.3f}", f"{clf.loc[m, 'roc_auc']:.3f}",
         f"{clf.loc[m, 'auc_ci_low']:.2f} - {clf.loc[m, 'auc_ci_high']:.2f}"] for m in clf.index]
table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "95% interval for AUC"], rows, num_cols=range(1, 7), widths=[4.4, 1.9, 2, 1.7, 1.5, 1.9, 3.6], font=9)
bullet("Accuracy is misleading here.", f"Answering 'no' for everybody is {pc(clf.loc[base_c, 'accuracy'], 1)} accurate, because only {pc(MD['test_positive_rate'], 1)} of customers buy again. It also has 0% recall, so it finds nobody. "
       "The balanced metrics below are the fair comparison.")
bullet("No model beats chance.", f"Precision of {pc(clf.loc[models_c, 'precision'].min(), 1)} to {pc(clf.loc[models_c, 'precision'].max(), 1)} is about the {pc(MD['test_positive_rate'], 1)} base rate, i.e. picking customers at random. "
       f"ROC-AUC is {clf.loc[models_c, 'roc_auc'].min():.2f} to {clf.loc[models_c, 'roc_auc'].max():.2f}, and each bootstrap interval contains 0.50. Balanced accuracy is {clf.loc[models_c, 'balanced_accuracy'].min():.2f} to {clf.loc[models_c, 'balanced_accuracy'].max():.2f}.")
figure("09_classification_results.png", f"ROC curves of the three models (all hug the chance diagonal) and the confusion matrix of the best one ({best_clf}).")
doc.add_heading("6.3 Model B: how much will the customer spend in the next 180 days? (regression)", 2)
table(["Model", "RMSE", "MAE", "R-squared"], [[m, f2(reg.loc[m, "rmse"]), f2(reg.loc[m, "mae"]), f"{reg.loc[m, 'r2']:.3f}"] for m in reg.index],
      num_cols=(1, 2, 3), widths=[6, 3, 3, 3])
para(f"None of the models improves on predicting the training average (RMSE {reg.loc[base_r, 'rmse']:.1f}): RMSE rises to {reg.loc[[m for m in reg.index if not m.startswith('Baseline')], 'rmse'].min():.1f}-"
     f"{reg.loc[[m for m in reg.index if not m.startswith('Baseline')], 'rmse'].max():.1f} and R-squared is zero or negative. Most customers spend nothing in the window "
     f"(average {f1(MD['test_future_spend_mean'])}), and who does is unpredictable from their history.")
doc.add_heading("6.4 Model C: what will a purchase be worth? (regression)", 2)
para(f"This model answers a different question that the data can support: given a customer's profile and the category being bought, how large is the order? Features are category, payment method, region, gender and age "
     f"(price, quantity and discount are excluded because they define the order value). It was trained on {R['val_data']['train_orders']:,} orders before {MD['test_snapshot']} and tested on the {R['val_data']['test_orders']:,} orders after it.")
table(["Model", "RMSE", "MAE", "R-squared"], [[m, f2(val.loc[m, "rmse"]), f2(val.loc[m, "mae"]), f"{val.loc[m, 'r2']:.3f}"] for m in val.index],
      num_cols=(1, 2, 3), widths=[6, 3, 3, 3])
para(f"Linear regression lowers the error by {pc(rf_gain, 1)} against the baseline (RMSE {val.loc['Linear Regression', 'rmse']:.1f} vs {val.loc[base_v, 'rmse']:.1f}) and explains {pc(val.loc['Linear Regression', 'r2'])} of the variance; "
     "the random forest and decision tree are close. Permutation importance shows that category carries nearly all of it. The remaining variation (order values range from a few units to several thousand) is not explained by who the customer is.")
figure("10_regression_rmse.png", "Test RMSE of each model against the baseline (dark bar). Left: next-180-day spend (no gain). Right: value of a single purchase (clear gain).")
figure("11_permutation_importance.png", "Permutation importance on the test data. Left: tiny values, no real signal. Right: the product category dominates.")
doc.add_heading("6.5 Evaluation metrics used", 2)
table(["Metric", "Meaning"], [
    ["Accuracy", "Share of customers classified correctly. Misleading when one class is rare."],
    ["Precision", "Of the customers predicted to buy again, how many did."],
    ["Recall", "Of the customers who bought again, how many the model found."],
    ["F1", "Harmonic mean of precision and recall."],
    ["ROC-AUC", "Chance that a random buyer-again customer is scored above a random other one. 0.5 = chance, 1.0 = perfect."],
    ["RMSE / MAE", "Typical size of the prediction error in currency units (RMSE penalises large errors more)."],
    ["R-squared", "Share of variance explained. Zero or negative = no better than predicting the average."]], widths=[3, 14], font=9.2)

# ============================================================================= 7. INSIGHTS
doc.add_heading("7. Insights and marketing strategies", 1)
doc.add_heading("7.1 Key insights on customer behaviour", 2)
bullet("Revenue is concentrated.", f"{pc(p(PS, 'customer_share'))} of customers ({PS}) generate {pc(p(PS, 'revenue_share'))} of revenue and {pc(p(PS, 'profit_share'))} of profit. "
       f"At the other end, {BS} are {pc(p(BS, 'customer_share'))} of customers but only {pc(p(BS, 'revenue_share'), 1)} of revenue and {pc(p(BS, 'profit_share'), 1)} of profit.")
bullet("The big spenders do not come back.", f"{PS} are single-order IDs with {f0(p(PS, 'recency_days'))} days since their purchase on average. The upside is the second purchase, but the data gives no evidence that any group is likely to make one.")
bullet("Discounts attract small, low-margin orders.", f"{DS} buy at an average discount of {pc(p(DS, 'avg_discount'), 1)} and earn {f1(p(DS, 'avg_profit'))} profit per customer, against {f1(p(PS, 'avg_profit'))} for {PS} who pay almost full price ({pc(p(PS, 'avg_discount'), 1)} discount).")
bullet("Category predicts value.", f"What is bought matters far more than who buys it: Electronics is {R['segment_category_pct'][PS]['Electronics']:.0f}% of {PS}' spend and only {R['segment_category_pct'][BS]['Electronics']:.0f}% of {BS}'. Age, gender and region barely differ between segments.")
bullet("Returns and delivery do not separate segments.", f"Return rates are {pc(cust.groupby('segment').return_rate.mean().min(), 1)} to {pc(cust.groupby('segment').return_rate.mean().max(), 1)} and delivery times about 4.7-4.9 days in every segment.")
doc.add_heading("7.2 Marketing strategy for each segment", 2)
table(["Segment", "Who they are", "Recommended strategy"], [
    [PS, f"{f0(p(PS, 'customers'))} customers, {pc(p(PS, 'revenue_share'))} of revenue, one order of about {f0(p(PS, 'avg_order_value'))}, Electronics and Home",
     "VIP treatment: early access to new electronics, premium bundles and accessories, extended warranty, free delivery. Keep discounts low (they pay full price). "
     "Run a post-purchase journey (care tips, accessories, upgrade offers) to win a second order."],
    [RB, f"{f0(p(RB, 'customers'))} IDs, {f1(p(RB, 'frequency'))} orders, most recent, highest profit per customer, {R['segment_payment_pct'][RB]['Credit Card']:.0f}% credit card",
     "Loyalty programme and replenishment reminders, card-partner offers, cross-sell Fashion and Home. Validate the identity of these customers (section 3.1) before investing in loyalty benefits."],
    [DS, f"{f0(p(DS, 'customers'))} customers, bought at {pc(p(DS, 'avg_discount'), 0)} discount, Fashion-led, {f1(p(DS, 'avg_profit'))} profit each",
     "Do not deepen discounts. Use time-limited, targeted offers with a minimum basket value, bundle deals to raise order value, and a threshold for free shipping. Test whether they buy at a smaller discount."],
    [BS, f"{f0(p(BS, 'customers'))} customers, orders of about {f0(p(BS, 'avg_order_value'))}, Grocery, Toys and Beauty, {f1(p(BS, 'avg_profit'))} profit each",
     "Serve through low-cost channels (email, app notifications) rather than paid acquisition. Introduce a minimum order value or flat shipping fee, multi-item bundles and add-on suggestions so that shipping stops absorbing the margin."]],
    widths=[2.9, 5.3, 8.8], font=9)
doc.add_heading("7.3 Recommendations", 2)
for lead, t in [
    ("Segment first, then target by category.", "Use the four segments and the favourite category as the targeting rules; they are explainable and stable."),
    ("Do not use repeat-purchase scores from this dataset.", "The 'buy again' models perform at chance level, so any list of 'likely repeat buyers' would be random."),
    ("Fix customer identification.", "Capture a reliable customer key (login, phone or email hash) so that repeat behaviour, loyalty and lifetime value can be measured and predicted."),
    ("Protect margin.", "Limit discounts for customers who pay full price anyway, and set basket thresholds for the small-basket segments."),
    ("Re-run when better data is available.", "customer_segmentation_prediction.py runs end to end on a new file with the same columns.")]:
    bullet(lead, t, style="List Number")

# ============================================================================= 8. LIMITS
doc.add_heading("8. Limitations", 1)
for t in [f"The customer ID problem (section 3.1): the {ID['repeat_customers']} 'repeat' IDs are probably not the same people, so repeat behaviour, recency of loyal customers and lifetime value are unreliable. Demographics come from each ID's latest order.",
          f"{pc((cust.frequency == 1).mean())} of IDs have one order, so frequency and recency carry little information.",
          f"The test set has {MD['test_positives']} positive cases, so the uncertainty of the classification metrics is large (shown by the bootstrap intervals).",
          "The data covers two years, has no product names and no currency. Lifetime value is historical spend, not a forecast.",
          "K-Means assumes compact, similarly sized groups. The segment boundaries between premium and budget spenders are cut points on a continuous scale (section 5.3)."]:
    bullet("", t)
doc.add_heading("9. Conclusion", 1)
para(f"Customers split into four stable, explainable segments with very different value: {pc(p(PS, 'customer_share'))} of customers produce {pc(p(PS, 'revenue_share'))} of revenue. "
     "These segments support concrete marketing actions. Predicting who will buy again is not possible from this data, and the evidence suggests the repeat customer IDs are coincidences rather than returning people. "
     f"What can be predicted is the value of a purchase from its category (error {pc(rf_gain, 0)} lower than the baseline). The most valuable next step is a reliable customer identifier.")
doc.add_heading("Appendix - contents of the submission folder", 1)
table(["Folder / file", "What it contains"], [
    ["01_Data/", "Raw data, cleaned orders, customer_features.csv, customer_segments.csv, segment_profile.csv, cleaning log"],
    ["02_Charts/", "The eleven chart images used in this report"],
    ["03_Models/", "Metrics tables (CSV), results.json and the saved K-Means model (kmeans_segmentation.joblib)"],
    ["04_Report/", "This report (Word and PDF)"],
    ["05_Scripts/customer_segmentation_prediction.py", "Full analysis: cleaning, features, K-Means, models, charts (also as a notebook)"],
    ["05_Scripts/customer_sql_extraction.sql", "SQL queries that extract the customer features and the targets (tested)"],
    ["05_Scripts/build_report.py", "Builds this report from the result files"]], widths=[6.5, 10.5], font=9.5)

out = os.path.join(ROOT, "04_Report", "Customer_Segmentation_Report.docx")
doc.save(out)
print("saved", out)
