"""
Project 2 - Customer Segmentation and Prediction (Python)

Usage:  python3 customer_segmentation_prediction.py <project_root>

Reads    <root>/01_Data/ecommerce_sales_raw.csv
Writes   01_Data/  ecommerce_orders_cleaned.csv, customer_features.csv, customer_segments.csv,
                   data_cleaning_log.csv, segment_profile.csv
         02_Charts/*.png
         03_Models/ model_metrics_*.csv, kmeans_segmentation.joblib, results.json

The script is organised in numbered cells (# %%) so it can also be run cell by cell in VS Code,
Spyder or Jupyter.
"""
# %% 0. Setup ---------------------------------------------------------------------------------
import json
import os
import sys
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, average_precision_score, balanced_accuracy_score,
                             calinski_harabasz_score, confusion_matrix, davies_bouldin_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
SEED = 42
ROOT = sys.argv[1]
P = {k: os.path.join(ROOT, v) for k, v in
     dict(data="01_Data", charts="02_Charts", models="03_Models").items()}
RAW = os.path.join(P["data"], "ecommerce_sales_raw.csv")

# Chart style: validated categorical palette, fixed slot order (blue, orange, aqua, yellow, magenta)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUE, ORANGE = CAT[0], CAT[1]
MARKERS = ["o", "s", "^", "D", "P"]
sns.set_theme(style="whitegrid", rc={
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "grid.color": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 10, "legend.frameon": False})
DIV = sns.diverging_palette(250, 30, l=60, center="light", as_cmap=True)   # blue <-> orange, neutral middle
SEQ = sns.light_palette("#1c5cab", as_cmap=True)


def save(fig, name):
    fig.savefig(os.path.join(P["charts"], name), dpi=170, bbox_inches="tight")
    plt.close(fig)


def short(v):
    return f"{v / 1e3:.1f}K" if abs(v) >= 1e3 else f"{v:.0f}"


R = {}          # results collected for the report
LOG = []        # cleaning log

# %% 1. Data cleaning -------------------------------------------------------------------------
raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
LOG.append(("Load raw file", f"{len(raw):,} rows x {raw.shape[1]} columns", "None"))

blank = raw.apply(lambda r: "".join(r).strip() == "", axis=1)
df = raw[~blank].copy()
LOG.append(("Fully empty rows", f"{int(blank.sum()):,} rows contain no data at all", f"Removed -> {len(df):,} orders remain"))

df = df.replace(r"^\s*$", np.nan, regex=True)
LOG.append(("Missing values", f"{int(df.isna().sum().sum())} missing cells in the remaining rows", "Nothing to impute"))

for c in ["price", "discount", "total_amount", "shipping_cost", "profit_margin"]:
    df[c] = pd.to_numeric(df[c])
for c in ["quantity", "delivery_time_days", "customer_age"]:
    df[c] = pd.to_numeric(df[c]).astype(int)
for c in ["order_id", "customer_id", "product_id", "category", "payment_method", "region", "returned", "customer_gender"]:
    df[c] = df[c].str.strip()
df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y")
LOG.append(("Variable types", "order_date text (dd-mm-yyyy) -> date; numbers parsed; text trimmed", "Converted"))

n_dup = int(df.duplicated().sum() + df.order_id.duplicated().sum())
df = df.drop_duplicates().drop_duplicates("order_id")
LOG.append(("Duplicates", f"{n_dup} duplicated rows / repeated order_id", "Nothing removed"))

calc = (df.price * df.quantity * (1 - df.discount)).round(2)
LOG.append(("Consistency", f"{int(((calc - df.total_amount).abs() > 0.05).sum())} orders where total != price x qty x (1 - discount); "
            f"{int(((df.price <= 0) | (df.quantity <= 0) | (df.shipping_cost < 0)).sum())} invalid price/quantity/shipping values",
            "Totals reconcile; nothing to fix"))

df["is_returned"] = (df.returned == "Yes").astype(int)
df["profit_pct"] = df.profit_margin / df.total_amount
df = df.sort_values(["order_date", "order_id"]).reset_index(drop=True)
REF = df.order_date.max() + pd.Timedelta(days=1)       # "today" for recency
R["n_orders"], R["n_customers"] = len(df), int(df.customer_id.nunique())
R["date_min"], R["date_max"], R["ref_date"] = str(df.order_date.min().date()), str(df.order_date.max().date()), str(REF.date())

# --- customer_id reliability test: are repeated IDs really the same person? -------------------
ids = df.customer_id.str[1:].astype(int)
span = int(ids.max() - ids.min() + 1)
rng = np.random.default_rng(SEED)
sim_unique = [len(np.unique(rng.integers(0, span, len(df)))) for _ in range(500)]
rep_ids = df.groupby("customer_id").size()
rep_ids = rep_ids[rep_ids > 1].index
g = df[df.customer_id.isin(rep_ids)].groupby("customer_id")
chk = {}
for col, lab in (("customer_gender", "Gender"), ("region", "Region"), ("customer_age", "Age")):
    p = df[col].value_counts(normalize=True)
    chk[lab] = dict(observed=float((g[col].nunique() > 1).mean()), random=float(1 - (p ** 2).sum()))
R["idcheck"] = dict(id_span=span, unique_observed=R["n_customers"], unique_random_mean=float(np.mean(sim_unique)),
                    unique_random_sd=float(np.std(sim_unique)), repeat_customers=int(len(rep_ids)), mismatch=chk)
LOG.append(("customer_id reliability",
            f"{len(rep_ids)} IDs appear in more than one order, but the 'same' customer changes gender in {chk['Gender']['observed']:.0%}, "
            f"region in {chk['Region']['observed']:.0%} and age in {chk['Age']['observed']:.0%} of cases - almost exactly what random ID collisions would give "
            f"({chk['Gender']['random']:.0%} / {chk['Region']['random']:.0%} / {chk['Age']['random']:.0%}); unique IDs {R['n_customers']:,} vs "
            f"{np.mean(sim_unique):,.0f} +/- {np.std(sim_unique):.0f} expected from random draws over {span:,} possible IDs",
            "Kept customer_id as the customer key (as the brief requires); demographics taken from the latest order; limitation reported"))

fig, ax = plt.subplots(figsize=(8.4, 4.2))
lab = list(chk)
x = np.arange(len(lab))
b1 = ax.bar(x - .2, [chk[k]["observed"] * 100 for k in lab], .38, color=BLUE, label="Observed for repeated customer IDs")
b2 = ax.bar(x + .2, [chk[k]["random"] * 100 for k in lab], .38, color=ORANGE, label="Expected if IDs were assigned at random")
for bars in (b1, b2):
    for r_ in bars:
        ax.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + 1.5, f"{r_.get_height():.0f}%", ha="center", fontsize=9)
ax.set_xticks(x, [f"{k} differs" for k in lab])
ax.set_ylim(0, 118)
ax.set_ylabel("% of repeated customer IDs")
ax.set_title("Does a repeated customer ID mean the same person?")
ax.legend(loc="upper left", ncol=1, fontsize=9)
save(fig, "01_customer_id_reliability.png")

# %% 2. Feature engineering (one row per customer) ---------------------------------------------
o = df.sort_values("order_date")
cust = o.groupby("customer_id").agg(
    frequency=("order_id", "count"), units=("quantity", "sum"), monetary=("total_amount", "sum"),
    total_profit=("profit_margin", "sum"), avg_discount=("discount", "mean"), return_rate=("is_returned", "mean"),
    avg_delivery_days=("delivery_time_days", "mean"), avg_shipping=("shipping_cost", "mean"),
    n_categories=("category", "nunique"), first_order=("order_date", "min"), last_order=("order_date", "max"))
cust["recency_days"] = (REF - cust.last_order).dt.days
cust["tenure_days"] = (cust.last_order - cust.first_order).dt.days
cust["avg_order_value"] = cust.monetary / cust.frequency                    # average order value
cust["lifetime_value"] = cust.monetary                                      # historical customer lifetime value
net = o[o.is_returned == 0].groupby("customer_id").total_amount.sum()
cust["net_value_after_returns"] = net.reindex(cust.index).fillna(0)
last = o.groupby("customer_id").last()                                      # demographics from the latest order
cust["age"], cust["gender"], cust["region"] = last.customer_age, last.customer_gender, last.region
fav_cat = (o.groupby(["customer_id", "category"]).total_amount.sum().reset_index()
           .sort_values(["customer_id", "total_amount"], ascending=[True, False]).drop_duplicates("customer_id").set_index("customer_id").category)
fav_pay = (o.groupby(["customer_id", "payment_method"]).size().rename("n").reset_index()
           .sort_values(["customer_id", "n"], ascending=[True, False]).drop_duplicates("customer_id").set_index("customer_id").payment_method)
cust["favourite_category"], cust["favourite_payment"] = fav_cat, fav_pay
share = o.pivot_table(index="customer_id", columns="category", values="total_amount", aggfunc="sum", fill_value=0)
share = share.div(share.sum(axis=1), axis=0).add_prefix("share_")
cust = cust.join(share)
cust["log_monetary"], cust["log_aov"] = np.log1p(cust.monetary), np.log1p(cust.avg_order_value)
skew_before, skew_after = float(cust.monetary.skew()), float(cust.log_monetary.skew())
LOG.append(("Feature engineering", "recency, frequency, monetary (RFM), average order value, lifetime value, net value after returns, "
            "average discount, return rate, delivery/shipping, categories bought, favourite category and payment, category spend shares, "
            "age/gender/region of latest order", f"{cust.shape[1]} columns for {len(cust):,} customers"))
LOG.append(("Variable transformation", f"monetary and order value are right-skewed (skewness {skew_before:.1f})",
            f"log1p transform (skewness {skew_after:.1f}); features standardised (z-score) before clustering; categories one-hot encoded for models"))
cust.reset_index().to_csv(os.path.join(P["data"], "customer_features.csv"), index=False)
df.assign(order_date=df.order_date.dt.strftime("%Y-%m-%d")).to_csv(os.path.join(P["data"], "ecommerce_orders_cleaned.csv"), index=False)

# EDA charts
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
sns.histplot(cust.recency_days, bins=30, color=BLUE, ax=axs[0], edgecolor=SURFACE)
axs[0].set(title="Recency (days since last order)", xlabel="days", ylabel="customers")
fc = cust.frequency.value_counts().sort_index()
axs[1].bar(fc.index.astype(str), fc.values, color=BLUE, width=.6)
for i, v in enumerate(fc.values):
    axs[1].text(i, v + 15, f"{v:,}", ha="center", fontsize=9)
axs[1].set(title="Frequency (orders per customer ID)", xlabel="orders", ylabel="customers")
sns.histplot(cust.monetary, bins=np.logspace(0, np.log10(cust.monetary.max()), 30), color=BLUE, ax=axs[2], edgecolor=SURFACE)
axs[2].set(xscale="log", title="Monetary (total spend, log scale)", xlabel="total spend", ylabel="customers")
fig.tight_layout()
save(fig, "02_rfm_distributions.png")

CL_FEATURES = ["recency_days", "frequency", "log_monetary", "log_aov", "avg_discount"]
corr_cols = ["recency_days", "frequency", "monetary", "avg_order_value", "avg_discount", "return_rate", "avg_delivery_days", "age"]
fig, ax = plt.subplots(figsize=(7.4, 5.6))
sns.heatmap(cust[corr_cols].corr(), annot=True, fmt=".2f", cmap=DIV, center=0, vmin=-1, vmax=1, linewidths=2,
            linecolor=SURFACE, cbar_kws={"shrink": .8}, ax=ax, annot_kws={"size": 8.5})
ax.set_title("Correlation between customer features")
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
save(fig, "03_feature_correlation.png")

# %% 3. Customer segmentation: K-Means -----------------------------------------------------------
X = cust[CL_FEATURES]
scaler = StandardScaler().fit(X)
Z = scaler.transform(X)
rows = []
for k in range(2, 9):
    km = KMeans(k, n_init=20, random_state=SEED).fit(Z)
    rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(Z, km.labels_),
                     davies_bouldin=davies_bouldin_score(Z, km.labels_), calinski_harabasz=calinski_harabasz_score(Z, km.labels_)))
kdf = pd.DataFrame(rows)
K = int(kdf.loc[kdf.silhouette.idxmax(), "k"])
R["k_table"], R["k_chosen"] = kdf.round(4).to_dict("records"), K

fig, axs = plt.subplots(1, 3, figsize=(13, 3.7))
for a, col, ttl in zip(axs, ["inertia", "silhouette", "davies_bouldin"],
                       ["Elbow: inertia (lower = tighter)", "Silhouette (higher = better)", "Davies-Bouldin (lower = better)"]):
    a.plot(kdf.k, kdf[col], color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5)
    a.axvline(K, color=ORANGE, lw=1.6, ls="--")
    a.set(title=ttl, xlabel="number of clusters k")
    a.set_xticks(kdf.k)
axs[1].text(K + .1, kdf.silhouette.min(), f"chosen k = {K}", color=ORANGE, fontsize=9.5, va="bottom")
fig.tight_layout()
save(fig, "04_choosing_k.png")

km = KMeans(K, n_init=50, random_state=SEED).fit(Z)
cust["cluster"] = km.labels_
# stability: refit on 30 bootstrap samples, predict everyone, compare with the final labels
aris = []
for b in range(30):
    idx = rng.integers(0, len(Z), len(Z))
    kb = KMeans(K, n_init=10, random_state=b).fit(Z[idx])
    aris.append(adjusted_rand_score(km.labels_, kb.predict(Z)))
agg = AgglomerativeClustering(K, linkage="ward").fit_predict(Z)
gmm = GaussianMixture(K, covariance_type="full", n_init=5, random_state=SEED).fit(Z).predict(Z)
R["cluster_check"] = dict(
    kmeans_silhouette=float(silhouette_score(Z, km.labels_)), bootstrap_ari_mean=float(np.mean(aris)), bootstrap_ari_min=float(np.min(aris)),
    ward_silhouette=float(silhouette_score(Z, agg)), ward_ari_vs_kmeans=float(adjusted_rand_score(km.labels_, agg)),
    gmm_silhouette=float(silhouette_score(Z, gmm)), gmm_ari_vs_kmeans=float(adjusted_rand_score(km.labels_, gmm)))

# profile and name the segments
prof = cust.groupby("cluster").agg(
    customers=("frequency", "size"), recency_days=("recency_days", "mean"), frequency=("frequency", "mean"),
    monetary=("monetary", "mean"), avg_order_value=("avg_order_value", "mean"), avg_discount=("avg_discount", "mean"),
    return_rate=("return_rate", "mean"), avg_profit=("total_profit", "mean"), avg_age=("age", "mean"),
    pct_female=("gender", lambda s: (s == "Female").mean()), total_revenue=("monetary", "sum"))
names = {}
for cid, r_ in prof.iterrows():
    if r_.frequency > 1.5:
        names[cid] = "Repeat buyers"
    elif r_.avg_discount > 0.10:
        names[cid] = "Discount seekers"
rest = prof.drop(index=list(names)).sort_values("avg_order_value", ascending=False).index.tolist()
if len(rest) == 2:
    names[rest[0]], names[rest[1]] = "Premium spenders", "Budget shoppers"
else:
    for cid in rest:
        names[cid] = f"Segment {cid}"
cust["segment"] = cust.cluster.map(names)
prof["segment"] = prof.index.map(names)
prof["customer_share"] = prof.customers / prof.customers.sum()
prof["revenue_share"] = prof.total_revenue / prof.total_revenue.sum()
prof["profit_share"] = prof.avg_profit * prof.customers / (prof.avg_profit * prof.customers).sum()
prof["top_region"] = [cust[cust.cluster == c].region.mode()[0] for c in prof.index]
prof["top_category"] = [cust[cust.cluster == c].favourite_category.mode()[0] for c in prof.index]
prof["top_payment"] = [cust[cust.cluster == c].favourite_payment.mode()[0] for c in prof.index]
prof = prof.sort_values("revenue_share", ascending=False)
ORDER = prof.segment.tolist()
COL = {s: CAT[i] for i, s in enumerate(ORDER)}
MK = {s: MARKERS[i] for i, s in enumerate(ORDER)}
prof.set_index("segment").round(4).to_csv(os.path.join(P["data"], "segment_profile.csv"))
R["segments"] = prof.round(4).to_dict("records")
cust.reset_index().to_csv(os.path.join(P["data"], "customer_segments.csv"), index=False)
joblib.dump(dict(scaler=scaler, kmeans=km, features=CL_FEATURES, names=names, ref_date=str(REF.date())),
            os.path.join(P["models"], "kmeans_segmentation.joblib"))

# segmentation charts
pc = PCA(2, random_state=SEED).fit(Z)
Z2 = pc.transform(Z)
fig, ax = plt.subplots(figsize=(8.6, 6))
for s in ORDER:
    m = (cust.segment == s).values
    ax.scatter(Z2[m, 0], Z2[m, 1], s=26, color=COL[s], marker=MK[s], alpha=.8, edgecolor=SURFACE, lw=.5,
               label=f"{s} ({m.sum():,})")
ax.set(xlabel=f"Principal component 1 ({pc.explained_variance_ratio_[0]:.0%} of variance)",
       ylabel=f"Principal component 2 ({pc.explained_variance_ratio_[1]:.0%} of variance)",
       title=f"K-Means customer segments (k = {K}) projected to 2 dimensions")
ax.legend(loc="best", fontsize=9, markerscale=1.5)
save(fig, "05_segments_pca.png")

fig, ax = plt.subplots(figsize=(9.4, 4.4))
x = np.arange(len(prof))
b1 = ax.bar(x - .2, prof.customer_share * 100, .38, color=BLUE, label="% of customers")
b2 = ax.bar(x + .2, prof.revenue_share * 100, .38, color=ORANGE, label="% of revenue")
for bars in (b1, b2):
    for r_ in bars:
        ax.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + .8, f"{r_.get_height():.0f}%", ha="center", fontsize=9.5)
ax.set_xticks(x, [f"{s}\n({int(n):,} customers)" for s, n in zip(prof.segment, prof.customers)])
ax.set_ylim(0, max(prof.customer_share.max(), prof.revenue_share.max()) * 100 * 1.18)
ax.set_ylabel("share (%)")
ax.set_title("Segment size versus revenue contribution")
ax.legend(loc="upper right")
save(fig, "06_segment_size_vs_revenue.png")

zprof = pd.DataFrame(Z, columns=CL_FEATURES, index=cust.index).groupby(cust.segment).mean().loc[ORDER]
zprof.columns = ["Recency (days)", "Frequency", "Total spend (log)", "Order value (log)", "Avg discount"]
fig, ax = plt.subplots(figsize=(8.6, 3.9))
sns.heatmap(zprof, annot=True, fmt="+.2f", cmap=DIV, center=0, vmin=-2, vmax=2, linewidths=2, linecolor=SURFACE,
            cbar_kws={"label": "std. deviations from average", "shrink": .85}, ax=ax)
ax.set_ylabel("")
ax.set_title("What defines each segment (standardised feature averages)")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
save(fig, "07_segment_profile_heatmap.png")

cm_ = cust.groupby("segment")[[c for c in cust.columns if c.startswith("share_")]].mean().loc[ORDER]
cm_.columns = [c.replace("share_", "") for c in cm_.columns]
fig, ax = plt.subplots(figsize=(9.4, 3.9))
sns.heatmap(cm_ * 100, annot=True, fmt=".0f", cmap=SEQ, linewidths=2, linecolor=SURFACE, ax=ax,
            cbar_kws={"label": "% of segment's spend", "shrink": .85})
ax.set_ylabel("")
ax.set_title("Product category preference by segment (% of spend)")
save(fig, "08_segment_category_mix.png")

demo = pd.crosstab(cust.segment, cust.region, normalize="index").loc[ORDER] * 100
R["segment_region_pct"] = demo.round(1).to_dict("index")
R["segment_gender_pct"] = (pd.crosstab(cust.segment, cust.gender, normalize="index").loc[ORDER] * 100).round(1).to_dict("index")
R["segment_payment_pct"] = (pd.crosstab(cust.segment, cust.favourite_payment, normalize="index").loc[ORDER] * 100).round(1).to_dict("index")
R["segment_category_pct"] = (cm_ * 100).round(1).to_dict("index")
print("Segmentation done:", prof[["segment", "customers", "revenue_share"]].to_string(index=False))

# %% 4. Predictive modelling -------------------------------------------------------------------
WINDOW = 180                                           # predict the next 180 days (about 6 months)
TRAIN_SNAPS = pd.to_datetime(["2023-12-12", "2024-03-12", "2024-06-12", "2024-09-12"])
TEST_SNAP = pd.Timestamp("2025-03-12")                 # train labels end 2025-03-11, before the test snapshot -> no leakage
assert TRAIN_SNAPS.max() + pd.Timedelta(days=WINDOW) <= TEST_SNAP
assert TEST_SNAP + pd.Timedelta(days=WINDOW) <= REF


def snapshot(snap):
    """Features known on `snap` (orders strictly before it) and what happens in the next WINDOW days."""
    h = df[df.order_date < snap].sort_values("order_date")
    f = h.groupby("customer_id").agg(
        n_orders=("order_id", "count"), total_spend=("total_amount", "sum"), avg_discount=("discount", "mean"),
        return_rate=("is_returned", "mean"), avg_delivery=("delivery_time_days", "mean"), n_categories=("category", "nunique"),
        first=("order_date", "min"), last=("order_date", "max"))
    f["recency_days"] = (snap - f["last"]).dt.days
    f["tenure_days"] = (f["last"] - f["first"]).dt.days
    f["aov"] = f.total_spend / f.n_orders
    f["log_spend"], f["log_aov"] = np.log1p(f.total_spend), np.log1p(f.aov)
    la = h.groupby("customer_id").last()
    f["age"], f["gender"], f["region"] = la.customer_age, la.customer_gender, la.region
    f["fav_category"] = (h.groupby(["customer_id", "category"]).total_amount.sum().reset_index()
                         .sort_values(["customer_id", "total_amount"], ascending=[True, False]).drop_duplicates("customer_id")
                         .set_index("customer_id").category)
    w = df[(df.order_date >= snap) & (df.order_date < snap + pd.Timedelta(days=WINDOW))].groupby("customer_id").total_amount.agg(["sum", "count"])
    f["future_spend"] = w["sum"].reindex(f.index).fillna(0)
    f["bought_again"] = (w["count"].reindex(f.index).fillna(0) > 0).astype(int)
    f["snapshot"] = snap
    return f.drop(columns=["first", "last"]).reset_index()


train = pd.concat([snapshot(s) for s in TRAIN_SNAPS], ignore_index=True)
test = snapshot(TEST_SNAP)
NUM = ["n_orders", "recency_days", "tenure_days", "log_spend", "log_aov", "avg_discount", "return_rate", "avg_delivery", "n_categories", "age"]
CATF = ["gender", "region", "fav_category"]
pre = lambda: ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CATF)])
R["model_data"] = dict(window_days=WINDOW, train_rows=len(train), train_customers=int(train.customer_id.nunique()),
                       train_positive_rate=float(train.bought_again.mean()), train_positives=int(train.bought_again.sum()),
                       test_rows=len(test), test_positives=int(test.bought_again.sum()), test_positive_rate=float(test.bought_again.mean()),
                       test_snapshot=str(TEST_SNAP.date()), train_snapshots=[str(s.date()) for s in TRAIN_SNAPS],
                       test_future_spend_mean=float(test.future_spend.mean()), test_future_spend_sd=float(test.future_spend.std()))

# --- Model A: will the customer buy again in the next 180 days? (classification) -----------------
Xtr, ytr, Xte, yte = train[NUM + CATF], train.bought_again, test[NUM + CATF], test.bought_again
clfs = {
    "Baseline (always 'no')": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=3000),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=40, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(300, min_samples_leaf=15, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
}
fitted, prob, crow = {}, {}, []
brng = np.random.default_rng(SEED)
for name, est in clfs.items():
    pipe = Pipeline([("pre", pre()), ("m", est)]).fit(Xtr, ytr)
    fitted[name] = pipe
    pr = pipe.predict_proba(Xte)[:, 1]
    prob[name] = pr
    pred = pipe.predict(Xte)
    aucs = []
    for _ in range(1000):
        i = brng.integers(0, len(yte), len(yte))
        if yte.values[i].min() != yte.values[i].max():
            aucs.append(roc_auc_score(yte.values[i], pr[i]))
    crow.append(dict(model=name, accuracy=accuracy_score(yte, pred), balanced_accuracy=balanced_accuracy_score(yte, pred),
                     precision=precision_score(yte, pred, zero_division=0), recall=recall_score(yte, pred, zero_division=0),
                     f1=f1_score(yte, pred, zero_division=0), roc_auc=roc_auc_score(yte, pr), pr_auc=average_precision_score(yte, pr),
                     auc_ci_low=np.percentile(aucs, 2.5), auc_ci_high=np.percentile(aucs, 97.5)))
cmet = pd.DataFrame(crow)
cmet.round(4).to_csv(os.path.join(P["models"], "model_metrics_classification.csv"), index=False)
R["clf_metrics"] = cmet.round(4).to_dict("records")
best_clf = cmet[cmet.model != "Baseline (always 'no')"].sort_values("roc_auc", ascending=False).iloc[0].model
R["best_clf"] = best_clf

# --- Model B: how much will the customer spend in the next 180 days? (regression) ------------------
regs = {
    "Baseline (training mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=40, random_state=SEED),
    "Random Forest": RandomForestRegressor(300, min_samples_leaf=15, random_state=SEED, n_jobs=-1),
}
rrow, rfit = [], {}
for name, est in regs.items():
    pipe = Pipeline([("pre", pre()), ("m", est)]).fit(Xtr, train.future_spend)
    rfit[name] = pipe
    pr = pipe.predict(Xte)
    rrow.append(dict(model=name, rmse=mean_squared_error(test.future_spend, pr) ** .5, mae=mean_absolute_error(test.future_spend, pr),
                     r2=r2_score(test.future_spend, pr)))
rmet = pd.DataFrame(rrow)
rmet.round(4).to_csv(os.path.join(P["models"], "model_metrics_future_spend.csv"), index=False)
R["reg_metrics"] = rmet.round(4).to_dict("records")

# --- Model C: expected value of a customer's purchase from profile and preferences ----------------
CF = ["category", "payment_method", "region", "customer_gender", "customer_age"]
split = pd.Timestamp("2025-03-12")
otr, ote = df[df.order_date < split], df[df.order_date >= split]
preC = lambda: ColumnTransformer([("num", StandardScaler(), ["customer_age"]),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), ["category", "payment_method", "region", "customer_gender"])])
cmods = {
    "Baseline (training mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=25, random_state=SEED),
    "Random Forest": RandomForestRegressor(300, min_samples_leaf=10, random_state=SEED, n_jobs=-1),
}
crow2, cfit = [], {}
for name, est in cmods.items():
    pipe = Pipeline([("pre", preC()), ("m", est)]).fit(otr[CF], otr.total_amount)
    cfit[name] = pipe
    pr = pipe.predict(ote[CF])
    crow2.append(dict(model=name, rmse=mean_squared_error(ote.total_amount, pr) ** .5, mae=mean_absolute_error(ote.total_amount, pr),
                      r2=r2_score(ote.total_amount, pr)))
vmet = pd.DataFrame(crow2)
vmet.round(4).to_csv(os.path.join(P["models"], "model_metrics_order_value.csv"), index=False)
R["val_metrics"] = vmet.round(4).to_dict("records")
R["val_data"] = dict(train_orders=len(otr), test_orders=len(ote), test_mean=float(ote.total_amount.mean()), test_sd=float(ote.total_amount.std()))

# permutation importance (on the held-out data)
pi_c = permutation_importance(cfit["Random Forest"], ote[CF], ote.total_amount, n_repeats=15, random_state=SEED, scoring="neg_root_mean_squared_error")
pi_a = permutation_importance(fitted[best_clf], Xte, yte, n_repeats=15, random_state=SEED, scoring="roc_auc")
R["importance_value"] = dict(zip(CF, pi_c.importances_mean.round(4)))
R["importance_clf"] = dict(zip(NUM + CATF, pi_a.importances_mean.round(4)))

# model charts
fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.7))
ax = axs[0]
ax.plot([0, 1], [0, 1], color=INK2, lw=1.2, ls="--", label="Chance (AUC 0.50)")
for i, name in enumerate([n for n in clfs if not n.startswith("Baseline")]):
    fpr, tpr, _ = roc_curve(yte, prob[name])
    ax.plot(fpr, tpr, color=CAT[i], lw=2, label=f"{name} (AUC {cmet.set_index('model').roc_auc[name]:.2f})")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"Buy again within {WINDOW} days? ROC curves")
ax.legend(loc="lower right", fontsize=9)
cmx = confusion_matrix(yte, fitted[best_clf].predict(Xte))
sns.heatmap(cmx, annot=True, fmt="d", cmap=SEQ, cbar=False, linewidths=2, linecolor=SURFACE, ax=axs[1], annot_kws={"size": 13})
axs[1].set(xticklabels=["Predicted: no", "Predicted: yes"], yticklabels=["Actual: no", "Actual: yes"], title=f"Confusion matrix: {best_clf}")
fig.tight_layout()
save(fig, "09_classification_results.png")

fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.3))
for ax, met, ttl in ((axs[0], rmet, f"Next-{WINDOW}-day spend per customer"), (axs[1], vmet, "Value of a single purchase")):
    m_ = met.set_index("model").rmse
    cols = [INK2 if n.startswith("Baseline") else BLUE for n in m_.index]
    bars = ax.barh(m_.index[::-1], m_.values[::-1], color=cols[::-1], height=.6)
    for r_, v in zip(bars, m_.values[::-1]):
        ax.text(v + m_.max() * .01, r_.get_y() + r_.get_height() / 2, f"{v:,.1f}", va="center", fontsize=9.5)
    ax.set_xlim(0, m_.max() * 1.15)
    ax.set(xlabel="RMSE on test data (lower is better)", title=ttl)
fig.tight_layout()
save(fig, "10_regression_rmse.png")

fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.0))
for ax, imp, ttl, xl in ((axs[0], R["importance_clf"], "Buy-again model: tiny values = noise", "drop in ROC-AUC when shuffled (max 0.02)"),
                         (axs[1], R["importance_value"], "Purchase-value model: category dominates", "increase in RMSE when shuffled")):
    s = pd.Series(imp).sort_values()
    s = s.tail(8)
    ax.barh(s.index, s.values, color=BLUE, height=.6)
    ax.axvline(0, color=INK2, lw=1)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set(title=ttl, xlabel=xl)
fig.tight_layout()
save(fig, "11_permutation_importance.png")

# %% 5. Save results ---------------------------------------------------------------------------
LOGDF = pd.DataFrame(LOG, columns=["Step", "Finding", "Action"])
LOGDF.to_csv(os.path.join(P["data"], "data_cleaning_log.csv"), index=False)
json.dump(R, open(os.path.join(P["models"], "results.json"), "w"), indent=1, default=str)
print("\nClassification (test):\n", cmet.round(3).to_string(index=False))
print("\nFuture spend (test):\n", rmet.round(3).to_string(index=False))
print("\nOrder value (test):\n", vmet.round(3).to_string(index=False))
print("\nModel data:", R["model_data"])
print("ID check:", R["idcheck"])
print("Cluster check:", R["cluster_check"])
