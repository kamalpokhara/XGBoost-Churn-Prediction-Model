import pandas as pd
import numpy as np

# ── 1. LOAD & INSPECT ─────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/ecommerce_dataset/events.csv")

print("Shape:", df.shape)
print("\nDtypes:")
print(df.dtypes)
print("\nHead:")
print(df.head())
print("\nEvent types:")
print(df["event_type"].value_counts())
print("\nNull check:")
print(df.isnull().sum())
print("\nDate range:")
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
print(f"  {df['event_timestamp'].min()} → {df['event_timestamp'].max()}")
print(f"\nUnique users:    {df['user_id'].nunique()}")
print(f"Unique products: {df['product_id'].nunique()}")

# ── 2. CLEAN ──────────────────────────────────────────────────────────────────
df["event_type"] = df["event_type"].str.strip().str.lower()

# Drop exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"\nDropped {before - len(df)} duplicate rows")

# Validate event types
valid_events = {"view", "cart", "purchase", "wishlist"}
invalid = df[~df["event_type"].isin(valid_events)]
if len(invalid) > 0:
    print(f"WARNING: {len(invalid)} rows with unexpected event types:")
    print(invalid["event_type"].value_counts())
    df = df[df["event_type"].isin(valid_events)].copy()

# ── 3. INTERACTION WEIGHTS ────────────────────────────────────────────────────
# wishlist sits between view and cart in purchase intent
weight_map = {
    "view": 1,
    "wishlist": 2,
    "cart": 3,
    "purchase": 5,
}
df["interaction_weight"] = df["event_type"].map(weight_map).astype(int)

print("\nWeight distribution:")
print(df.groupby("event_type")["interaction_weight"].first())

# ── 4. CHURN ANCHOR ───────────────────────────────────────────────────────────
CHURN_DAYS = 30
today = df["event_timestamp"].max()
print(f"\nTODAY (dataset end): {today.date()}")

# ── 5. CHURN LABEL ────────────────────────────────────────────────────────────
last_ts = (
    df.groupby("user_id")["event_timestamp"]
    .max()
    .rename("last_timestamp")
    .reset_index()
)
last_ts["days_since_last_activity"] = (today - last_ts["last_timestamp"]).dt.days
last_ts["churn"] = (last_ts["days_since_last_activity"] >= CHURN_DAYS).astype(int)

print(
    f"\nChurn rate ({CHURN_DAYS}d): {last_ts['churn'].mean():.3f} "
    f"({last_ts['churn'].sum()} churned / {len(last_ts)} users)"
)

# ── 6. CORE AGGREGATIONS ──────────────────────────────────────────────────────
agg = (
    df.groupby("user_id")
    .agg(
        total_interactions=("event_type", "count"),
        unique_products=("product_id", "nunique"),
        total_weight=("interaction_weight", "sum"),
        avg_weight=("interaction_weight", "mean"),
    )
    .reset_index()
)

# ── 7. PER-TYPE COUNTS ────────────────────────────────────────────────────────
type_counts = (
    df.groupby(["user_id", "event_type"]).size().unstack(fill_value=0).reset_index()
)
for col in ["view", "wishlist", "cart", "purchase"]:
    if col not in type_counts.columns:
        type_counts[col] = 0

type_counts = type_counts.rename(
    columns={
        "view": "view_count",
        "wishlist": "wishlist_count",
        "cart": "cart_count",
        "purchase": "purchase_count",
    }
)[["user_id", "view_count", "wishlist_count", "cart_count", "purchase_count"]]

# ── 8. ACTIVITY SPAN ──────────────────────────────────────────────────────────
span = (
    df.groupby("user_id")["event_timestamp"]
    .agg(first_ts="min", last_ts="max")
    .reset_index()
)
span["activity_span_days"] = (span["last_ts"] - span["first_ts"]).dt.days

# ── 9. ASSEMBLE ───────────────────────────────────────────────────────────────
features = (
    last_ts[["user_id", "days_since_last_activity", "churn"]]
    .merge(agg, on="user_id", how="left")
    .merge(type_counts, on="user_id", how="left")
    .merge(span[["user_id", "activity_span_days"]], on="user_id", how="left")
    .fillna(0)
)

# ── 10. DERIVED FEATURES ──────────────────────────────────────────────────────
# Conversion ratios — all use +1 Laplace smoothing
features["view_to_purchase_ratio"] = features["view_count"] / (
    features["purchase_count"] + 1
)
features["cart_to_purchase_ratio"] = features["cart_count"] / (
    features["purchase_count"] + 1
)
features["wishlist_to_purchase_ratio"] = features["wishlist_count"] / (
    features["purchase_count"] + 1
)
features["cart_abandonment_rate"] = features["cart_count"] / (
    features["cart_count"] + features["purchase_count"] + 1
)

# Engagement intensity
features["interactions_per_day"] = features["total_interactions"] / (
    features["activity_span_days"] + 1
)

# Wishlist conversion — did wishlisting lead to purchase?
features["wishlist_to_cart_ratio"] = features["wishlist_count"] / (
    features["cart_count"] + 1
)

# ── 11. SPARSITY REPORT ───────────────────────────────────────────────────────
total = len(features)
print("\n── Sparsity Report ──────────────────────────────────────")
for col in ["view_count", "wishlist_count", "cart_count", "purchase_count"]:
    has = (features[col] > 0).sum()
    print(f"  Users with {col:20s}: {has:6} ({has/total:.1%})")

print("\nChurn rate by user type:")
features["user_type"] = "view_only"
features.loc[features["wishlist_count"] > 0, "user_type"] = "wishlisted"
features.loc[features["cart_count"] > 0, "user_type"] = "carted"
features.loc[features["purchase_count"] > 0, "user_type"] = "purchased"
print(features.groupby("user_type")["churn"].agg(["mean", "count"]).round(3))

print("\nCorrelations with churn:")
num_cols = [
    c
    for c in features.columns
    if c not in ["user_id", "churn", "days_since_last_activity", "user_type"]
]
corr = (
    features[num_cols].corrwith(features["churn"]).sort_values(key=abs, ascending=False)
)
print(corr.round(3))

features = features.drop(columns=["user_type"])

# ── 12. FINAL CHECKS ──────────────────────────────────────────────────────────
print("\nFinal shape:", features.shape)
print("Columns:", features.columns.tolist())
print("\nNull check:")
print(features.isnull().sum()[features.isnull().sum() > 0])
print("\nDescribe:")
print(features.describe().to_string())
print("\nFeatures info\n",features.info())

# # ── 13. SAVE ──────────────────────────────────────────────────────────────────
# features.to_parquet("new_ds_churn_features_final.parquet", index=False)
# print("\nSaved: churn_features_final.parquet")
