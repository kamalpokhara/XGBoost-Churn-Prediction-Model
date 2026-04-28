# Initial trainng. Refer to training2.py for the final version with hyperparameter 
# tuning and model saving.
# This script trains XGBoost and LightGBM classifiers on the churn dataset, 
# evaluates their performance,
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # avoids tkinter crash

# 1. LOAD 
features = pd.read_parquet("new_ds_churn_features_final.parquet")

FEATURES = [
    "activity_span_days",
    "total_interactions",
    "unique_products",
    "view_count",
    "total_weight",
    "view_to_purchase_ratio",
    "cart_count",
    "cart_to_purchase_ratio",
    "cart_abandonment_rate",
    "wishlist_count",
    "interactions_per_day",
    "wishlist_to_purchase_ratio",
]

X = features[FEATURES]
y = features["churn"]

print("Feature set shape:", X.shape)
print("Churn distribution:")
print(y.value_counts(normalize=True).round(3))

# 2. SPLIT 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# 3. CLASS IMBALANCE WEIGHT 
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {scale:.3f}")

# 4. MODELS 
models = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    ),
}

# 5. TRAIN & EVALUATE 
results = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"ROC-AUC: {auc:.4f}")
    print(
        classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"])
    )

    results[name] = {
        "model": model,
        "auc": auc,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
    }

# 6. FEATURE IMPORTANCE 
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (AUC={results[best_name]['auc']:.4f})")

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": best_model.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\nFeature importances:")
print(importance_df.to_string(index=False))

# 7. PLOTS 
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Not Churned", "Churned"]).plot(
        ax=axes[i], colorbar=False
    )
    axes[i].set_title(f"{name}  |  AUC={res['auc']:.4f}")

importance_df.plot.barh(
    x="feature", y="importance", ax=axes[2], legend=False, color="steelblue"
)
axes[2].set_title(f"Feature Importance — {best_name}")
axes[2].invert_yaxis()

plt.tight_layout()
# plt.savefig(
#     "reports/figures/churn_model_evaluation.png", dpi=150, bbox_inches="tight"
# )
# print("\nPlot saved: churn_model_evaluation.png")
