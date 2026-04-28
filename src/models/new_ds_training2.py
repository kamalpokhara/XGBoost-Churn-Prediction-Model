# the 3rd stop
# This script performs hyperparameter tuning for both XGBoost and LightGBM classifiers
# using RandomizedSearchCV. It evaluates both models on the test set, 
# compares their performance, and saves the best model along with 
# its metadata for production use.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import warnings

warnings.filterwarnings("ignore")

#1 LOAD
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"scale_pos_weight: {scale:.3f}\n")

# 2. SEARCH SPACES
xgb_params = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3],
    "reg_alpha": [0, 0.01, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}

lgbm_params = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    "min_child_samples": [10, 20, 30, 50],
    "reg_alpha": [0, 0.01, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
    "num_leaves": [20, 31, 50, 70, 100],
}

# 3. BASE ESTIMATORS
base_models = {
    "XGBoost": XGBClassifier(
        scale_pos_weight=scale,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    ),
    "LightGBM": LGBMClassifier(
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    ),
}

param_spaces = {
    "XGBoost": xgb_params,
    "LightGBM": lgbm_params,
}

# 4. TUNE BOTH MODELS
tuned = {}

for name, base in base_models.items():
    print(f"{'='*55}")
    print(f"  Tuning {name}  (60 iterations × 5 folds = 300 fits)")
    print(f"{'='*55}")

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_spaces[name],
        n_iter=60,
        scoring="roc_auc",
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_pred_prob = best.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\nBest params:")
    for k, v in search.best_params_.items():
        print(f"  {k:22s}: {v}")
    print(f"\nCV AUC  : {search.best_score_:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"])
    )

    tuned[name] = {
        "model": best,
        "auc": auc,
        "cv_auc": search.best_score_,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
        "params": search.best_params_,
    }

# 5. COMPARE
print(f"\n{'='*55}")
print("  FINAL COMPARISON")
print(f"{'='*55}")
print(f"{'Model':>12}  {'CV AUC':>8}  {'Test AUC':>8}")
for name, res in tuned.items():
    print(f"{name:>12}  {res['cv_auc']:>8.4f}  {res['auc']:>8.4f}")

winner_name = max(tuned, key=lambda k: tuned[k]["auc"])
winner = tuned[winner_name]
print(f"\nWinner: {winner_name} (Test AUC={winner['auc']:.4f})")

# 6. THRESHOLD ANALYSIS ON WINNER
print(f"\nThreshold analysis — {winner_name}:")
print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    y_t = (winner["y_pred_prob"] >= thresh).astype(int)
    p = precision_score(y_test, y_t)
    r = recall_score(y_test, y_t)
    f = f1_score(y_test, y_t)
    print(f"{thresh:>10.2f} {p:>10.3f} {r:>10.3f} {f:>10.3f}")

# 7. FEATURE IMPORTANCE, WINNER
importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": winner["model"].feature_importances_,
    }
).sort_values("importance", ascending=False)

print(f"\nFeature importances ({winner_name}):")
print(importance_df.to_string(index=False))

# 8. PLOTS
fig, axes = plt.subplots(1, 4, figsize=(24, 5))

# ROC curves, both models
for name, res in tuned.items():
    RocCurveDisplay.from_predictions(
        y_test, res["y_pred_prob"], name=f"{name} (AUC={res['auc']:.4f})", ax=axes[0]
    )
axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_title("ROC Curve Comparison")
axes[0].legend()

# Confusion matrices — both models
for i, (name, res) in enumerate(tuned.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Not Churned", "Churned"]).plot(
        ax=axes[i + 1], colorbar=False
    )
    axes[i + 1].set_title(f"{name}\nAUC={res['auc']:.4f}")

# Feature importance  winner
importance_df.plot.barh(
    x="feature", y="importance", ax=axes[3], legend=False, color="steelblue"
)
axes[3].set_title(f"Feature Importance\n{winner_name}")
axes[3].invert_yaxis()

plt.tight_layout()
plt.savefig(
    "reports/figures/tuned_model_comparison.png", dpi=150, bbox_inches="tight"
)
print("\nPlot saved: tuned_model_comparison.png")

# 9. SAVE BOTH MODELS + WINNER FLAG
for name, res in tuned.items():
    fname = f"models/{'xgb' if name == 'XGBoost' else 'lgbm'}_churn_tuned.pkl"
    joblib.dump(res["model"], fname)
    print(f"Saved: {fname}")

# Save winner separately for easy loading in production
winner_path = "models/churn_model_best.pkl"
joblib.dump(winner["model"], winner_path)
print(f"Saved winner: {winner_path}  ← use this in production")

# Save metadata
import json

meta = {
    "winner": winner_name,
    "test_auc": round(winner["auc"], 4),
    "cv_auc": round(winner["cv_auc"], 4),
    "features": FEATURES,
    "params": {k: str(v) for k, v in winner["params"].items()},
}
with open("models/churn_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved: churn_model_meta.json")
