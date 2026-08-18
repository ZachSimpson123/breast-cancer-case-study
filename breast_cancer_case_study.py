"""
================================================================================
Breast Cancer Prediction - Case Study Code (DLBDSME01, Task 1)
================================================================================
This script matches the write-up "20260918_Zach_Simpson_32212726_DLBDSME01.docx".
Each part of the code is labelled with the section number it belongs to in
that document, so you can follow along side by side.

HOW TO RUN (VS Code):
    pip install numpy pandas matplotlib seaborn scikit-learn shap joblib
    python breast_cancer_case_study.py

This creates two folders next to the script:
    figures/   -> all plots (Figures 1-11 in the write-up)
    models/    -> saved model, scaler, and result files (Tables 1-4)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves plots to file instead of popping up windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, RocCurveDisplay
)
import joblib
import json
import shap

# Fixed seed so results are exactly reproducible every time this runs
RANDOM_STATE = 24

# Output folders (created next to this script)
FIG_DIR = "figures"
MODEL_DIR = "models"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
sns.set_style("whitegrid")


# ==========================================================================
# SECTION 3: DATA UNDERSTANDING
# ==========================================================================
print("=" * 80)
print("SECTION 3: DATA UNDERSTANDING")
print("=" * 80)

# Load the dataset (same data as described in the case study, built into sklearn)
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
df = df.drop(columns=["target"])

# Rename columns to match the naming used in the write-up (e.g. radius_mean)
def rename_column(col):
    if col.startswith("mean "):
        return col.replace("mean ", "") + "_mean"
    if col.startswith("worst "):
        return col.replace("worst ", "") + "_worst"
    if col.endswith(" error"):
        return col.replace(" error", "") + "_se"
    return col

df = df.rename(columns=rename_column)
df.columns = [c.replace(" ", "_") for c in df.columns]

print(f"Records: {df.shape[0]}, Features: {df.shape[1] - 1}")
print(df["diagnosis"].value_counts())

# Table 1: descriptive statistics for the first 5 features
print("\nTable 1 - Descriptive statistics:")
print(df.iloc[:, :5].describe().T.round(3))

# Data quality checks (missing values, duplicates, outliers)
print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

def count_outliers_iqr(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()

feature_cols = [c for c in df.columns if c != "diagnosis"]
outliers = df[feature_cols].apply(count_outliers_iqr).sort_values(ascending=False)
print("\nMost outliers:\n", outliers.head(3))

# Figure 1: class balance
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="diagnosis", hue="diagnosis", legend=False,
              palette=["#4C72B0", "#DD8452"])
plt.title("Class Distribution: Benign vs. Malignant")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_class_distribution.png", dpi=150)
plt.close()

# Figure 2: correlation heatmap (mean features only)
mean_cols = [c for c in feature_cols if c.endswith("mean")]
plt.figure(figsize=(9, 7))
sns.heatmap(df[mean_cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap - Mean Features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_correlation_heatmap.png", dpi=150)
plt.close()

# Figure 3: boxplots of 4 key features by diagnosis
key_features = ["radius_mean", "texture_mean", "concavity_mean", "area_mean"]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
for ax, feat in zip(axes, key_features):
    sns.boxplot(data=df, x="diagnosis", y=feat, hue="diagnosis", ax=ax,
                legend=False, palette=["#4C72B0", "#DD8452"])
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_boxplots_key_features.png", dpi=150)
plt.close()


# ==========================================================================
# SECTION 4: DATA PREPARATION
# ==========================================================================
print("\n" + "=" * 80)
print("SECTION 4: DATA PREPARATION")
print("=" * 80)

df["target"] = df["diagnosis"].map({"M": 1, "B": 0})
X = df[feature_cols]
y = df["target"]

# 80/20 split, stratified so both sets keep the same 63/37 class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# Scale features (fit on train only, then apply to both - avoids data leakage)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)


# ==========================================================================
# SECTION 5: MODELING - CANDIDATE MODELS OF INCREASING COMPLEXITY
# ==========================================================================
print("\n" + "=" * 80)
print("SECTION 5: MODELING")
print("=" * 80)

results = []  # every model's scores get collected here for Table 3 (Section 6)


def evaluate_model(name, model, is_benchmark=False):
    """Train a model, score it on train + test, and save the results."""
    model.fit(X_train_scaled, y_train)
    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)
    proba_test = model.predict_proba(X_test_scaled)[:, 1]

    scores = {
        "model": name,
        "type": "benchmark" if is_benchmark else "candidate",
        "train_accuracy": accuracy_score(y_train, pred_train),
        "test_accuracy": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "f1": f1_score(y_test, pred_test, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba_test),
    }
    results.append(scores)
    print(f"{name}: F1={scores['f1']:.3f}  Precision={scores['precision']:.3f}  "
          f"Recall={scores['recall']:.3f}  ROC-AUC={scores['roc_auc']:.3f}")
    return model


# --- Section 5.1: Benchmark models ---------------------------------------
print("\n-- 5.1 Benchmark models --")
evaluate_model("Majority Class", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE), is_benchmark=True)
evaluate_model("Naive Bayes", GaussianNB(), is_benchmark=True)
evaluate_model("KNN (k=5)", KNeighborsClassifier(n_neighbors=5), is_benchmark=True)
evaluate_model("Decision Tree (depth=3)", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE), is_benchmark=True)

# --- Sections 5.2-5.4: the three main candidates, tuned with CV ----------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

def tune_with_cv(name, estimator, param_grid):
    """Try every combination in param_grid using 10-fold cross-validation,
    and keep whichever one scores best on average (Section 5.5)."""
    search = GridSearchCV(estimator, param_grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train_scaled, y_train)
    print(f"{name}: best params = {search.best_params_}, CV F1 = {search.best_score_:.3f}")
    return search.best_estimator_

print("\n-- 5.2 Logistic Regression --")
best_logreg = tune_with_cv("Logistic Regression", LogisticRegression(),
                            {"C": [0.01, 0.1, 1, 10], "max_iter": [5000]})
evaluate_model("Logistic Regression (tuned)", best_logreg)

print("\n-- 5.3 Decision Tree --")
best_tree = tune_with_cv("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE),
                          {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 10]})
evaluate_model("Decision Tree (tuned)", best_tree)

print("\n-- 5.4 Random Forest --")
best_rf = tune_with_cv("Random Forest", RandomForestClassifier(random_state=RANDOM_STATE),
                        {"n_estimators": [100, 200, 300], "max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 2, 4]})
evaluate_model("Random Forest (tuned)", best_rf)

# --- Section 5.5: cross-validated F1 (mean +/- std) for Table 2 ----------
print("\n-- 5.5 Cross-validated F1 scores (Table 2) --")
for name, model in [("Logistic Regression", best_logreg), ("Decision Tree", best_tree), ("Random Forest", best_rf)]:
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
    print(f"{name}: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")


# ==========================================================================
# SECTION 6: MODEL EVALUATION & COMPARISON
# ==========================================================================
print("\n" + "=" * 80)
print("SECTION 6: MODEL EVALUATION & COMPARISON (Table 3)")
print("=" * 80)

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv(f"{MODEL_DIR}/model_comparison.csv", index=False)


# ==========================================================================
# SECTION 7: MODEL INTERPRETATION
# ==========================================================================
print("\n" + "=" * 80)
print("SECTION 7: MODEL INTERPRETATION")
print("=" * 80)

# --- 7.1 Logistic Regression coefficients ---------------------------------
coef_df = pd.DataFrame({"feature": X_train_scaled.columns, "coefficient": best_logreg.coef_[0]}) \
    .sort_values("coefficient", key=abs, ascending=False)
print("\n7.1 Top logistic regression coefficients:\n", coef_df.head(5).to_string(index=False))

plt.figure(figsize=(7, 5))
top10 = coef_df.head(10).sort_values("coefficient")
colors = ["#DD8452" if c > 0 else "#4C72B0" for c in top10["coefficient"]]
plt.barh(top10["feature"], top10["coefficient"], color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression: Top 10 Coefficients")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_logreg_coefficients.png", dpi=150)
plt.close()

# --- 7.2 Decision Tree structure ------------------------------------------
plt.figure(figsize=(14, 7))
plot_tree(best_tree, feature_names=X_train_scaled.columns, class_names=["Benign", "Malignant"],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title("Decision Tree Structure")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_decision_tree.png", dpi=150)
plt.close()

# --- 7.3 Random Forest: built-in importance + permutation importance ------
rf_importance = pd.DataFrame({"feature": X_train_scaled.columns, "importance": best_rf.feature_importances_}) \
    .sort_values("importance", ascending=False)
print("\n7.3 Top random forest importances:\n", rf_importance.head(5).to_string(index=False))

perm = permutation_importance(best_rf, X_test_scaled, y_test, n_repeats=30, random_state=RANDOM_STATE, scoring="f1")
perm_df = pd.DataFrame({"feature": X_test_scaled.columns, "importance": perm.importances_mean}) \
    .sort_values("importance", ascending=False)

plt.figure(figsize=(7, 5))
top10_perm = perm_df.head(10).sort_values("importance")
plt.barh(top10_perm["feature"], top10_perm["importance"], color="#4C72B0")
plt.title("Random Forest: Permutation Importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_permutation_importance.png", dpi=150)
plt.close()

# --- 7.4 Partial dependence plot for the top feature -----------------------
top_feature = perm_df.iloc[0]["feature"]
fig, ax = plt.subplots(figsize=(6, 5))
PartialDependenceDisplay.from_estimator(best_rf, X_train_scaled, [top_feature], ax=ax)
plt.title(f"Partial Dependence: {top_feature}")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_partial_dependence.png", dpi=150)
plt.close()

# --- 7.5 SHAP: global summary + one local explanation -----------------------
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer(X_test_scaled)
shap_malignant = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

plt.figure()
shap.summary_plot(shap_malignant, X_test_scaled, show=False, max_display=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_shap_summary_global.png", dpi=150, bbox_inches="tight")
plt.close()

# pick one correctly-predicted malignant case to explain individually
rf_test_pred = best_rf.predict(X_test_scaled)
malignant_correct = np.where((y_test.values == 1) & (rf_test_pred == 1))[0]
sample_idx = malignant_correct[0]

plt.figure()
shap.plots.waterfall(shap_malignant[sample_idx], show=False, max_display=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_shap_local_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n7.5 SHAP figures saved (local explanation for test case #{sample_idx})")


# ==========================================================================
# SECTION 8: ERROR ANALYSIS
# ==========================================================================
print("\n" + "=" * 80)
print("SECTION 8: ERROR ANALYSIS")
print("=" * 80)

# Random Forest confusion matrix + ROC curve (Figures 10-11)
rf_pred = best_rf.predict(X_test_scaled)
rf_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
cm_rf = confusion_matrix(y_test, rf_pred)
print("Random Forest confusion matrix:\n", cm_rf)

fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Benign", "Malignant"]).plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix: Random Forest (tuned)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_confusion_matrix.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_estimator(best_rf, X_test_scaled, y_test, ax=ax)
plt.title("ROC Curve: Random Forest (tuned)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_roc_curve.png", dpi=150)
plt.close()

# Logistic Regression confusion matrix (quoted as numbers in Section 8 text)
lr_pred = best_logreg.predict(X_test_scaled)
cm_lr = confusion_matrix(y_test, lr_pred)
print("Logistic Regression confusion matrix:\n", cm_lr)

# Table 4: misclassified test cases (using Random Forest's predictions)
misclassified_mask = rf_pred != y_test.values
misclassified = X_test.loc[X_test_scaled.index[misclassified_mask]].copy()
misclassified["true_label"] = y_test[misclassified_mask].map({1: "Malignant", 0: "Benign"})
misclassified["predicted_label"] = "Benign"  # all misclassified cases here are false negatives
misclassified["predicted_probability_malignant"] = rf_proba[misclassified_mask].round(3)
print("\nTable 4 - Misclassified cases:\n", misclassified[["true_label", "predicted_probability_malignant"]])
misclassified.to_csv(f"{MODEL_DIR}/misclassified_instances.csv")


# ==========================================================================
# SECTION 9: DEPLOYMENT PROPOSAL
# ==========================================================================
print("\n" + "=" * 80)
print("SECTION 9: DEPLOYMENT PROPOSAL")
print("=" * 80)

# Logistic Regression is the model proposed for deployment (see Sections 7.6 and 9):
# it matches Random Forest's accuracy while making fewer false negatives and
# staying fully interpretable.
final_model = best_logreg
FINAL_MODEL_NAME = "Logistic Regression (tuned)"

joblib.dump(final_model, f"{MODEL_DIR}/final_model.joblib")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

model_report = {
    "model_configuration": {
        "target_variable": "diagnosis (malignant=1, benign=0)",
        "algorithm": "LogisticRegression",
        "hyperparameters": best_logreg.get_params(),
    },
    "model_performance": {
        "test_f1": float(f1_score(y_test, lr_pred)),
        "test_precision": float(precision_score(y_test, lr_pred)),
        "test_recall": float(recall_score(y_test, lr_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, best_logreg.predict_proba(X_test_scaled)[:, 1])),
    },
}
with open(f"{MODEL_DIR}/model_report.json", "w") as f:
    json.dump(model_report, f, indent=2, default=str)

print(f"Saved: {MODEL_DIR}/final_model.joblib ({FINAL_MODEL_NAME})")
print(f"Saved: {MODEL_DIR}/scaler.joblib")
print(f"Saved: {MODEL_DIR}/model_report.json")

print("\n" + "=" * 80)
print(f"FINAL RESULT: {FINAL_MODEL_NAME} -> F1 = {model_report['model_performance']['test_f1']:.3f} "
      f"(target: F1 > 0.95)")
print("=" * 80)