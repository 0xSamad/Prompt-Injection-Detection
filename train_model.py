"""
=============================================================
 Taxonomy-Driven AI-Based Detection of Prompt Injection
 Attacks in Large Language Model Applications
=============================================================
 Authors : Abdul Samad & Maqbool Hussain
 Institute: Institute of Management Sciences, Peshawar
 Course   : CSC-601 Artificial Intelligence (Lab)
 Instructor: Mr. Ali Haider
=============================================================

 USAGE:
   python train_model.py

 REQUIREMENTS:
   pip install scikit-learn pandas numpy matplotlib seaborn openpyxl

 OUTPUT:
   - figures/   : all result plots (PNG)
   - results.csv: model performance summary
=============================================================
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ── Configuration ─────────────────────────────────────────────
DATASET_PATH = "prompt_injection_dataset.csv"
FIGURES_DIR  = "figures"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

CLASS_NAMES = [
    "Benign",
    "Instruction Override",
    "Role Manipulation",
    "Data Exfiltration",
    "Obfuscation/Encoding",
    "Social Engineering"
]
SHORT_NAMES = ["Benign", "Inst.Ovr", "Role Man", "Data Exf", "Obfusc.", "Soc.Eng"]
COLORS      = ["#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#3498db", "#1abc9c"]
MODEL_COLORS= ["#2980b9", "#e74c3c", "#27ae60"]

os.makedirs(FIGURES_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATASET
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("  PROMPT INJECTION DETECTION — MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("train.csv")
X  = df["text"].values
y  = df["label"].values

print(f"\n[1] Dataset loaded: {len(df)} samples, {len(np.unique(y))} classes")
print("\nClass distribution:")
for i, name in enumerate(CLASS_NAMES):
    count = (y == i).sum()
    bar   = "█" * int(count / 2)
    print(f"    {i} — {name:<26} {count:>3}  {bar}")

# ── Figure 1: Class Distribution ──────────────────────────────
counts = pd.Series(y).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(CLASS_NAMES, counts.values, color=COLORS, edgecolor='white', height=0.6)
for bar, val in zip(bars, counts.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=10)
ax.set_xlabel("Number of Samples", fontsize=11)
ax.set_title("Figure 1. Dataset Class Distribution", fontsize=11, pad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_class_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[Fig 1] Class distribution saved.")

# ══════════════════════════════════════════════════════════════
# STEP 2 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n[2] Split: {len(X_train)} train / {len(X_test)} test (stratified)")

# ══════════════════════════════════════════════════════════════
# STEP 3 — DEFINE MODEL PIPELINES
# ══════════════════════════════════════════════════════════════
#
# Each pipeline = TF-IDF vectorizer + classifier
# TF-IDF config:
#   - ngram_range=(1,2)  : unigrams + bigrams
#   - max_features=15000 : vocabulary cap
#   - sublinear_tf=True  : log-scale TF to reduce high-freq dominance
#
tfidf_params = dict(ngram_range=(1, 2), max_features=15000,
                    sublinear_tf=True, min_df=1)

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
        ))
    ]),

    "SVM (RBF)": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ]),
}

print(f"\n[3] {len(models)} classifiers defined: {', '.join(models.keys())}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — TRAIN, CROSS-VALIDATE, EVALUATE
# ══════════════════════════════════════════════════════════════
results     = {}
conf_mats   = {}
probas      = {}

print("\n" + "=" * 60)
print("  TRAINING & EVALUATION")
print("=" * 60)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for name, pipe in models.items():
    print(f"\n── {name} ──")

    # 5-fold CV on training set
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    print(f"   CV F1-Macro ({CV_FOLDS}-fold): "
          f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Train on full training set
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec  = recall_score(y_test, y_pred, average="macro")
    f1   = f1_score(y_test, y_pred, average="macro")
    pcf1 = f1_score(y_test, y_pred, average=None)

    results[name] = {
        "Accuracy":     round(acc,  4),
        "Precision":    round(prec, 4),
        "Recall":       round(rec,  4),
        "F1-Score":     round(f1,   4),
        "CV F1 Mean":   round(cv_scores.mean(), 4),
        "CV F1 Std":    round(cv_scores.std(),  4),
        "per_class_f1": pcf1.tolist(),
    }
    conf_mats[name] = confusion_matrix(y_test, y_pred)
    probas[name]    = y_proba

    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ══════════════════════════════════════════════════════════════
# STEP 5 — GENERATE FIGURES
# ══════════════════════════════════════════════════════════════
model_names = list(results.keys())
best_model  = max(results, key=lambda m: results[m]["F1-Score"])
print(f"\n[Best Model] {best_model} — F1={results[best_model]['F1-Score']:.4f}")

# ── Figure 2: Overall Performance Bar Charts ──────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
fig.suptitle("Figure 2. Overall Classifier Performance (Macro-Averaged)",
             fontsize=11, y=1.02)
for i, metric in enumerate(metrics):
    vals = [results[m][metric] for m in model_names]
    bars = axes[i].bar(["LR", "SVM", "RF"], vals,
                       color=MODEL_COLORS, width=0.5, edgecolor='white')
    axes[i].set_title(metric, fontsize=11, fontweight='bold')
    axes[i].set_ylim(0.7, 1.05)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    for bar, v in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.004,
                     f"{v:.3f}", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_performance.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Fig 2] Performance comparison saved.")

# ── Figure 3: Per-Class F1 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(CLASS_NAMES))
w = 0.25
for idx, (name, col) in enumerate(zip(model_names, MODEL_COLORS)):
    ax.bar(x + idx * w, results[name]["per_class_f1"],
           w, label=name, color=col, edgecolor='white')
ax.set_xticks(x + w)
ax.set_xticklabels(SHORT_NAMES, fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("F1-Score", fontsize=11)
ax.set_title("Figure 3. Per-Class F1-Score Comparison", fontsize=11, pad=10)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(0.8, color='gray', ls='--', alpha=0.4, lw=1)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_perclass_f1.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Fig 3] Per-class F1 saved.")

# ── Figure 4: Confusion Matrices ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("Figure 4. Normalised Confusion Matrices", fontsize=11)
for ax, (name, cm) in zip(axes, conf_mats.items()):
    cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
                ax=ax, linewidths=0.4, linecolor='white',
                annot_kws={"size": 8}, vmin=0, vmax=1)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_confusion.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Fig 4] Confusion matrices saved.")

# ── Figure 5: ROC Curves (best model) ────────────────────────
y_bin  = label_binarize(y_test, classes=list(range(6)))
y_sc   = probas[best_model]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(f"Figure 5. ROC Curves per Class — {best_model}", fontsize=11)
for i, (ax, cname, col) in enumerate(zip(axes.flat, CLASS_NAMES, COLORS)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_sc[:, i])
    ra = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, lw=2, label=f"AUC = {ra:.3f}")
    ax.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.08, color=col)
    ax.set_title(cname, fontsize=10, fontweight='bold')
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_roc.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Fig 5] ROC curves saved.")

# ── Figure 6: CV Score Distribution ──────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
means = [results[m]["CV F1 Mean"] for m in model_names]
stds  = [results[m]["CV F1 Std"]  for m in model_names]
bars  = ax.bar(["LR", "SVM", "RF"], means,
               color=MODEL_COLORS, yerr=stds, capsize=8,
               width=0.45, edgecolor='white')
ax.set_ylim(0.7, 1.05)
ax.set_ylabel("F1-Macro", fontsize=11)
ax.set_title("Figure 6. 5-Fold Cross-Validation Scores (Mean ± Std)",
             fontsize=11, pad=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, v, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            v + s + 0.008, f"{v:.3f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig6_cv.png", dpi=150, bbox_inches='tight')
plt.close()
print("[Fig 6] CV scores saved.")

# ══════════════════════════════════════════════════════════════
# STEP 6 — SAVE RESULTS TABLE
# ══════════════════════════════════════════════════════════════
rows = []
for name in model_names:
    r = results[name]
    rows.append({
        "Model":       name,
        "Accuracy":    r["Accuracy"],
        "Precision":   r["Precision"],
        "Recall":      r["Recall"],
        "F1-Score":    r["F1-Score"],
        "CV F1 Mean":  r["CV F1 Mean"],
        "CV F1 Std":   r["CV F1 Std"],
    })
results_df = pd.DataFrame(rows)
results_df.to_csv("results.csv", index=False)

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\n  Best Model : {best_model}")
print(f"  Best F1    : {results[best_model]['F1-Score']:.4f}")
print(f"  Best Acc   : {results[best_model]['Accuracy']:.4f}")
print("\n  Figures saved to  : ./figures/")
print("  Results saved to  : ./results.csv")
print("=" * 60)
