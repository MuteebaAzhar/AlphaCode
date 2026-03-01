# ============================================================
# TASK 1: CREDIT SCORING MODEL
# CodeAlpha Machine Learning Internship
# ============================================================
# WHAT THIS CODE DOES:
# 1. Generates realistic financial dataset (or you can load your own)
# 2. Preprocesses and engineers features
# 3. Trains 3 models: Logistic Regression, Random Forest, XGBoost
# 4. Evaluates with Precision, Recall, F1, ROC-AUC
# 5. Plots confusion matrix and ROC curves
# ============================================================

# ── STEP 0: Install dependencies (run this in Google Colab first) ──
# !pip install xgboost scikit-learn pandas numpy matplotlib seaborn

# ── STEP 1: Import Libraries ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")

# ── STEP 2: Load or Generate Dataset ─────────────────────────────
# OPTION A: Use a real dataset from UCI or Kaggle
# Example (German Credit Dataset):
# df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
#                  sep=" ", header=None)
# (You'd need to label columns manually for this dataset)

# OPTION B: Generate a realistic synthetic dataset (we'll use this)
# This simulates real-world credit data with:
# income, age, debt, num_credit_cards, payment_history, loan_amount, default (target)

np.random.seed(42)
n_samples = 2000

age           = np.random.randint(18, 70, n_samples)
income        = np.random.randint(20000, 150000, n_samples)
loan_amount   = np.random.randint(1000, 50000, n_samples)
debt_ratio    = np.round(np.random.uniform(0.05, 0.90, n_samples), 2)
num_credit_cards = np.random.randint(0, 10, n_samples)
missed_payments = np.random.randint(0, 10, n_samples)
years_employed  = np.random.randint(0, 30, n_samples)

# Create target: default (1) or not (0)
# Higher debt_ratio, missed_payments → more likely to default
default_prob = (
    0.3 * debt_ratio +
    0.3 * (missed_payments / 10) +
    0.1 * (loan_amount / 50000) -
    0.2 * (income / 150000) -
    0.1 * (years_employed / 30)
)
default_prob = np.clip(default_prob, 0, 1)
default = (np.random.rand(n_samples) < default_prob).astype(int)

df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'debt_ratio': debt_ratio,
    'num_credit_cards': num_credit_cards,
    'missed_payments': missed_payments,
    'years_employed': years_employed,
    'default': default   # TARGET: 1 = defaulted, 0 = did not default
})

print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n🔍 First 5 rows:\n{df.head()}")
print(f"\n📈 Target Distribution:\n{df['default'].value_counts()}")
print(f"\n📉 Default Rate: {df['default'].mean()*100:.1f}%")

# ── STEP 3: Exploratory Data Analysis (EDA) ───────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Credit Scoring - Feature Distributions by Default Status', fontsize=16, y=1.02)

features_to_plot = ['age', 'income', 'debt_ratio', 'missed_payments', 'loan_amount', 'years_employed']
colors = ['#2ecc71', '#e74c3c']

for i, feature in enumerate(features_to_plot):
    ax = axes[i // 3][i % 3]
    for label, color in zip([0, 1], colors):
        subset = df[df['default'] == label][feature]
        ax.hist(subset, bins=30, alpha=0.6, color=color,
                label='No Default' if label == 0 else 'Default')
    ax.set_title(f'{feature} Distribution')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend()

plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA plot saved!")

# ── STEP 4: Feature Engineering ───────────────────────────────────
# Feature engineering = creating new meaningful features from existing ones
# This often improves model performance significantly!

df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)  # Higher = riskier
df['payment_risk_score']   = df['missed_payments'] * df['debt_ratio']  # Combined risk
df['credit_utilization']   = df['num_credit_cards'] * df['debt_ratio']  # Usage indicator
df['income_per_year_employed'] = df['income'] / (df['years_employed'] + 1)

print("\n✅ Feature engineering complete!")
print(f"📊 New dataset shape: {df.shape}")
print("New features created: loan_to_income_ratio, payment_risk_score, credit_utilization, income_per_year_employed")

# ── STEP 5: Correlation Heatmap ────────────────────────────────────
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap\n(Positive values = more correlated with default)', fontsize=13)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Correlation heatmap saved!")

# ── STEP 6: Prepare Data for Modeling ─────────────────────────────
X = df.drop('default', axis=1)  # Features
y = df['default']                # Target

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for Logistic Regression)
# StandardScaler: transforms data so mean=0, std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train, transform train
X_test_scaled  = scaler.transform(X_test)        # Only transform test (no fitting!)

print(f"\n✅ Data split complete!")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples:  {X_test.shape[0]}")

# ── STEP 7: Train Models ───────────────────────────────────────────
print("\n🚀 Training models...")

# MODEL 1: Logistic Regression
# How it works: Finds a linear boundary between classes.
# Best for: Understanding feature importance, fast training
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("✅ Logistic Regression trained!")

# MODEL 2: Random Forest
# How it works: Builds many decision trees and averages their predictions.
# Best for: Handles non-linear patterns, resistant to overfitting
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaling
print("✅ Random Forest trained!")

# MODEL 3: XGBoost
# How it works: Sequentially builds trees, each correcting errors of the previous.
# Best for: Usually top performer on structured/tabular data
xgb_model = XGBClassifier(n_estimators=100, random_state=42,
                           eval_metric='logloss', use_label_encoder=False)
xgb_model.fit(X_train, y_train)
print("✅ XGBoost trained!")

# ── STEP 8: Evaluate All Models ────────────────────────────────────
def evaluate_model(model, X_test_data, y_test, model_name, needs_scaling=False):
    """
    Evaluates a classification model and prints all key metrics.
    
    Parameters:
    - model: trained sklearn model
    - X_test_data: test features (scaled or unscaled depending on model)
    - y_test: true labels
    - model_name: string name for display
    - needs_scaling: whether to use scaled data
    """
    y_pred = model.predict(X_test_data)
    y_prob = model.predict_proba(X_test_data)[:, 1]  # Probability of default
    
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=['No Default', 'Default'])
    cm     = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*55}")
    print(f"  📊 {model_name} Results")
    print(f"{'='*55}")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"\n  Classification Report:\n{report}")
    
    return y_pred, y_prob, cm, auc

# Evaluate all 3 models
lr_pred,  lr_prob,  lr_cm,  lr_auc  = evaluate_model(lr_model,  X_test_scaled, y_test, "Logistic Regression")
rf_pred,  rf_prob,  rf_cm,  rf_auc  = evaluate_model(rf_model,  X_test,        y_test, "Random Forest")
xgb_pred, xgb_prob, xgb_cm, xgb_auc = evaluate_model(xgb_model, X_test,        y_test, "XGBoost")

# ── STEP 9: Confusion Matrices ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices — All Models', fontsize=15)

models_info = [
    (lr_cm,  "Logistic Regression", "#3498db"),
    (rf_cm,  "Random Forest",       "#2ecc71"),
    (xgb_cm, "XGBoost",             "#e67e22"),
]

for ax, (cm, name, color) in zip(axes, models_info):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    ax.set_title(f'{name}', fontsize=12)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved!")

# ── STEP 10: ROC Curves ────────────────────────────────────────────
# ROC curve shows the trade-off between True Positive Rate and False Positive Rate
# AUC (Area Under Curve): 0.5 = random, 1.0 = perfect

plt.figure(figsize=(9, 7))
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)', alpha=0.5)

for (prob, name, color, auc) in [
    (lr_prob,  "Logistic Regression", "#3498db", lr_auc),
    (rf_prob,  "Random Forest",       "#2ecc71", rf_auc),
    (xgb_prob, "XGBoost",             "#e67e22", xgb_auc),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.4f})')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — Credit Scoring Models', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ROC curve saved!")

# ── STEP 11: Feature Importance (Random Forest) ────────────────────
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(9, 7))
colors_bar = ['#e74c3c' if v > feature_importance.median() else '#3498db'
              for v in feature_importance]
feature_importance.plot(kind='barh', color=colors_bar)
plt.title('Feature Importance — Random Forest\n(Red = Most Important Features)', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Feature importance plot saved!")

# ── STEP 12: Model Comparison Summary ─────────────────────────────
print("\n" + "="*55)
print("  🏆 FINAL MODEL COMPARISON SUMMARY")
print("="*55)
print(f"  {'Model':<25} {'ROC-AUC':>10}")
print(f"  {'-'*35}")
for name, auc in [
    ("Logistic Regression", lr_auc),
    ("Random Forest",       rf_auc),
    ("XGBoost",             xgb_auc),
]:
    marker = " ⭐ Best!" if auc == max(lr_auc, rf_auc, xgb_auc) else ""
    print(f"  {name:<25} {auc:>10.4f}{marker}")
print("="*55)

# ── STEP 13: Make a Prediction on New Data ─────────────────────────
print("\n🔮 DEMO: Predicting creditworthiness for a new applicant...")

new_applicant = pd.DataFrame({
    'age': [35],
    'income': [55000],
    'loan_amount': [15000],
    'debt_ratio': [0.45],
    'num_credit_cards': [3],
    'missed_payments': [2],
    'years_employed': [7],
    'loan_to_income_ratio': [15000/55001],
    'payment_risk_score': [2 * 0.45],
    'credit_utilization': [3 * 0.45],
    'income_per_year_employed': [55000/8]
})

prediction = xgb_model.predict(new_applicant)[0]
probability = xgb_model.predict_proba(new_applicant)[0][1]

print(f"  Applicant Details: Income=55k, Loan=15k, Debt Ratio=45%, Missed Payments=2")
print(f"  Default Probability: {probability*100:.1f}%")
print(f"  Decision: {'❌ HIGH RISK - Likely to Default' if prediction == 1 else '✅ LOW RISK - Likely to Repay'}")

print("\n✅ TASK 1 COMPLETE! All plots saved as PNG files.")
print("📁 Files saved: eda_distributions.png, correlation_heatmap.png,")
print("                confusion_matrices.png, roc_curves.png, feature_importance.png")
