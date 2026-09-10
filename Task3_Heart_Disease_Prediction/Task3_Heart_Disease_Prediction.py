# ============================================================
# Task 3: Heart Disease Prediction
# DevelopersHub Corporation – AI/ML Engineering Internship
# Dataset: UCI Heart Disease Dataset (Cleveland)
# Models: Logistic Regression + Decision Tree (GridSearchCV)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score
)

# Set visual style
sns.set_style('whitegrid')

# ── 1. Load Dataset ──────────────────────────────────────────
url = 'https://raw.githubusercontent.com/dsrscientist/dataset1/refs/heads/master/heart_disease.csv'

try:
    df = pd.read_csv(url)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"URL failed: {e}")
    print("Please place heart.csv in this folder and use: df = pd.read_csv('heart.csv')")
    raise

print("\nDataset shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ── 2. Dataset Inspection ────────────────────────────────────
print("\nDataset Info:")
df.info()

# ── 3. Data Cleaning ─────────────────────────────────────────
print("\nMissing values per column:")
print(df.isnull().sum())

df.dropna(inplace=True)
print(f"\nAfter dropping missing values: {df.shape}")

# Target is already binary (0 = no disease, 1 = disease)
print("\nTarget distribution:")
print(df['target'].value_counts())

# ── 4. Exploratory Data Analysis (EDA) ───────────────────────
print("\nStatistical Summary:")
print(df.describe())

# Target distribution plot
plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df)
plt.title('Distribution of Heart Disease (0=No, 1=Yes)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Pairplot of selected features
selected = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']
sns.pairplot(df[selected], hue='target', diag_kind='kde')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.tight_layout()
plt.show()

# Boxplots of features by target
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for i, feat in enumerate(features):
    row, col = i // 3, i % 3
    sns.boxplot(x='target', y=feat, data=df, ax=axes[row, col])
    axes[row, col].set_title(f'{feat} vs target')
axes[1, 2].set_visible(False)
plt.tight_layout()
plt.show()

# ── 5. Preprocessing ─────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nTraining set shape:", X_train_scaled.shape)
print("Test set shape:    ", X_test_scaled.shape)

# ── 6. Model Training ─────────────────────────────────────────
# Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
print("\nLogistic Regression trained successfully.")

# Decision Tree with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_dt = grid_search.best_estimator_
print("Decision Tree trained successfully.")
print("Best Decision Tree parameters:", grid_search.best_params_)

# ── 7. Model Evaluation ───────────────────────────────────────
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_dt  = best_dt.predict(X_test_scaled)

acc_log = accuracy_score(y_test, y_pred_log)
acc_dt  = accuracy_score(y_test, y_pred_dt)
print(f"\nLogistic Regression Accuracy: {acc_log * 100:.2f}%")
print(f"Decision Tree Accuracy:       {acc_dt  * 100:.2f}%")

print("\n=== Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred_log,
      target_names=['No Disease', 'Disease']))

print("\n=== Decision Tree Classification Report ===")
print(classification_report(y_test, y_pred_dt,
      target_names=['No Disease', 'Disease']))

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d',
            cmap='Blues', ax=axes[0],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[0].set_title('Logistic Regression — Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d',
            cmap='Greens', ax=axes[1],
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
axes[1].set_title('Decision Tree — Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.show()

# ROC Curves
y_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]
y_proba_dt  = best_dt.predict_proba(X_test_scaled)[:, 1]

fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
fpr_dt,  tpr_dt,  _ = roc_curve(y_test, y_proba_dt)
auc_log = roc_auc_score(y_test, y_proba_log)
auc_dt  = roc_auc_score(y_test, y_proba_dt)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.2f})', color='blue')
plt.plot(fpr_dt,  tpr_dt,  label=f'Decision Tree       (AUC = {auc_dt:.2f})',  color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — Logistic Regression vs Decision Tree')
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nLogistic Regression AUC: {auc_log:.2f}")
print(f"Decision Tree AUC:       {auc_dt:.2f}")

# ── 8. Feature Importance Analysis ───────────────────────────
coef_log = pd.Series(
    np.abs(log_reg.coef_[0]), index=X.columns
).sort_values(ascending=False)

print("\nLogistic Regression Feature Importance:")
print(coef_log)

importances_dt = pd.Series(
    best_dt.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\nDecision Tree Feature Importances:")
print(importances_dt)

# Feature importance plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
coef_log.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Logistic Regression Coefficients (Absolute)')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Coefficient Value')
axes[0].tick_params(axis='x', rotation=45)

importances_dt.plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Decision Tree Feature Importances')
axes[1].set_xlabel('Features')
axes[1].set_ylabel('Importance Score')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Decision Tree structure
plt.figure(figsize=(20, 10))
plot_tree(best_dt, feature_names=X.columns.tolist(),
          class_names=['No Disease', 'Disease'],
          filled=True, max_depth=3)
plt.title('Decision Tree Structure (max_depth=3)')
plt.tight_layout()
plt.show()

# ── 9. Final Insights ─────────────────────────────────────────
print(f"""
============ KEY INSIGHTS ============
1. Logistic Regression Accuracy : {acc_log * 100:.2f}%
   Decision Tree Accuracy       : {acc_dt  * 100:.2f}%

2. Logistic Regression AUC : {auc_log:.2f}
   Decision Tree AUC       : {auc_dt:.2f}

3. Top important features:
   - cp       (chest pain type)
   - thalach  (maximum heart rate achieved)
   - oldpeak  (ST depression)
   - ca       (number of major vessels)
   - thal     (blood disorder type)

4. Logistic Regression is simpler and more interpretable.
   Decision Tree captures non-linear relationships but
   needs pruning to avoid overfitting.

5. Further improvements possible with:
   Random Forest, XGBoost, or more feature engineering.
======================================
""")
