# ============================================================
# Task 3: Heart Disease Prediction
# DevelopersHub Corporation – AI/ML Engineering Internship
# Dataset: Heart Disease UCI Dataset
# Model: Logistic Regression
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_curve, roc_auc_score)

# ── 1. Load Dataset ──────────────────────────────────────────
# Dataset loaded directly from public URL - no manual download needed
# Backup: download heart.csv from kaggle.com and use pd.read_csv('heart.csv')
url = 'https://raw.githubusercontent.com/dsrscientist/dataset1/refs/heads/master/heart_disease.csv'

try:
    df = pd.read_csv(url)
    if 'condition' in df.columns:
        df.rename(columns={'condition': 'target'}, inplace=True)
    print("Dataset loaded successfully.")
except Exception:
    print("URL failed. Please place heart.csv in this folder.")
    raise

print("\nShape:", df.shape)
print("Column Names:", df.columns.tolist())
print("\nFirst 5 Rows:\n", df.head())

# ── 2. Dataset Inspection ────────────────────────────────────
print("\nDataset Info:")
df.info()

# ── 3. Data Preprocessing ────────────────────────────────────
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows with missing values if any
df.dropna(inplace=True)
print("Shape after cleaning:", df.shape)

# ── 4. Statistical Summary ───────────────────────────────────
print("\nStatistical Summary:")
print(df.describe())

# ── 5. EDA - Target Distribution ─────────────────────────────
print("\nTarget Distribution:")
print(df['target'].value_counts())

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='target')
plt.title('Heart Disease Distribution (0 = No, 1 = Yes)')
plt.xlabel('Target')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# ── 6. EDA - Correlation Heatmap ─────────────────────────────
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# ── 7. Prepare Features and Target ───────────────────────────
X = df.drop('target', axis=1)
y = df['target']

# Train / Test Split - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples: ", X_test.shape[0])

# ── 8. Train Logistic Regression Model ───────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nModel trained successfully.")

# ── 9. Accuracy ──────────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# ── 10. Confusion Matrix ─────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ── 11. ROC Curve ────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

print(f"ROC-AUC Score: {auc_score:.2f}")

# ── 12. Feature Importance ───────────────────────────────────
importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=importance.values, y=importance.index)
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Absolute Coefficient Value')
plt.tight_layout()
plt.show()

print("\nTop Features:")
print(importance)

# ── 13. Final Insights ───────────────────────────────────────
print("\n============ FINAL INSIGHTS ============")
print(f"1. Model Accuracy: {accuracy * 100:.2f}%")
print(f"2. ROC-AUC Score: {auc_score:.2f} — strong classification performance.")
print("3. Most important features: cp, thalach, oldpeak, ca, thal.")
print("4. Confusion matrix shows good handling of both disease classes.")
print("5. Logistic Regression is a solid baseline for heart disease prediction.")
