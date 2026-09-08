# ============================================================
# Task 1: Exploring and Visualizing a Simple Dataset
# DevelopersHub Corporation – AI/ML Engineering Internship
# Dataset: Iris Dataset (loaded via seaborn)
# Libraries: pandas, matplotlib, seaborn
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Load Dataset ──────────────────────────────────────────
# Load Iris dataset via seaborn into a pandas DataFrame
df = sns.load_dataset('iris')

# ── 2. Basic Inspection ──────────────────────────────────────
print("Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nFirst 5 Rows:\n", df.head())

# ── 3. Dataset Info ──────────────────────────────────────────
print("\nDataset Info:")
df.info()

# ── 4. Data Preprocessing ────────────────────────────────────
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check and remove duplicates
print("\nDuplicates:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# ── 5. Statistical Summary ───────────────────────────────────
print("\nStatistical Summary:")
print(df.describe())

# ── 6. Scatter Plots ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Scatter Plots - Feature Relationships', fontsize=14)

# Sepal scatter plot
sns.scatterplot(data=df, x='sepal_length', y='sepal_width',
                hue='species', ax=axes[0])
axes[0].set_title('Sepal Length vs Sepal Width')

# Petal scatter plot
sns.scatterplot(data=df, x='petal_length', y='petal_width',
                hue='species', ax=axes[1])
axes[1].set_title('Petal Length vs Petal Width')

plt.tight_layout()
plt.show()

# ── 7. Histograms ────────────────────────────────────────────
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle('Histograms - Value Distributions', fontsize=14)

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    sns.histplot(data=df, x=feature, hue='species', kde=True, ax=axes[row][col])
    axes[row][col].set_title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()

# ── 8. Box Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
fig.suptitle('Box Plots - Outlier Detection', fontsize=14)

for i, feature in enumerate(features):
    row, col = i // 2, i % 2
    sns.boxplot(data=df, x='species', y=feature, ax=axes[row][col])
    axes[row][col].set_title(f'{feature} by Species')

plt.tight_layout()
plt.show()

# ── 9. Final Insights ────────────────────────────────────────
print("\n============ FINAL INSIGHTS ============")
print("1. Dataset has", df.shape[0], "rows and", df.shape[1], "columns.")
print("2. No missing values found in the dataset.")
print("3. Setosa species is clearly separable from others.")
print("4. Petal length and width are the strongest distinguishing features.")
print("5. A few mild outliers found in Sepal Width of Setosa species.")
