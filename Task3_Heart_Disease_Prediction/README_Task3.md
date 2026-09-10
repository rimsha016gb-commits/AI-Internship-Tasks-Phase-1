# Task 3: Heart Disease Prediction

---

## Objective

Build a binary classification model to predict whether a person
is at risk of heart disease based on their medical health data.

---

## Dataset

| Property | Details |
|----------|---------|
| Name | UCI Heart Disease Dataset (Cleveland) |
| Source | University of California, Irvine (UCI) ML Repository |
| Loaded From | Public GitHub URL (auto-download, no setup needed) |
| Backup | https://www.kaggle.com/datasets/ronitf/heart-disease-uci |
| Size | 303 rows × 14 columns |
| Missing Values | None |
| Target | 0 = No Disease, 1 = Has Disease |

### Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| age | int | Age of the patient in years |
| sex | int | Gender — 1 = Male, 0 = Female |
| cp | int | Chest pain type — 0 to 3 |
| trestbps | int | Resting blood pressure in mm Hg |
| chol | int | Cholesterol level in mg/dl |
| fbs | int | Fasting blood sugar > 120 mg/dl — 1 = True, 0 = False |
| restecg | int | Resting ECG results — 0 to 2 |
| thalach | int | Maximum heart rate achieved |
| exang | int | Exercise induced chest pain — 1 = Yes, 0 = No |
| oldpeak | float | ST depression induced by exercise |
| slope | int | Slope of peak exercise ST segment — 0 to 2 |
| ca | int | Number of major vessels — 0 to 4 |
| thal | int | Blood disorder type — 0 to 3 |
| target | int | Heart disease — 1 = Yes, 0 = No |

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| pandas | Data loading and manipulation |
| numpy | Mathematical operations |
| matplotlib | Creating charts and graphs |
| seaborn | Statistical visualizations |
| scikit-learn | Models, scaling, metrics, GridSearchCV |

---

## Install Requirements

```
pip install -r requirements.txt

---

## Models Applied

| Model | Details |
|-------|---------|
| Logistic Regression | random_state=42, max_iter=1000 |
| Decision Tree | GridSearchCV — max_depth, min_samples_split, min_samples_leaf |

### Preprocessing
- **StandardScaler** applied before training
- **Stratified train/test split** — 80% train, 20% test
- **random_state = 42** for reproducibility

---

## Output

### Printed Output
```
Dataset loaded successfully.
Dataset shape: (303, 14)
Missing values: all zeros (clean dataset)
Target: 1=165, 0=138 (balanced)
Training set shape: (242, 13)
Test set shape:     (61, 13)
Logistic Regression Accuracy: 80.33%
Decision Tree Accuracy:       75.41%
Logistic Regression AUC: 0.87
Decision Tree AUC:       0.82
Top Features: cp, thal, oldpeak, ca, thalach ...
```

### Graphs Produced — Total 8 Graphs

| # | Graph | Purpose |
|---|-------|---------|
| 1 | Count Plot | Target class distribution |
| 2 | Correlation Heatmap | Feature relationships |
| 3 | Pairplot | Selected feature distributions by target |
| 4 | Boxplots | 5 features vs target — outlier detection |
| 5 | Confusion Matrix (LR) | Logistic Regression predictions breakdown |
| 6 | Confusion Matrix (DT) | Decision Tree predictions breakdown |
| 7 | ROC Curves | Both models performance comparison |
| 8 | Feature Importance | Coefficients and importance scores |
| 9 | Decision Tree Structure | Visual tree diagram (max_depth=3) |

---

## Key Results and Findings

| Metric | Logistic Regression | Decision Tree |
|--------|--------------------:|-------------:|
| Accuracy | 80.33% | 75.41% |
| AUC Score | 0.87 | 0.82 |

### Top Important Features (Both Models Agree)

1. **cp** — Chest pain type (strongest predictor)
2. **thal** — Blood disorder type
3. **oldpeak** — ST depression
4. **ca** — Number of major vessels
5. **thalach** — Maximum heart rate achieved

### Key Observations
1. Logistic Regression outperforms Decision Tree on this dataset.
2. Dataset is balanced — 165 disease vs 138 non-disease cases.
3. No missing values found — dataset is clean and ready.
4. Stratified split ensures balanced classes in train and test sets.
5. cp (chest pain type) is the single most important feature in both models.

---

## Evaluation Metrics Explained

| Metric | Value | Meaning |
|--------|-------|---------|
| Accuracy | 80.33% | % of correct predictions out of total |
| AUC-ROC | 0.87 | 1.0 = perfect, 0.5 = random guessing |
| Precision | 0.81 | Of all predicted positives, how many were correct |
| Recall | 0.80 | Of all actual positives, how many were found |
| F1-Score | 0.80 | Harmonic mean of precision and recall |

---

## Conclusion

Logistic Regression is the better model — simpler, more
interpretable, and achieves higher AUC (0.87) compared to
Decision Tree (0.82). Chest pain type (cp), blood disorder
type (thal), and ST depression (oldpeak) are the most
informative features for predicting heart disease risk.

Further improvements can be made using ensemble methods
like Random Forest or XGBoost, or deep learning approaches.

---

## Author

**Name:** Rimsha Aslam    **Internship:** AI/ML Engineering Intern      **Organization:** DevelopersHub Corporation
