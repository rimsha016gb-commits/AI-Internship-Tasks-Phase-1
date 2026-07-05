# Task 1: Iris-Exploration
---

## Objective

Load, inspect, and visualize the Iris dataset to understand
data trends and distributions using Python libraries.

---

## Dataset

- **Name:** Iris Dataset
- **Source:** Loaded directly via seaborn library
- **Size:** 150 rows × 5 columns
- **No missing values**

### Dataset Columns

| Column | Description |
|--------|-------------|
| sepal_length | Length of the sepal in cm |
| sepal_width | Width of the sepal in cm |
| petal_length | Length of the petal in cm |
| petal_width | Width of the petal in cm |
| species | Flower species (setosa, versicolor, virginica) |

---

## Libraries Used

```
pandas
matplotlib
seaborn
```

Install with:
```
pip install pandas matplotlib seaborn
```

---

## Models / Methods Applied

- Data loading and inspection using **pandas**
- Missing value and duplicate checking
- Descriptive statistics using `.info()` and `.describe()`
- **Scatter plots** — to show relationships between features
- **Histograms** — to show value distributions per species
- **Box plots** — to identify outliers per species

---

## How to Run

1. Clone this repository
2. Open `Task1_Iris_Exploration.ipynb` in VS Code or Jupyter
3. Run all cells from top to bottom

```
jupyter notebook Task1_Iris_Exploration.ipynb
```

Or in VS Code — open the file and click **Run All**

---

## Key Results and Findings

1. **Setosa** is clearly separable from other species
   based on petal measurements alone.

2. **Petal length and petal width** are the strongest
   distinguishing features among the three species.

3. **Sepal measurements** show overlap between
   Versicolor and Virginica — less useful for classification.

4. **Histograms** show distinct and different distributions
   for each species across all four features.

5. **Box plots** reveal a few mild outliers in the
   Sepal Width of Iris Setosa species.

6. Dataset is **clean** — no missing values, no major issues.

---

## Visualizations Produced

| Plot | Purpose |
|------|---------|
| Scatter: Sepal Length vs Sepal Width | Feature relationship by species |
| Scatter: Petal Length vs Petal Width | Feature relationship by species |
| Histogram: All 4 features | Value distribution per species |
| Box Plot: All 4 features | Outlier detection per species |

---

## Conclusion

The Iris dataset demonstrates clear patterns among flower species.
Petal measurements are the most informative features and will be
highly useful for future classification tasks. This exploratory
analysis confirms the dataset is clean, balanced, and ready
for machine learning model training.

---

## Author

**Name:** RimshaAslam
**Internship:** AI/ML Engineering Intern
**Organization:** DevelopersHub Corporation

