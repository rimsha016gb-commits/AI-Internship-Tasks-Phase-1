# Task 2: Stock Price Prediction  

---

## Objective

Use historical stock market data to predict the next day's
closing price of Apple Inc. (AAPL) stock using a
Linear Regression machine learning model.

---

## Dataset

| Property | Details |
|----------|---------|
| Name | Apple (AAPL) Stock Market Data |
| Source | Yahoo Finance via yfinance library |
| Period | January 2022 – January 2024 |
| Size | ~501 rows × 5 columns |
| Missing Values | None |

### Dataset Columns

| Column | Description |
|--------|-------------|
| Open | Stock price when market opened that day |
| High | Highest price reached during that day |
| Low | Lowest price reached during that day |
| Close | Stock price when market closed that day |
| Volume | Total number of shares traded that day |
| Target | Next day's closing price (created by us) |

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| pandas | Data loading and manipulation |
| numpy | Mathematical operations |
| matplotlib | Plotting actual vs predicted prices |
| yfinance | Downloading stock data from Yahoo Finance |
| scikit-learn | Linear Regression model, train/test split, metrics |

---

## Install Requirements

```
pip install -r requirements.txt
```

---

## Model Applied

**Linear Regression** (from scikit-learn)

| Property | Details |
|----------|---------|
| Model Type | Regression |
| Features (inputs) | Open, High, Low, Volume |
| Target (output) | Next day's Close price |
| Train Split | 80% |
| Test Split | 20% |
| shuffle | False (time order preserved) |

---

## How to Run

### Option 1 — Run Python Script
```
python task2_stock_price_prediction.py
```

### Option 2 — Run Jupyter Notebook
```
jupyter notebook Task2_Stock_Price_Prediction.ipynb
```

### Option 3 — VS Code
Open the `.ipynb` file in VS Code and click **Run All**

> Internet connection required to download stock data via yfinance

---

## Output

| Output | Description |
|--------|-------------|
| MAE | Mean Absolute Error — average price prediction error |
| RMSE | Root Mean Squared Error — penalizes large errors more |
| Line Chart | Actual vs Predicted closing price comparison |

---

## Key Results and Findings

1. **MAE: ~1.87** — average prediction error of only $1.87
2. **RMSE: ~2.39** — model handles most predictions accurately
3. **~1% error rate** on stock priced around $150–$180
4. **Open, High, Low** are strongest predictors of next day price
5. **Volume** contributes less compared to price-based features
6. **Predicted line** closely tracks the actual price line in the plot

---

## Visualizations Produced

| Plot | Purpose |
|------|---------|
| Actual vs Predicted Line Chart | Compare real vs predicted closing prices |

---

## Evaluation Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| MAE | ~1.87 | Average dollar error in predictions |
| RMSE | ~2.39 | Error with penalty for large mistakes |

---

## Conclusion

The Linear Regression model successfully predicts the next
day's closing price of Apple stock with high accuracy.
Price-based features (Open, High, Low) are the most
informative inputs. This model serves as a strong baseline
for short-term stock price prediction and can be further
improved with Random Forest or LSTM neural networks.

---

## Author

**Name:** Rimsha Aslam  **Internship:** AI/ML Engineering Intern   **Organization:** DevelopersHub Corporation
