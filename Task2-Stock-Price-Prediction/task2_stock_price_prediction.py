# ============================================================
# Task 2: Predict Future Stock Prices (Short-Term)
# DevelopersHub Corporation – AI/ML Engineering Internship
# Dataset: Apple (AAPL) stock data via yfinance
# Model: Linear Regression
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
import numpy as np

# ── 1. Load Historical Stock Data ────────────────────────────
# Downloading 2 years of Apple stock data
df = yf.download('AAPL', start='2022-01-01', end='2024-01-01')

print("Shape:", df.shape)
print("\nFirst 5 Rows:\n", df.head())
print("\nDataset Info:")
df.info()

# ── 2. Prepare Features and Target ───────────────────────────
# Features: Open, High, Low, Volume  |  Target: Close (next day)
# Shift Close by -1 to predict the NEXT day's closing price
df = df[['Open', 'High', 'Low', 'Volume', 'Close']].copy()
df['Target'] = df['Close'].shift(-1)  # next day's close
df.dropna(inplace=True)               # drop last row (no target)

X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Target']

# ── 3. Train / Test Split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # keep time order
)

# ── 4. Train Linear Regression Model ─────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── 5. Predict ────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── 6. Evaluate ───────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nMean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# ── 7. Plot Actual vs Predicted Closing Prices ───────────────
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label='Actual Close Price', color='blue')
plt.plot(y_pred,        label='Predicted Close Price', color='orange', linestyle='--')
plt.title('AAPL – Actual vs Predicted Next-Day Closing Price')
plt.xlabel('Test Sample Index')
plt.ylabel('Price (USD)')
plt.legend()
plt.tight_layout()
plt.show()
