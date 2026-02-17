# ==========================================
# AIR AWARE - TRAINING & ANALYSIS SCRIPT
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("aqi.csv")
df.columns = df.columns.str.strip()

# Rename columns
df.rename(columns={
    "PM10 in æg/m3": "PM10",
    "SO2 in æg/m3": "SO2",
    "NOx  in æg/m3": "NOx",
    "PM2.5  in æg/m3": "PM25",
    "Ammonia - NH3  in æg/m3": "NH3",
    "O3   in æg/m3": "O3",
    "CO  in mg/m3": "CO",
    "Benzene  in æg/m3": "Benzene"
}, inplace=True)

# Convert Month column
df["Mounths"] = pd.to_datetime(df["Mounths"], format="%b-%y")
df["Month"] = df["Mounths"].dt.month
df["Year"] = df["Mounths"].dt.year

df.drop(["Id", "Mounths"], axis=1, inplace=True)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nDATA AFTER CLEANING")
print(df.head())

# ------------------------------------------
# Statistical Summary
# ------------------------------------------

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ------------------------------------------
# AQI Trend Visualization
# ------------------------------------------

df_sorted = df.sort_values(by=["Year", "Month"])

plt.figure(figsize=(10,5))
plt.plot(df_sorted["AQI"], marker="o")
plt.title("AQI Trend Over Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.grid(True)
plt.show()

# ------------------------------------------
# Correlation Heatmap
# ------------------------------------------

plt.figure(figsize=(10,6))
sns.heatmap(df.select_dtypes(include=["number"]).corr(),
            annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ------------------------------------------
# Train-Test Split
# ------------------------------------------

X = df.drop("AQI", axis=1)
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------
# Random Forest Model
# ------------------------------------------

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRANDOM FOREST PERFORMANCE")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("R2 Score:", r2_score(y_test, rf_pred))

# ------------------------------------------
# Cross Validation
# ------------------------------------------

cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\nCROSS VALIDATION MEAN:", cv_scores.mean())

# ------------------------------------------
# Linear Regression Comparison
# ------------------------------------------

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLINEAR REGRESSION PERFORMANCE")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("MSE:", mean_squared_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))

# ------------------------------------------
# Feature Importance
# ------------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFEATURE IMPORTANCE")
print(importance)

plt.figure(figsize=(8,5))
importance.sort_values(by="Importance").plot(
    kind="barh", x="Feature", y="Importance", legend=False
)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()
