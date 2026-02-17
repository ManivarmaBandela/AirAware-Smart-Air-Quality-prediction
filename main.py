import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("aqi.csv")
df.columns = df.columns.str.strip()

# Rename columns (based on your dataset)
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

# Convert month column
df["Mounths"] = pd.to_datetime(df["Mounths"], format="%b-%y")
df["Month"] = df["Mounths"].dt.month
df["Year"] = df["Mounths"].dt.year

# Drop unnecessary columns
df.drop(["Id", "Mounths"], axis=1, inplace=True)

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Define Features and Target
X = df.drop("AQI", axis=1)
y = df["AQI"]

# Train Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save Model
joblib.dump(model, "aqi_model.pkl")

print("Model trained and saved successfully as aqi_model.pkl")
