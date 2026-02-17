import streamlit as st
import pandas as pd
import numpy as np
import joblib

import base64

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("background.jpg")

st.title("🌍 Air Aware - Public AQI Information System")
st.image("banner.jpg", use_container_width=True)

#Styling

st.markdown("""
<div style="
    background-color: rgba(255,255,255,0.95);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
">
    <h1 style="color:black;">🌍 Air Aware - Public AQI Information System</h1>
    <p style="color:black;">
    This system predicts Air Quality Index (AQI) using historical pollution data.
    Select a month and year to view predicted AQI and health advisory.
    </p>
</div>
""", unsafe_allow_html=True)

#names styling----------
st.markdown("""
<style>

/* Selectbox label text */
label {
    color: black !important;
    font-weight: bold;
}

/* Dropdown selected value */
div[data-baseweb="select"] {
    color: black !important;
}

/* Dropdown input text */
div[data-baseweb="select"] span {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------
# Load Dataset & Model
# ------------------------------------------

df = pd.read_csv("aqi.csv")
df.columns = df.columns.str.strip()

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

df["Mounths"] = pd.to_datetime(df["Mounths"], format="%b-%y")
df["Month"] = df["Mounths"].dt.month
df["Year"] = df["Mounths"].dt.year
df.drop(["Id", "Mounths"], axis=1, inplace=True)

df.fillna(df.mean(numeric_only=True), inplace=True)

model = joblib.load("aqi_model.pkl")

# ------------------------------------------
# AQI Category Function
# ------------------------------------------

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# ------------------------------------------
# Health Advisory
# ------------------------------------------

def health_advisory(aqi):
    if aqi <= 50:
        return "Safe for outdoor activities."
    elif aqi <= 100:
        return "Minor breathing discomfort for sensitive people."
    elif aqi <= 200:
        return "Reduce prolonged outdoor exertion."
    elif aqi <= 300:
        return "Avoid outdoor activities."
    elif aqi <= 400:
        return "Stay indoors and wear a mask."
    else:
        return "Health emergency. Avoid going outside."

# ------------------------------------------
# UI Inputs
# ------------------------------------------

st.markdown(
    '<h3 style="color:black;">Select Month and Year</h3>',
    unsafe_allow_html=True
)


month = st.selectbox("Select Month", list(range(1, 13)))
year = st.selectbox("Select Year", sorted(df["Year"].unique()))

# ------------------------------------------
# Prediction Button
# ------------------------------------------

if st.button("Predict AQI"):

    filtered = df[(df["Month"] == month) & (df["Year"] == year)]

    if not filtered.empty:

        input_data = filtered.drop("AQI", axis=1).mean().values.reshape(1, -1)

        prediction = model.predict(input_data)[0]
        category = categorize_aqi(prediction)
        advice = health_advisory(prediction)

        st.markdown(
    f'<h3 style="color:black;">Predicted AQI: {round(prediction,2)}</h3>',
    unsafe_allow_html=True
)
        
        if prediction <= 50:
            color = "#00cc44"   # Bright Green
        elif prediction <= 100:
            color = "#ffcc00"   # Bright Yellow
        elif prediction <= 200:
            color = "#ff8800"   # Bright Orange
        else:
            color = "#ff3333"   # Bright Red

        st.markdown(
    f"""
    <div style="
        background-color: {color};
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: black;
        font-weight: bold;
        font-size: 20px;
    ">
        Category: {category}
    </div>
    """,
    unsafe_allow_html=True
)


        st.markdown(
    f'<h3 style="color:black;">🏥 Health Advisory</h3>'
    f'<p style="color:black; font-size:18px;">{advice}</p>',
    unsafe_allow_html=True
)
        st.markdown(
    '<h3 style="color:black;">AQI Trend Over Time</h3>',
    unsafe_allow_html=True
)

        df_sorted = df.sort_values(by=["Year", "Month"])
        st.line_chart(df_sorted["AQI"])

    else:
        st.error("No data available for selected Month & Year.")