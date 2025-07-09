import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fire_sentinel_model.pkl")

st.set_page_config(page_title="🔥 FireSentinel 2.0", layout="centered")
st.title("🔥 FireSentinel 2.0 - Wildfire Risk Prediction")
st.write("This app predicts wildfire risk and allows you to control the probability threshold for fire alerts.")

# Fire threshold control in sidebar
st.sidebar.header("⚙️ Fire Detection Settings")
threshold = st.sidebar.slider("Fire Risk Alert Threshold (%)", min_value=0, max_value=100, value=50)

# 🌲 Vegetation & Location
st.header("🌲 Vegetation & Geography")
ndvi = st.slider("NDVI (Greenness Index)", 0.0, 1.0, 0.5)
elevation = st.slider("Elevation (m)", 0.0, 3000.0, 500.0)
population = st.slider("Population Density", 0.0, 10000.0, 100.0)
prev_fire = st.selectbox("Previous Fire (Yes=1, No=0)", options=[0.0, 1.0])

# 🌦️ Weather
st.header("🌦️ Weather Conditions")
tmmn = st.slider("Min Temperature (°C)", -20.0, 50.0, 15.0)
tmmx = st.slider("Max Temperature (°C)", -10.0, 60.0, 35.0)
pr = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
sph = st.slider("Specific Humidity", 0.0, 1.0, 0.5)
vs = st.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0)
pdsi = st.slider("PDSI (Drought Index)", -5.0, 5.0, 0.0)
erc = st.slider("ERC (Energy Release Component)", 0.0, 100.0, 20.0)
th = st.slider("TH Index", 250.0, 350.0, 300.0)

# Input data
input_data = pd.DataFrame([{
    'NDVI': ndvi,
    'PrevFireMask': prev_fire,
    'elevation': elevation,
    'erc': erc,
    'pdsi': pdsi,
    'population': population,
    'pr': pr,
    'sph': sph,
    'th': th,
    'tmmn': tmmn,
    'tmmx': tmmx,
    'vs': vs
}])

# Predict
if st.button("🔍 Predict Fire Risk"):
    try:
        prob_fire = model.predict_proba(input_data)[0][1] * 100
        st.info(f"🔥 Fire Probability: **{prob_fire:.2f}%**")

        if prob_fire >= threshold:
            st.error("🔥 High Fire Risk Detected!")
        else:
            st.success("✅ No Fire Risk Detected.")
    except Exception as e:
        st.warning(f"⚠️ Prediction failed: {e}")
