import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fire_sentinel_model.pkl")

st.set_page_config(page_title="🔥 FireSentinel - Wildfire Prediction", layout="centered")
st.title("🔥 FireSentinel - Wildfire Prediction App")
st.write("Predict wildfire risk using the trained Random Forest model.")

ndvi = st.slider("NDVI", 0.0, 1.0, 0.5)
tmmn = st.slider("Minimum Temperature (°C)", -20.0, 50.0, 15.0)
tmmx = st.slider("Maximum Temperature (°C)", -10.0, 60.0, 35.0)
sph = st.slider("Specific Humidity", 0.0, 1.0, 0.5)
vs = st.slider("Wind Speed", 0.0, 20.0, 3.0)
pdsi = st.slider("PDSI", -5.0, 5.0, 0.0)
pr = st.slider("Precipitation", 0.0, 50.0, 0.0)
th = st.slider("TH Index", 250.0, 350.0, 300.0)
elevation = st.slider("Elevation", 0.0, 3000.0, 500.0)
population = st.slider("Population Density", 0.0, 10000.0, 100.0)
erc = st.slider("ERC", 0.0, 100.0, 20.0)
prev_fire = st.selectbox("Previous Fire (Yes=1, No=0)", options=[0.0, 1.0])
if st.button("Predict Fire Risk"):
    input_data = pd.DataFrame([[
        ndvi, tmmn, tmmx, sph, vs, pdsi, pr, th,
        elevation, population, erc, prev_fire
    ]], columns=[
        'NDVI', 'tmmn', 'tmmx', 'sph', 'vs', 'pdsi', 'pr', 'th',
        'elevation', 'population', 'erc', 'PrevFireMask'
    ])
    
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🔥 High Fire Risk Detected!")
    else:
        st.success("✅ No Fire Risk Detected.")
