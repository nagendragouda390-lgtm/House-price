import streamlit as st
import joblib
import numpy as np

# Page Config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load Model
model = joblib.load("house_price_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("🏠 House Price Prediction App")
st.markdown("Predict house prices using Machine Learning")

st.divider()

# Input Section
col1, col2 = st.columns(2)

with col1:
    square_footage = st.number_input(
        "📏 Square Footage",
        min_value=500,
        max_value=10000,
        value=1360
    )

    num_bedrooms = st.slider(
        "🛏 Number of Bedrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    num_bathrooms = st.slider(
        "🚿 Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=1
    )

    year_built = st.number_input(
        "🏗 Year Built",
        min_value=1900,
        max_value=2025,
        value=1981
    )

with col2:
    lot_size = st.number_input(
        "🌳 Lot Size",
        min_value=0.0,
        value=0.599637,
        step=0.1
    )

    garage_size = st.slider(
        "🚗 Garage Size",
        min_value=0,
        max_value=5,
        value=0
    )

    neighborhood_quality = st.slider(
        "⭐ Neighborhood Quality",
        min_value=1,
        max_value=10,
        value=5
    )

st.divider()

# Prediction Button
if st.button("🔮 Predict House Price", use_container_width=True):

    features = np.array([[
        square_footage,
        num_bedrooms,
        num_bathrooms,
        year_built,
        lot_size,
        garage_size,
        neighborhood_quality
    ]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]

    st.success(
        f"🏡 Estimated House Price: ${prediction:,.2f}"
    )

    st.balloons()

# Footer
st.markdown("---")
st.caption("Built with Scikit-Learn and Streamlit")