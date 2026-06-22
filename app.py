import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load Model
with open("model2.pkl", "rb") as file:
    model = pickle.load(file)

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
        max_value=2026,
        value=1981
    )

with col2:
    lot_size = st.number_input(
        "🌳 Lot Size",
        min_value=0.0,
        value=0.6,
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

    prediction = model.predict(features)[0]

    st.success(
        f"🏡 Estimated House Price: ${prediction:,.2f}"
    )

    st.balloons()

# Footer
st.markdown("---")
st.caption("Built with Scikit-Learn and Streamlit")
