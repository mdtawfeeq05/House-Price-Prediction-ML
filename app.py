
import streamlit as st
import pandas as pd
import joblib

# Load trained ML model
model = joblib.load("house_price_model.pkl")

st.title("House Price Prediction")
st.write("Predict house prices using Machine Learning")

# User inputs
area = st.number_input(
    "Enter House Area (sq. ft.)",
    min_value=100,
    value=1750
)

bedrooms = st.number_input(
    "Enter Number of Bedrooms",
    min_value=1,
    value=3
)

age = st.number_input(
    "Enter House Age (years)",
    min_value=0,
    value=4
)

# Prediction button
if st.button("Predict Price"):

    house = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "age": [age]
    })

    prediction = model.predict(house)

    st.success(
        f"Predicted House Price: "
        f"₹{prediction[0]:.2f} lakh"
    )

st.caption(
    "Educational demo using synthetic housing data. "
    "Not a real property valuation."
)
