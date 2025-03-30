import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained Decision Tree model
model = joblib.load("model/decision_tree_model.pkl")

# Set the title
st.title("🩺 Diabetes Prediction Web App")

# Input fields
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, format="%.2f")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)

# Predict Button
if st.button("Predict Diabetes Risk"):
    try:
        # Prepare input for model (match training feature order)
        input_data = pd.DataFrame({
            "Glucose": [glucose],
            "BMI": [bmi],
            "Pregnancies": [pregnancies]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        risk = "High" if prediction == 1 else "Low"
        st.success(f"🩸 Diabetes Risk: **{risk}**")
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
