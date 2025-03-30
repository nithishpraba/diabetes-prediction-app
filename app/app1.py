import streamlit as st
import joblib
import pandas as pd

# Load the pipeline model
try:
    model = joblib.load("model/diabetes_pipeline_model.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Please check the file path.")
    st.stop()

st.title("🩺 Diabetes Prediction Web App (Pipeline Version)")

# Input fields
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, format="%.2f")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)
age = st.number_input("Age", min_value=10, max_value=100, value=35)

# Predict Button
if st.button("Predict Diabetes Risk"):
    try:
        input_data = pd.DataFrame({
            "Glucose": [glucose],
            "BMI": [bmi],
            "Pregnancies": [pregnancies],
            "Age": [age],
            # Other one-hot encoded columns are handled inside the pipeline
            # We do NOT manually add them here anymore!
        })

        # Predict using pipeline
        prediction = model.predict(input_data)[0]
        risk = "High" if prediction == 1 else "Low"
        st.success(f"🩸 Diabetes Risk: **{risk}**")

    except Exception as e:
        st.error(f"❌ Error: {e}")
