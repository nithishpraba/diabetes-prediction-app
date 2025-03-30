import streamlit as st
import requests

# Set the title
st.title("🩺 Diabetes Prediction Web App")

# Input fields
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, format="%.2f")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)

# Define API base URL
API_BASE = "http://127.0.0.1:5000"

# Predict Button
if st.button("Predict Diabetes Risk"):
    try:
        response = requests.post(f"{API_BASE}/predict", json={
            "glucose": glucose,
            "bmi": bmi,
            "pregnancies": pregnancies
        })

        if response.status_code == 200:
            result = response.json()
            risk = "High" if result.get("diabetes_risk") == 1 else "Low"
            st.success(f"🩸 Diabetes Risk: **{risk}**")
        else:
            st.error(f"❌ API Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Is the Flask server running?")

# View Past Predictions Button
if st.button("View Past Predictions"):
    try:
        response = requests.get(f"{API_BASE}/predictions")

        if response.status_code == 200:
            predictions = response.json()
            if predictions:
                st.write("📊 **Past Predictions:**")
                st.table(predictions)
            else:
                st.info("No past predictions found.")
        else:
            st.error(f"❌ API Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the API. Is the Flask server running?")
