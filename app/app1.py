import streamlit as st
import joblib
import pandas as pd

st.title("🩺 Diabetes Prediction Web App")

# Inputs matching model features
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0, format="%.2f")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=2)

# Categorical options
age_group = st.selectbox("Age Group", ["Middle-aged", "Senior", "Young"])
bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)

# Convert to one-hot format to match training
input_dict = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age_Group_Middle-aged': 1 if age_group == "Middle-aged" else 0,
    'Age_Group_Senior': 1 if age_group == "Senior" else 0,
    'BMI_Category_Overweight': 1 if bmi_category == "Overweight" else 0,
    'BMI_Category_Obese': 1 if bmi_category == "Obese" else 0,
}

input_df = pd.DataFrame([input_dict])

# Load pipeline model
model = joblib.load("model/diabetes_pipeline_model.pkl")

if st.button("Predict Diabetes Risk"):
    prediction = model.predict(input_df)[0]
    risk = "High" if prediction == 1 else "Low"
    st.success(f"🩸 Diabetes Risk: **{risk}**")
