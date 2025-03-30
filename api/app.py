from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("decision_tree_model.pkl")

# Initialize Flask app
app = Flask(__name__)

# Database Connection Function
def get_db_connection():
    conn = sqlite3.connect("diabetes_predictions.db")
    conn.row_factory = sqlite3.Row
    return conn

# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Diabetes Prediction API is running!"})

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON data
        data = request.get_json()

        # Extract values from request
        glucose = float(data["glucose"])
        bmi = float(data["bmi"])
        pregnancies = int(data["pregnancies"])

        # Format input for prediction
        input_data = np.array([[glucose, bmi, pregnancies]])

        # Get model prediction
        prediction = model.predict(input_data)[0]

        # Store prediction in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Predictions (user_id, glucose, bmi, pregnancies, diabetes_risk)
            VALUES (?, ?, ?, ?, ?)""",
            (None, glucose, bmi, pregnancies, prediction))
        conn.commit()
        conn.close()

        return jsonify({"diabetes_risk": int(prediction), "message": "Prediction stored successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Retrieve stored predictions
@app.route("/predictions", methods=["GET"])
def get_predictions():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Predictions ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()

        predictions = [dict(row) for row in rows]

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
    
