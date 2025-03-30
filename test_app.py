import unittest
import joblib
import pandas as pd

class TestDiabetesPrediction(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load("model/diabetes_pipeline_model.pkl")

    def test_model_loads(self):
        self.assertIsNotNone(self.model)

    def test_prediction_with_valid_input(self):
        input_data = pd.DataFrame([{
            'Pregnancies': 2,
            'Glucose': 120,
            'BMI': 25.0,
            'DiabetesPedigreeFunction': 0.5,
            'Age_Group_Middle-aged': 1,
            'Age_Group_Senior': 0,
            'BMI_Category_Overweight': 0,
            'BMI_Category_Obese': 1
        }])
        prediction = self.model.predict(input_data)
        self.assertIn(prediction[0], [0, 1])  # Should return a class

if __name__ == '__main__':
    unittest.main()
