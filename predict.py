import pandas as pd
import numpy as np
import joblib

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
                 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
                 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']

print("Enter feature values:")

user_inputs = []
for feat in feature_names:
    val = float(input(f"{feat}: "))
    user_inputs.append(val)

# Create DataFrame with proper column names
input_df = pd.DataFrame([user_inputs], columns=feature_names)

# Scale with scaler (which expects DataFrame with column names)
input_scaled = scaler.transform(input_df)

prob = model.predict_proba(input_scaled)[0][1]
pred = int(prob >= 0.4)

print(f"\nPrediction: {'Malignant' if pred == 1 else 'Benign'}")
print(f"Probability of malignancy: {prob:.4f}")
