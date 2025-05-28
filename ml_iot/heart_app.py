import streamlit as st
import pandas as pd
import joblib

# Load model and model columns
model = joblib.load("heart_disease_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter your health information below:")

# Input form
age = st.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["TA", "ATA", "NAP", "ASY"])
chol = st.number_input("Cholesterol (mg/dL)", min_value=0, value=200)
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])

# Add all other fields you used during training below if any (e.g., RestingBP, FastingBS, etc.)

# Prepare input dictionary
input_dict = {
    "Age": age,
    "Sex": sex,
    "ChestPainType": chest_pain,
    "Cholesterol": chol,
    "MaxHR": max_hr,
    "ExerciseAngina": exercise_angina,
    # Add other inputs here...
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][prediction]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Confidence: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Confidence: {prediction_proba:.2f})")
