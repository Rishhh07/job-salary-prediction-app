import streamlit as st
import pandas as pd
import joblib
import os

# --- Load model and necessary objects ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'salary_model.joblib')
columns_path = os.path.join(current_dir, 'model_columns.joblib')  # NEW

st.write("Current working dir:", os.getcwd())
st.write("Model path:", model_path)
st.write("File exists?", os.path.isfile(model_path))

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)  # Load saved columns from training

# --- Streamlit UI ---
st.title("💼 Job Salary Prediction App")
st.write("This app predicts salaries based on job-related inputs.")

job_title = st.text_input("Job Title")
experience_level = st.selectbox("Experience Level", ['EN', 'MI', 'SE', 'EX'])
employment_type = st.selectbox("Employment Type", ['FT', 'PT', 'CT', 'FL'])
company_size = st.selectbox("Company Size", ['S', 'M', 'L'])
remote_ratio = st.slider("Remote Work %", 0, 100, 0)

if st.button("Predict Salary"):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'job_title': [job_title],
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'company_size': [company_size],
        'remote_ratio': [remote_ratio]
    })

    # Preprocess input using one-hot encoding to match training format
    input_encoded = pd.get_dummies(input_df)

    # Add any missing columns from training
    for col in model_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]  # Ensure correct order

    # Make prediction
    prediction = model.predict(input_encoded)
    st.success(f"🎯 Predicted Salary (USD): ${int(prediction[0])}")

#cd "C:\Users\HP\OneDrive\Desktop\Job_Prediction_Project"
#python -m streamlit run salary_app.py
