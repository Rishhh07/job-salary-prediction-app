#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('salary_model.joblib')

st.title("💼 Job Salary Prediction App")
st.write("Enter job details below to predict the expected salary:")

# Input fields
experience_level = st.selectbox("Experience Level", ['EN', 'MI', 'SE', 'EX'])
employment_type = st.selectbox("Employment Type", ['FT', 'PT', 'CT', 'FL'])
job_title = st.text_input("Job Title (e.g. Data Scientist)")
company_location = st.text_input("Company Location (e.g. US)")
remote_ratio = st.slider("Remote Work (%)", 0, 100, step=25)
company_size = st.selectbox("Company Size", ['S', 'M', 'L'])

# Submit button
if st.button("Predict Salary"):
    # Create a DataFrame with the inputs
    input_data = pd.DataFrame({
        'experience_level': [experience_level],
        'employment_type': [employment_type],
        'job_title': [job_title],
        'company_location': [company_location],
        'remote_ratio': [remote_ratio],
        'company_size': [company_size]
    })

    # Optional: preprocess if needed (e.g., one-hot encode)
    # If you trained model on processed data, you must replicate same preprocessing

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Salary: ${int(prediction):,}")


# In[ ]:




