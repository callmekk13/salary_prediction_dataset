import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load('random_forest_regressor_model.pkl.zip')
except FileNotFoundError:
    st.error("Model file 'random_forest_regressor_model.pkl' not found. Make sure it's in the same directory as this app.py file.")
    st.stop()

st.title('Data Science Salary Predictor')
st.write('Enter the details below to predict the data science salary.')

# Input fields for prediction (matching your model's features)
# Note: For actual deployment, you would need to load and use the original LabelEncoders
# used during training to transform categorical inputs from the user.
# For simplicity, we are using direct numerical inputs here, but they should correspond
# to the encoded values your model expects.

rating = st.slider('Rating', 1.0, 5.0, 3.5)
company_name_encoded = st.number_input('Company Name (Encoded Value)', min_value=0, value=8129)
job_title_encoded = st.number_input('Job Title (Encoded Value)', min_value=0, value=28)
salaries_reported = st.number_input('Salaries Reported', min_value=1, value=3)
location_encoded = st.number_input('Location (Encoded Value)', min_value=0, value=0)
employment_status_encoded = st.number_input('Employment Status (Encoded Value)', min_value=0, value=1)
job_roles_encoded = st.number_input('Job Roles (Encoded Value)', min_value=0, value=0)

# Create a DataFrame for the input
input_data = pd.DataFrame([[rating,
                              company_name_encoded,
                              job_title_encoded,
                              salaries_reported,
                              location_encoded,
                              employment_status_encoded,
                              job_roles_encoded]],
                            columns=['Rating', 'Company Name', 'Job Title', 'Salaries Reported', 'Location', 'Employment Status', 'Job Roles'])

if st.button('Predict Salary'):
    prediction = model.predict(input_data)
    st.success(f'Predicted Salary (Encoded Value): {prediction[0]:.2f}')
    st.warning('Note: The predicted salary is an encoded value. You would need to reverse transform this to get the original salary.')
