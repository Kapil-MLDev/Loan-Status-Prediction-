import streamlit as st
import joblib
import numpy as np


st.image("C:\\Users\\ASUS\\Documents\\ML projects\loan prediction\\home.jpg",width=150) 


with open("C:\\Users\\ASUS\\Documents\\ML projects\\loan prediction\\loan_status_model.pkl", "rb") as file:
    model = joblib.load(file)

with open("C:\\Users\\ASUS\\Documents\\ML projects\\loan prediction\\scaler.pkl", "rb") as file:
    scaler = joblib.load(file)



# Streamlit App Title
st.title("Loan Status Prediction System")

# User Inputs
account_number = st.text_input("Account Number")
full_name = st.text_input("Full Name")
gender = st.selectbox("Gender", ("Male", "Female"))
marital_status = st.selectbox("Marital Status", ("Married", "Single"))
dependents = st.number_input("Dependents", min_value=0, max_value=10, step=1)
education = st.selectbox("Education", ("Graduate", "Not Graduate"))
employment_status = st.selectbox("Employment Status", ("Employed", "Self Employed", "Unemployed"))
property_area = st.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
applicant_income = st.number_input("Applicant's Monthly Income (LKR)")
coapplicant_income = st.number_input("Co-Applicant's Monthly Income (LKR)")
loan_amount = st.number_input("Loan Amount (LKR)")
loan_duration = st.number_input("Loan Duration (Months)")

# Encode Inputs
gender = 1 if gender == "Male" else 0
marital_status = 1 if marital_status == "Married" else 0
education = 1 if education == "Graduate" else 0
employment_status = 1 if employment_status == "Employed" else (2 if employment_status == "Self Employed" else 0)
property_area = {"Urban": 2, "Rural": 0, "Semiurban": 1}[property_area]

# Prediction
if st.button("Submit"):
    # Scale the inputs before prediction
    features = np.array([[gender, marital_status, dependents, education, employment_status,
                          property_area, credit_score, applicant_income, coapplicant_income,
                          loan_amount, loan_duration]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.success(f"Loan Approved for {full_name}")
    else:
        st.error(f"Loan Rejected for {full_name}")
