from scipy.__config__ import show
import streamlit as st
import pandas as pd
import joblib


xgb_model = joblib.load("Credit_Card_model.pkl")
scaler = joblib.load("scaler.pkl")
shap_values=joblib.load("shap_values.pkl")

expected_columns = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
                    'V10','V11','V12','V13','V14','V15','V16','V17','V18',
                    'V19','V20','V21','V22','V23','V24','V25','V26','V27',
                    'V28','Amount']

st.title("Credit Card Fraud Detection Dashboard")
st.write("Upload a CSV file of transactions to predict fraud probabilities and see SHAP explanations.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")


if uploaded_file:
    data = pd.read_csv(uploaded_file)
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)
    data = data[expected_columns]
    if data.isnull().sum().sum() > 0:
        st.warning("Missing values detected! Filling with 0.")
        data = data.fillna(0)
    data['Amount'] = scaler.transform(data['Amount'].values.reshape(-1,1))
    data['Time'] = scaler.transform(data['Time'].values.reshape(-1,1))

    preds = xgb_model.predict_proba(data)[:,1]
    data['Fraud_Probability'] = preds
    st.subheader("Predictions")
    st.dataframe(data.head())

    st.subheader("Top 15 High Fraud Transactions")
    top_fraud = data.sort_values(by='Fraud_Probability', ascending=False).head(15)
    st.dataframe(top_fraud)