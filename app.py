import streamlit as st
import joblib
import pandas as pd

st.title('Bankruptcy Prediction App')

xgb_model = joblib.load('xgb_model.joblib')
scaler = joblib.load('scaler.joblib')

feature_cols = ['industrial_risk', ' management_risk', ' financial_flexibility',
       ' credibility', ' competitiveness', ' operating_risk', ' class']

st.write("Enter the risk factors below to predict the likelihood of bankruptcy.")

input_data = {}
for col in feature_cols:
    input_data[col] = st.slider(f'Select {col.replace("_", " ").title()}', 0.0, 1.0, 0.5, 0.1)

input_df = pd.DataFrame([input_data])

scaled_input = scaler.transform(input_df)

if st.button('Predict'):
    prediction = xgb_model.predict(scaled_input)
    prediction_proba = xgb_model.predict_proba(scaled_input)

    st.subheader('Prediction Result:')
    if prediction[0] == 0:
        st.success(f'The model predicts: Non-Bankruptcy (Probability: {prediction_proba[0][0]*100:.2f}%)')
    else:
        st.error(f'The model predicts: Bankruptcy (Probability: {prediction_proba[0][1]*100:.2f}%)')

st.write("\n--- DISCLAIMER ---")
st.write("This prediction is based on a machine learning model and should not be used as the sole basis for financial decisions.")
