import streamlit as st
import pandas as pd
import joblib
import os

# Load trained pipeline safely
MODEL_PATH = "fraud_detection_pipline.pkl"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except AttributeError as e:
        st.error(f"Error loading the model: {e}\nCheck your scikit-learn version or re-save the pipeline.")
        st.stop()
else:
    st.error(f"Model file not found at '{MODEL_PATH}'")
    st.stop()

# App title & description
st.title("Fraud Detection Prediction App")
st.markdown("Please enter the transaction details and click **Predict**")

st.divider()

# User inputs
transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT"]
)

amount = st.number_input(
    "Amount",
    min_value=0.0,
    value=1000.0
)

oldbalanceOrg = st.number_input(
    "Old Balance Origin",
    min_value=0.0,
    value=5000.0
)

newbalanceOrig = st.number_input(
    "New Balance Origin",
    min_value=0.0,
    value=4000.0
)

oldbalanceDest = st.number_input(
    "Old Balance Destination",
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    "New Balance Destination",
    min_value=0.0,
    value=0.0
)

# Predict button
if st.button("Predict Fraud"):
    
    # Create dataframe (must match training columns)
    input_data = pd.DataFrame(
        [{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }]
    )

    # Prediction
    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Display result

    st.divider()

    if prediction == 1:
        st.error(f" FRAUD DETECTED\nProbability: {proba:.2%}")
    else:
        st.success(f" Transaction is NOT Fraud\nProbability: {proba:.2%}")
