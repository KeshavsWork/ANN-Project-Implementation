import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.set_page_config(page_title="Customer Churn", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_data
def load_artifacts():
    with open("encoder.pkl", "rb") as f:
        le_gender = pickle.load(f)
    with open("onehotencoder.pkl", "rb") as f:
        ohe_geo = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return le_gender, ohe_geo, scaler

model = load_model()
label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

st.title("Web App on Customer Churn Prediction")

# Safe defaults from fitted encoders
geo_options = list(onehot_encoder_geo.categories_[0])
gender_options = list(label_encoder_gender.classes_)

col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox("Geography", geo_options)
    gender = st.selectbox("Gender", gender_options)
    age = st.slider("Age", 18, 92, 30)
    tenure = st.slider("Tenure", 0, 10, 3)
    num_of_products = st.slider("Number of Products", 1, 4, 1)
with col2:
    credit_score = st.number_input("Credit Score", min_value=0, value=650, step=1)
    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=100.0, format="%.2f")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=100.0, format="%.2f")
    has_cr_card = st.selectbox("Has Credit Card", [0, 1], index=1)
    is_active_member = st.selectbox("Is Active Member", [0, 1], index=1)

# Prepare tabular features
X_tab = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary],
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_cols = onehot_encoder_geo.get_feature_names_out(["Geography"])
geo_df = pd.DataFrame(geo_encoded.toarray(), columns=geo_cols)

# Concatenate ensuring consistent column order
X_full = pd.concat([X_tab.reset_index(drop=True), geo_df], axis=1)

# Align columns to the scaler’s fitted feature names if available
if hasattr(scaler, "feature_names_in_"):
    # Add any missing cols with zeros, order columns
    for col in scaler.feature_names_in_:
        if col not in X_full.columns:
            X_full[col] = 0.0
    X_full = X_full.loc[:, scaler.feature_names_in_]

# Scale
X_scaled = scaler.transform(X_full)

# Predict
if st.button("Predict Churn"):
    pred = model.predict(X_scaled)
    proba = float(pred[0][0])
    st.write(f"Churn Probability: {proba:.2f}")
    if proba > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")
