import streamlit as st
import pandas as pd
from ml_model import load_data, train_model

# Load and train model
df = load_data("sample_mna_deals.xlsx")
model, feature_order = train_model(df)

st.title("Outcome Predictor")

st.markdown("Enter the details of a new transaction:")

# Input form
input_data = {}
for feature in feature_order:
    if df[feature].dtype == "object":
        input_data[feature] = st.selectbox(f"{feature}:", sorted(df[feature].dropna().unique()))
    elif feature == "Hostile":
        input_data[feature] = st.selectbox("Hostile:", [0, 1])
    else:
        input_data[feature] = st.number_input(f"{feature}:", min_value=0.0)

# Predict
if st.button("Predict Outcome Probability"):
    input_df = pd.DataFrame([input_data])
    proba = model.predict_proba(input_df)[0][1]
    st.success(f"🔮 Probability of Success: {proba:.2%}")