import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop(columns=["DealID", "Outcome"])
    y = df["Outcome"]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X, y)
    return model, categorical_cols + numeric_cols

# UI
st.title("Outcome Predictor")

df = load_data("sample_mna_deals.xlsx")
model, feature_order = train_model(df)

st.markdown("Enter details for a new observation:")

input_data = {}
for feature in feature_order:
    if df[feature].dtype == "object":
        input_data[feature] = st.selectbox(f"{feature}:", sorted(df[feature].dropna().unique()))
    elif feature == "Hostile":
        input_data[feature] = st.selectbox("Hostile:", [0, 1])
    else:
        input_data[feature] = st.number_input(f"{feature}:", min_value=0.0)

if st.button("Predict Outcome Probability"):
    input_df = pd.DataFrame([input_data])
    proba = model.predict_proba(input_df)[0][1]
    st.success(f"🔮 Probability of Success: {proba:.2%}")
