import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Predictor")

# app.py is inside models/ so paths are relative to models/
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    return {
        'Linear Regression': joblib.load(os.path.join(BASE, 'linear_regression.pkl')),
        'Ridge':             joblib.load(os.path.join(BASE, 'ridge.pkl')),
        'Lasso':             joblib.load(os.path.join(BASE, 'lasso.pkl')),
    }

models = load_models()

st.sidebar.header("House Features")
sqft     = st.sidebar.slider("Living Area (sq ft)", 500, 5000, 1500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 5, 3)
baths    = st.sidebar.slider("Bathrooms", 1, 4, 2)
year     = st.sidebar.slider("Year Built", 1900, 2010, 1990)
garage   = st.sidebar.slider("Garage Cars", 0, 3, 1)
qual     = st.sidebar.slider("Overall Quality (1-10)", 1, 10, 6)

input_df = pd.DataFrame(
    [[sqft, bedrooms, baths, year, garage, qual]],
    columns=['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'GarageCars', 'OverallQual']
)

if st.button("Predict"):
    cols = st.columns(3)
    preds = []
    for col, (name, model) in zip(cols, models.items()):
        pred = model.predict(input_df)[0]
        preds.append(pred)
        col.metric(name, f"${pred:,.0f}")
    st.success(f"Ensemble Average: ${np.mean(preds):,.0f}")

heatmap = os.path.join(BASE, '..', 'plots', 'correlation_heatmap.png')
if os.path.exists(heatmap):
    st.image(heatmap, caption='Feature Correlations')