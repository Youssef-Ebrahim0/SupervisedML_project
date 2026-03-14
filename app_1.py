# CharityML Donor Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb

st.sidebar.write(f"XGBoost version: {xgb.__version__}")

# ============================================
# FIX XGBOOST COMPATIBILITY
# ============================================

def patch_xgb_model(model):
    """
    Add missing attributes to older XGBoost models
    so they work with newer sklearn versions.
    """
    try:

        if hasattr(model, "named_steps"):
            for name, step in model.named_steps.items():

                if "xgb" in str(type(step)).lower():

                    if not hasattr(step, "gpu_id"):
                        step.gpu_id = -1

                    if not hasattr(step, "n_gpus"):
                        step.n_gpus = 0

                    if not hasattr(step, "predictor"):
                        step.predictor = "cpu_predictor"

        else:

            if not hasattr(model, "gpu_id"):
                model.gpu_id = -1

            if not hasattr(model, "n_gpus"):
                model.n_gpus = 0

            if not hasattr(model, "predictor"):
                model.predictor = "cpu_predictor"

    except Exception as e:
        st.warning(f"Model patch warning: {e}")

    return model


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():
    try:

        model = joblib.load("best_xgb_model.pkl")

        model = patch_xgb_model(model)

        return model

    except Exception as e:

        st.error(f"Error loading model: {e}")
        return None


model = load_model()

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_income(df):

    try:

        st.write("Debug - Input shape:", df.shape)

        processed_df = pd.get_dummies(df)

        if hasattr(model, "named_steps"):

            try:
                xgb_model = model.named_steps[list(model.named_steps.keys())[-1]]

                if hasattr(xgb_model, "feature_names_in_"):

                    required = xgb_model.feature_names_in_

                    for col in required:
                        if col not in processed_df.columns:
                            processed_df[col] = 0

                    processed_df = processed_df[required]

            except:
                pass

        prediction = model.predict(processed_df)[0]

        probability = model.predict_proba(processed_df)[0]

        return prediction, probability

    except Exception as e:

        st.error(f"Prediction error: {e}")

        return None, None


# ============================================
# UI
# ============================================

st.title("CharityML Donor Prediction")

col1, col2 = st.columns([2, 1])

with col1:

    st.subheader("Personal Information")

    age = st.number_input("Age", 17, 90, 35)

    education_num = st.number_input("Years of Education", 1, 16, 10)

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married-civ-spouse",
            "Never-married",
            "Divorced",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
            "Married-AF-spouse"
        ]
    )

    sex = st.radio("Sex", ["Male", "Female"])

    race = st.selectbox(
        "Race",
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
    )

    relationship = st.selectbox(
        "Relationship",
        [
            "Husband",
            "Not-in-family",
            "Wife",
            "Own-child",
            "Unmarried",
            "Other-relative"
        ]
    )

    workclass = st.selectbox(
        "Workclass",
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked"
        ]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving"
        ]
    )

    hours_per_week = st.number_input("Hours per week", 1, 99, 40)

    education_level = st.selectbox(
        "Education Level",
        [
            "Bachelors",
            "HS-grad",
            "11th",
            "Masters",
            "9th",
            "Some-college"
        ]
    )

    native_country = st.selectbox(
        "Native Country",
        [
            "United-States",
            "Mexico",
            "Philippines",
            "Germany",
            "Canada",
            "India"
        ]
    )

    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)

    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)

    if st.button("Predict Donor Potential"):

        input_data = pd.DataFrame([[
            age,
            workclass,
            education_level,
            education_num,
            marital_status,
            occupation,
            relationship,
            race,
            sex,
            capital_gain,
            capital_loss,
            hours_per_week,
            native_country
        ]], columns=[
            "age",
            "workclass",
            "education_level",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country"
        ])

        prediction, probability = predict_income(input_data)

        if prediction is not None:

            if prediction == 1:

                st.success(f"Likely Donor (Confidence {probability[1]*100:.1f}%)")

                st.balloons()

            else:

                st.error(f"Not a Donor (Confidence {probability[0]*100:.1f}%)")

with col2:

    st.subheader("Model Performance")

    st.write("Accuracy: 87.4%")

    st.write("Precision: 81%")

    st.write("Recall: 67%")

    st.write("F1 Score: 0.73")
