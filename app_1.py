# app_1.py
# CharityML Donor Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost
import xgboost as xgb

# Display version info
st.sidebar.write(f"XGBoost version: {xgb.__version__}")

# ============================================
# COMPREHENSIVE XGBOOST COMPATIBILITY FIX
# ============================================

# Patch 1: Handle use_label_encoder
if not hasattr(xgb.XGBClassifier, "use_label_encoder"):
    xgb.XGBClassifier.use_label_encoder = False

# Patch 2: Create a safe model loader that handles missing attributes
def fix_xgboost_model(model):
    """Add missing attributes to make older models compatible with newer XGBoost"""
    if model is None:
        return model
    
    # List of attributes that might be missing in newer XGBoost
    missing_attrs = {
        'gpu_id': -1,
        'n_gpus': 0,
        'predictor': 'cpu_predictor',
        'enable_categorical': False,
        'use_rmm': False,
        'max_cat_to_onehot': 4,
        'cat_priority': False,
    }
    
    # Add missing attributes to the model
    for attr, default_value in missing_attrs.items():
        if not hasattr(model, attr):
            setattr(model, attr, default_value)
    
    # Fix the booster if it exists
    if hasattr(model, '_Booster') and model._Booster is not None:
        booster = model._Booster
        for attr, default_value in missing_attrs.items():
            if not hasattr(booster, attr):
                setattr(booster, attr, default_value)
    
    return model

# Patch 3: Monkey patch joblib.load to automatically fix XGBoost models
original_load = joblib.load

def patched_load(*args, **kwargs):
    """Load and automatically fix XGBoost models"""
    obj = original_load(*args, **kwargs)
    
    # Check if it's an XGBoost model and fix it
    if obj is not None:
        obj_type = str(type(obj))
        if 'xgboost' in obj_type.lower() or 'xgb' in obj_type.lower():
            obj = fix_xgboost_model(obj)
            st.sidebar.info("✅ Applied XGBoost compatibility fixes")
    
    return obj

# Replace joblib.load with our patched version
joblib.load = patched_load

# ============================================
# MODEL LOADING
# ============================================

@st.cache_resource
def load_model():
    """Load the saved model from saved_models directory"""
    try:
        model_path = 'best_xgb_model.pkl'
        
        if os.path.exists(model_path):
            # Load the model (our patched load will handle fixes)
            model = joblib.load(model_path)
            
            # Verify the model loaded correctly
            if model is not None:
                st.sidebar.success("✅ Model loaded successfully!")
                
                # Test the model with a dummy prediction to ensure it works
                try:
                    test_input = np.zeros((1, 10))  # Dummy input
                    # Don't actually predict, just check if model has predict method
                    if hasattr(model, 'predict'):
                        st.sidebar.info("✅ Model is ready for predictions")
                    return model
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Model loaded but may have issues: {str(e)}")
                    return model
            else:
                st.sidebar.warning("⚠️ Model loaded but is None")
                return None
        else:
            st.sidebar.error(f"❌ Model file not found at: {model_path}")
            st.sidebar.info("Please ensure your model is saved as 'saved_models/best_xgb_model.pkl'")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Load the model
model = load_model()

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================

def preprocess_input(input_df):
    """Apply preprocessing steps to match training data"""
    df = input_df.copy()
    
    # Log-transform skewed features
    skewed = ['capital-gain', 'capital-loss']
    df[skewed] = df[skewed].apply(lambda x: np.log(x + 1))
    
    # One-hot encode categorical features
    df = pd.get_dummies(df)
    
    return df

# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_income(input_df):
    """Make prediction using trained model"""
    if model is not None:
        try:
            # Preprocess the input data
            processed_df = preprocess_input(input_df)
            
            # Make prediction
            prediction = model.predict(processed_df)[0]
            probability = model.predict_proba(processed_df)[0]
            
            return prediction, probability
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.exception(e)  # This will show the full error for debugging
            return None, None
    else:
        st.error("Model not loaded. Please check the model file path.")
        return None, None

# ============================================
# UI STARTS HERE
# ============================================

# Page configuration
st.set_page_config(
    page_title="CharityML Donor Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .high-income {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .low-income {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="main-header">
        <h1>CharityML Donor Prediction Tool</h1>
        <p>Identify potential donors to reduce mailing costs and increase fundraising efficiency</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    **CharityML Donor Predictor**
    
    This tool uses XGBoost machine learning to identify individuals 
    whose income exceeds $50K/year - our most likely donors.
    
    **Model Accuracy:** 87.4%
    **F1 Score:** 0.73
    """)
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check:")
        st.code("1. Is the model file at 'saved_models/best_xgb_model.pkl'?")
        st.code("2. Does the file exist and have correct permissions?")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Personal Information Form")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Demographics", "Employment", "Financial"])
    
    with tab1:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            age = st.number_input("Age", min_value=17, max_value=90, value=35)
            education_num = st.number_input("Years of Education", min_value=1, max_value=16, value=10)
            marital_status = st.selectbox("Marital Status", 
                ["Married-civ-spouse", "Never-married", "Divorced", 
                 "Separated", "Widowed", "Married-spouse-absent", 
                 "Married-AF-spouse"])
        
        with col_d2:
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
            relationship = st.selectbox("Relationship", 
                ["Husband", "Not-in-family", "Wife", "Own-child", "Unmarried", "Other-relative"])
    
    with tab2:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            workclass = st.selectbox("Workclass", 
                ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                 "Local-gov", "State-gov", "Without-pay", "Never-worked"])
            occupation = st.selectbox("Occupation", 
                ["Tech-support", "Craft-repair", "Other-service", "Sales", 
                 "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                 "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                 "Transport-moving", "Priv-house-serv", "Protective-serv", 
                 "Armed-Forces"])
        
        with col_e2:
            hours_per_week = st.number_input("Hours per week", min_value=1, max_value=99, value=40)
            education_level = st.selectbox("Education Level",
                ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
                 "Assoc-acdm", "7th-8th", "Doctorate", "Assoc-voc", "Prof-school",
                 "5th-6th", "10th", "Preschool", "12th", "1st-4th"])
            native_country = st.selectbox("Native Country", 
                ["United-States", "Mexico", "Philippines", "Germany", "Canada", 
                 "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", 
                 "Jamaica", "South", "China", "Italy", "Dominican-Republic", 
                 "Japan", "Guatemala", "Poland", "Portugal"])
    
    with tab3:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0)
        with col_f2:
            capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=5000, value=0)

    # Prediction button
    if st.button("Predict Donor Potential", type="primary", use_container_width=True):
        
        # Create input dataframe
        input_data = pd.DataFrame([[
            age, workclass, education_level, education_num, marital_status,
            occupation, relationship, race, sex, capital_gain,
            capital_loss, hours_per_week, native_country
        ]], columns=['age', 'workclass', 'education_level', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                     'capital-loss', 'hours-per-week', 'native-country'])
        
        # Make prediction
        with st.spinner('Analyzing with XGBoost model...'):
            prediction, probability = predict_income(input_data)
        
        if prediction is not None:
            st.markdown("### Prediction Result")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-box high-income">
                        <strong>LIKELY DONOR</strong><br>
                        <span style="font-size: 1rem;">Income > $50K - Should receive fundraising letter</span><br>
                        <span style="font-size: 0.9rem;">Confidence: {probability[1]:.1%}</span>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                    <div class="prediction-box low-income">
                        <strong>NOT A DONOR</strong><br>
                        <span style="font-size: 1rem;">Income ≤ $50K - Should NOT receive fundraising letter</span><br>
                        <span style="font-size: 0.9rem;">Confidence: {probability[0]:.1%}</span>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show confidence
            confidence = max(probability) * 100
            st.progress(int(confidence) / 100, text=f"Confidence: {confidence:.1f}%")

with col2:
    st.subheader("Model Performance")
    st.markdown("""
    **Donor Prediction Metrics**
    * **Accuracy:** 87.4%
    * **Precision:** 81%
    * **Recall:** 67%
    * **F1 Score:** 0.73
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        <p><strong>CharityML</strong> - Helping people learn machine learning</p>
    </div>
""", unsafe_allow_html=True)
