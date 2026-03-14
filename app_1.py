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
# XGBOOST MODEL WRAPPER (MOST ROBUST SOLUTION)
# ============================================

class XGBoostCompatibilityWrapper:
    """
    A wrapper class that makes older XGBoost models compatible with newer versions.
    This handles all missing attributes and method calls gracefully.
    """
    
    def __init__(self, model):
        self.model = model
        self.gpu_id = -1
        self.n_gpus = 0
        self.predictor = 'cpu_predictor'
        self.enable_categorical = False
        self.use_rmm = False
        self.max_cat_to_onehot = 4
        self.cat_priority = False
        
    def __getattr__(self, name):
        """Forward any attribute access to the underlying model"""
        return getattr(self.model, name)
    
    def get_params(self, deep=True):
        """Override get_params to include our added attributes"""
        params = {}
        if hasattr(self.model, 'get_params'):
            params = self.model.get_params(deep)
        
        # Add our attributes
        params['gpu_id'] = self.gpu_id
        params['n_gpus'] = self.n_gpus
        params['predictor'] = self.predictor
        params['enable_categorical'] = self.enable_categorical
        params['use_rmm'] = self.use_rmm
        params['max_cat_to_onehot'] = self.max_cat_to_onehot
        params['cat_priority'] = self.cat_priority
        
        return params
    
    def get_xgb_params(self):
        """Override get_xgb_params to include our attributes"""
        params = {}
        if hasattr(self.model, 'get_xgb_params'):
            params = self.model.get_xgb_params()
        
        # Add our attributes
        params['gpu_id'] = self.gpu_id
        params['n_gpus'] = self.n_gpus
        params['predictor'] = self.predictor
        params['enable_categorical'] = self.enable_categorical
        params['use_rmm'] = self.use_rmm
        params['max_cat_to_onehot'] = self.max_cat_to_onehot
        params['cat_priority'] = self.cat_priority
        
        return params
    
    def _can_use_inplace_predict(self):
        """Override to always return False to avoid GPU-related issues"""
        return False
    
    def predict(self, X, **kwargs):
        """Wrapper for predict method"""
        return self.model.predict(X, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        """Wrapper for predict_proba method"""
        return self.model.predict_proba(X, **kwargs)

# ============================================
# MODEL LOADING FUNCTION
# ============================================

@st.cache_resource
def load_model():
    """Load the saved model and wrap it for compatibility"""
    try:
        model_path = 'best_xgb_model.pkl'
        
        if os.path.exists(model_path):
            st.sidebar.info(f"Loading model from: {model_path}")
            
            # Load the raw model
            raw_model = joblib.load(model_path)
            
            # Wrap it with our compatibility wrapper
            wrapped_model = XGBoostCompatibilityWrapper(raw_model)
            
            st.sidebar.success("✅ Model loaded and wrapped for compatibility!")
            return wrapped_model
        else:
            st.sidebar.error(f"❌ Model file not found at: {model_path}")
            st.sidebar.info("Please ensure your model is saved as 'best_xgb_model.pkl'")
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
            
            st.write("Debug - Input shape:", processed_df.shape)
            
            # Make prediction
            prediction = model.predict(processed_df)[0]
            probability = model.predict_proba(processed_df)[0]
            
            return prediction, probability
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.exception(e)
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
