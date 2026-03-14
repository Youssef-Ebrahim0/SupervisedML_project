# CharityML Donor Prediction App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CharityML Donor Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .metric-card .value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 25px;
        width: 100%;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError {
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        animation: slideIn 0.5s;
    }
    
    @keyframes slideIn {
        from {
            transform: translateY(-20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Form section styling */
    .form-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .form-section h3 {
        color: #333;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

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

def predict_income(input_df):
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        return prediction[0], probability[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ============================================
# HEADER SECTION
# ============================================
st.markdown("""
<div class="main-header">
    <h1>💰 CharityML Donor Predictor</h1>
    <p>Identify potential donors using machine learning</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# MAIN CONTENT
# ============================================
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("""
    <div class="form-section">
        <h3>📋 Donor Profile Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for form inputs
    form_col1, form_col2 = st.columns(2)
    
    with form_col1:
        age = st.slider("Age", 18, 100, 35, help="Select the person's age")
        
        marital_status = st.selectbox(
            "Marital Status",
            [
                "Married-civ-spouse",
                "Divorced",
                "Never-married",
                "Separated",
                "Widowed",
                "Married-spouse-absent",
                "Married-AF-spouse"
            ],
            help="Current marital status"
        )
        
        relationship = st.selectbox(
            "Relationship",
            [
                "Wife",
                "Own-child",
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Unmarried"
            ],
            help="Relationship status in family"
        )
        
        race = st.selectbox(
            "Race",
            [
                "White",
                "Black",
                "Asian-Pac-Islander",
                "Amer-Indian-Eskimo",
                "Other"
            ],
            help="Racial background"
        )
        
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        
        education_num = st.number_input(
            "Education Years", 
            1, 16, 10,
            help="Total years of education completed"
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
            ],
            help="Type of employment"
        )
    
    with form_col2:
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
            ],
            help="Current occupation"
        )
        
        hours_per_week = st.slider(
            "Hours per week", 
            1, 99, 40,
            help="Average hours worked per week"
        )
        
        education_level = st.selectbox(
            "Education Level",
            [
                "Bachelors",
                "HS-grad",
                "11th",
                "Masters",
                "9th",
                "Some-college"
            ],
            help="Highest education level achieved"
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
            ],
            help="Country of origin"
        )
        
        capital_gain = st.number_input(
            "Capital Gain", 
            0, 100000, 0,
            help="Income from investments"
        )
        
        capital_loss = st.number_input(
            "Capital Loss", 
            0, 5000, 0,
            help="Losses from investments"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Predict Donor Potential", use_container_width=True):
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
                st.success(f"""
                🎉 **Likely Donor**  
                Confidence: {probability[1]*100:.1f}%
                """)
                st.balloons()
                st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3", format="audio/mp3", start_time=0)
            else:
                st.error(f"""
                ❌ **Not a Donor**  
                Confidence: {probability[0]*100:.1f}%
                """)
                
            # Display probability gauge
            prob_value = probability[1] if prediction == 1 else probability[0]
            st.progress(prob_value)

with col2:
    st.markdown("""
    <div class="form-section">
        <h3>📊 Model Performance Metrics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create metric cards
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Accuracy</h3>
            <p class="value">87.4%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>Precision</h3>
            <p class="value">81%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Recall</h3>
            <p class="value">67%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h3>F1 Score</h3>
            <p class="value">0.73</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add interpretation guide
    st.markdown("""
    <div class="form-section" style="margin-top: 2rem;">
        <h3>📈 How to Interpret</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li>✅ <strong>Likely Donor</strong> - Income >50K</li>
            <li>❌ <strong>Not a Donor</strong> - Income ≤50K</li>
        </ul>
        <p style="font-size: 0.9rem; color: #666; margin-top: 1rem;">
            The model predicts whether an individual's income exceeds $50K/year, 
            indicating potential for charitable donations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add feature importance note
    st.markdown("""
    <div class="form-section" style="margin-top: 1rem;">
        <h3>🔑 Key Features</h3>
        <p>Most influential factors:</p>
        <ul>
            <li>Education level</li>
            <li>Occupation</li>
            <li>Hours per week</li>
            <li>Capital gains</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | CharityML Donor Prediction App</p>
    <p style="font-size: 0.8rem;">© 2024 All rights reserved</p>
</div>
""", unsafe_allow_html=True)
