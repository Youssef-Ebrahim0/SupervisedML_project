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
        margin-bottom: 1rem;
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
        margin-top: 1rem;
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
    
    /* Info box styling */
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Feature importance styling */
    .feature-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem;
        background: white;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    
    .feature-name {
        font-weight: 600;
        color: #333;
    }
    
    .feature-value {
        color: #667eea;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.write(f"⚙️ XGBoost version: {xgb.__version__}")

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
# LOAD MODEL AND PREPROCESSORS
# ============================================

@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and any preprocessors"""
    try:
        # Try to load the model
        model = joblib.load("best_xgb_model.pkl")
        model = patch_xgb_model(model)
        
        # Try to load preprocessors if they exist
        try:
            preprocessors = joblib.load("preprocessors.pkl")
            return model, preprocessors
        except:
            return model, None
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, preprocessors = load_model_and_preprocessors()

def preprocess_input(input_df):
    """Preprocess input to match training format"""

    try:

        # Convert categorical variables to dummy variables
        df = pd.get_dummies(input_df)

        # Align columns with model features
        if hasattr(model, "feature_names_in_"):

            required_cols = model.feature_names_in_

            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0

            df = df[required_cols]

        return df

    except Exception as e:
        st.warning(f"Preprocessing warning: {e}")
        return input_df

def predict_income(input_df):
    """Make prediction with proper preprocessing"""

    try:

        processed_df = preprocess_input(input_df.copy())

        prediction = model.predict(processed_df)
        probability = model.predict_proba(processed_df)

        return prediction[0], probability[0]

    except Exception as e:

        st.error(f"Prediction error: {e}")
        st.info("💡 Tip: Make sure all fields are filled correctly")

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
        <p style="color: #666; font-size: 0.9rem;">Fill in the details below to predict donor potential</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for form inputs
    form_col1, form_col2 = st.columns(2)
    
    with form_col1:
        age = st.slider(
            "Age", 
            min_value=18, 
            max_value=100, 
            value=35,
            help="Select the person's age"
        )
        
        marital_status = st.selectbox(
            "Marital Status",
            options=[
                "Married-civ-spouse",
                "Divorced",
                "Never-married",
                "Separated",
                "Widowed",
                "Married-spouse-absent",
                "Married-AF-spouse"
            ],
            index=0,
            help="Current marital status"
        )
        
        relationship = st.selectbox(
            "Relationship",
            options=[
                "Wife",
                "Own-child",
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Unmarried"
            ],
            index=3,
            help="Relationship status in family"
        )
        
        race = st.selectbox(
            "Race",
            options=[
                "White",
                "Black",
                "Asian-Pac-Islander",
                "Amer-Indian-Eskimo",
                "Other"
            ],
            index=0,
            help="Racial background"
        )
        
        sex = st.radio(
            "Sex", 
            options=["Male", "Female"], 
            horizontal=True,
            index=0
        )
        
        education_num = st.number_input(
            "Education Years", 
            min_value=1, 
            max_value=16, 
            value=10,
            help="Total years of education completed"
        )
        
        workclass = st.selectbox(
            "Workclass",
            options=[
                "Private",
                "Self-emp-not-inc",
                "Self-emp-inc",
                "Federal-gov",
                "Local-gov",
                "State-gov",
                "Without-pay",
                "Never-worked"
            ],
            index=0,
            help="Type of employment"
        )
    
    with form_col2:
        occupation = st.selectbox(
            "Occupation",
            options=[
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
            index=4,
            help="Current occupation"
        )
        
        hours_per_week = st.slider(
            "Hours per week", 
            min_value=1, 
            max_value=99, 
            value=40,
            help="Average hours worked per week"
        )
        
        education_level = st.selectbox(
            "Education Level",
            options=[
                "Bachelors",
                "HS-grad",
                "11th",
                "Masters",
                "9th",
                "Some-college",
                "Assoc-acdm",
                "Assoc-voc",
                "7th-8th",
                "Doctorate",
                "Prof-school",
                "5th-6th",
                "10th",
                "1st-4th",
                "Preschool",
                "12th"
            ],
            index=0,
            help="Highest education level achieved"
        )
        
        native_country = st.selectbox(
            "Native Country",
            options=[
                "United-States",
                "Mexico",
                "Philippines",
                "Germany",
                "Canada",
                "India",
                "Puerto-Rico",
                "El-Salvador",
                "Cuba",
                "England",
                "Jamaica",
                "South",
                "China",
                "Italy",
                "Dominican-Republic",
                "Japan",
                "Guatemala",
                "Poland",
                "Vietnam",
                "Haiti"
            ],
            index=0,
            help="Country of origin"
        )
        
        capital_gain = st.number_input(
            "Capital Gain", 
            min_value=0, 
            max_value=100000, 
            value=0,
            step=100,
            help="Income from investments"
        )
        
        capital_loss = st.number_input(
            "Capital Loss", 
            min_value=0, 
            max_value=5000, 
            value=0,
            step=100,
            help="Losses from investments"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Predict Donor Potential", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame([[
            age,
            workclass,
            education_level,
            education_num,
            marital_status,
            occupation,
            relationship,
            race,
            sex.lower(),  # Convert to lowercase to match training data
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
        
        # Show loading spinner
        with st.spinner("Analyzing donor profile..."):
            prediction, probability = predict_income(input_data)
        
        if prediction is not None:
            # Create a nice result display
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                if prediction == 1:
                    st.markdown("""
                    <div style="text-align: center; padding: 1rem;">
                        <span style="font-size: 4rem;">🎉</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 1rem;">
                        <span style="font-size: 4rem;">💔</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with result_col2:
                if prediction == 1:
                    st.success(f"""
                    ### 🎯 Likely Donor
                    **Confidence:** {probability[1]*100:.1f}%
                    """)
                    st.balloons()
                else:
                    st.error(f"""
                    ### ❌ Not a Donor
                    **Confidence:** {probability[0]*100:.1f}%
                    """)
            
            # Display probability gauge
            st.markdown("### Prediction Confidence")
            prob_value = probability[1] if prediction == 1 else probability[0]
            confidence = float(max(probability))
            st.progress(confidence)
            st.write(f"Confidence: {confidence*100:.1f}%")
            
            # Additional insights
            st.markdown("""
            <div class="info-box">
                <strong>📊 Insight:</strong> This prediction is based on income levels. 
                'Likely Donor' indicates income >$50K, 'Not a Donor' indicates income ≤$50K.
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="form-section">
        <h3 style="margin-bottom: 1rem;">📊 Model Performance</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a single row of compact metric cards
    metrics_html = """
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin-bottom: 1rem;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 0.75rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Accuracy</div>
            <div style="color: white; font-size: 1.5rem; font-weight: 700; line-height: 1.2;">87.4%</div>
        </div>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 0.75rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Precision</div>
            <div style="color: white; font-size: 1.5rem; font-weight: 700; line-height: 1.2;">81%</div>
        </div>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 0.75rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Recall</div>
            <div style="color: white; font-size: 1.5rem; font-weight: 700; line-height: 1.2;">67%</div>
        </div>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 0.75rem; 
                    border-radius: 8px; 
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="color: rgba(255,255,255,0.8); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">F1 Score</div>
            <div style="color: white; font-size: 1.5rem; font-weight: 700; line-height: 1.2;">0.73</div>
        </div>
    </div>
    """
    
    st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Add compact interpretation guide
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea; margin: 0.5rem 0;">
        <div style="display: flex; align-items: center; gap: 1rem; font-size: 0.9rem;">
            <div style="display: flex; align-items: center;">
                <span style="color: #28a745; font-size: 1.2rem; margin-right: 0.25rem;">✓</span>
                <span><strong>Donor</strong> >$50K</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span style="color: #dc3545; font-size: 1.2rem; margin-right: 0.25rem;">✗</span>
                <span><strong>Non-Donor</strong> ≤$50K</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add compact feature importance
    st.markdown("""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #667eea;">
        <h4 style="margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #333;">🔑 Key Factors</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
            <span style="background: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 500; color: #667eea;">Education</span>
            <span style="background: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 500; color: #667eea;">Occupation</span>
            <span style="background: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 500; color: #667eea;">Hours/Week</span>
            <span style="background: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 500; color: #667eea;">Capital Gains</span>
            <span style="background: white; padding: 0.25rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 500; color: #667eea;">Age</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | CharityML Donor Prediction App</p>
    <p style="font-size: 0.8rem;">Model trained on UCI Adult Census Income dataset</p>
</div>
""", unsafe_allow_html=True)
