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
    """Preprocess the input data to match training format"""
    try:
        # If we have preprocessors from training, use them
        if preprocessors is not None:
            if 'encoder' in preprocessors:
                # Get categorical columns
                categorical_cols = ['workclass', 'education_level', 'marital-status', 
                                  'occupation', 'relationship', 'race', 'sex', 'native-country']
                
                # Apply encoding
                for col in categorical_cols:
                    if col in input_df.columns:
                        # Handle unknown categories
                        input_df[col] = input_df[col].astype('category')
                        
            if 'scaler' in preprocessors:
                # Scale numerical features if scaler exists
                numerical_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
                input_df[numerical_cols] = preprocessors['scaler'].transform(input_df[numerical_cols])
        
        # Alternative: convert categorical columns to category dtype
        categorical_cols = ['workclass', 'education_level', 'marital-status', 
                          'occupation', 'relationship', 'race', 'sex', 'native-country']
        
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category')
        
        return input_df
        
    except Exception as e:
        st.warning(f"Preprocessing warning: {e}")
        return input_df

def predict_income(input_df):
    """Make prediction with proper preprocessing"""
    try:
        # Preprocess the input
        processed_df = preprocess_input(input_df.copy())
        
        # Make prediction
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
            st.progress(prob_value)
            
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
        <h3>📊 Model Performance</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create metric cards
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
    <div class="form-section" style="margin-top: 1rem;">
        <h3>📈 How to Interpret</h3>
        <div style="background: white; padding: 1rem; border-radius: 8px;">
            <p><span style="color: #28a745;">✓</span> <strong>Likely Donor</strong> - Income >$50K</p>
            <p><span style="color: #dc3545;">✗</span> <strong>Not a Donor</strong> - Income ≤$50K</p>
            <p style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">
                Higher confidence scores indicate more reliable predictions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add feature importance note
    st.markdown("""
    <div class="form-section" style="margin-top: 1rem;">
        <h3>🔑 Key Factors</h3>
        <div style="background: white; padding: 1rem; border-radius: 8px;">
            <div class="feature-item">
                <span class="feature-name">Education Level</span>
                <span class="feature-value">High Impact</span>
            </div>
            <div class="feature-item">
                <span class="feature-name">Occupation</span>
                <span class="feature-value">High Impact</span>
            </div>
            <div class="feature-item">
                <span class="feature-name">Hours per Week</span>
                <span class="feature-value">Medium Impact</span>
            </div>
            <div class="feature-item">
                <span class="feature-name">Capital Gains</span>
                <span class="feature-value">Medium Impact</span>
            </div>
            <div class="feature-item">
                <span class="feature-name">Age</span>
                <span class="feature-value">Low Impact</span>
            </div>
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
