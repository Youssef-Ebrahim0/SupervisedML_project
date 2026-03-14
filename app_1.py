# app_1.py
# CharityML Donor Prediction App
# Identifies potential donors (income > $50K) to help CharityML target fundraising efforts

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost and add compatibility patch
mport xgboost as xgb

# Comprehensive compatibility patch for XGBoost models
# This handles multiple version compatibility issues

# Patch 1: Handle use_label_encoder
if not hasattr(xgb.XGBClassifier, "use_label_encoder"):
    xgb.XGBClassifier.use_label_encoder = False

# Patch 2: Handle GPU-related attributes
import functools

def _patch_xgboost_model(model):
    """Apply patches to an XGBoost model to handle version compatibility"""
    if hasattr(model, '_Booster'):
        # Handle gpu_id attribute if missing
        if not hasattr(model, 'gpu_id'):
            model.gpu_id = -1
        
        # Handle other common missing attributes
        if not hasattr(model, 'n_gpus'):
            model.n_gpus = 0
        if not hasattr(model, 'predictor'):
            model.predictor = 'cpu_predictor'
    
    return model

# Monkey patch the load method to apply fixes
original_load = joblib.load

def patched_load(*args, **kwargs):
    model = original_load(*args, **kwargs)
    # Check if it's an XGBoost model and apply patches
    if hasattr(model, '_Booster') or 'XGB' in str(type(model)):
        model = _patch_xgboost_model(model)
    return model

joblib.load = patched_load

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

st.sidebar.info("✅ Applied XGBoost compatibility patches")

# Page configuration
st.set_page_config(
    page_title="CharityML Donor Predictor",
    page_icon=":heart:",
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
    .charity-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4ECDC4;
        margin-bottom: 1rem;
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
    .stats-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title with charity focus
st.markdown("""
    <div class="main-header">
        <h1>CharityML Donor Prediction Tool</h1>
        <p style="font-size: 1.2rem;">Help us identify potential donors to reduce mailing costs and increase fundraising efficiency</p>
    </div>
""", unsafe_allow_html=True)

# Charity explanation
st.markdown("""
    <div class="charity-box">
        <h3>Our Mission</h3>
        <p>CharityML is a fictitious charity organization in Silicon Valley that provides financial support 
        for people eager to learn machine learning. After analyzing 32,000 letters, we discovered that 
        <strong>every donation came from someone making more than $50,000 annually</strong>.</p>
        <p>With nearly 15 million working Californians, we need your help to build an algorithm that 
        identifies potential donors, reducing our mailing overhead while maximizing donation yield.</p>
        <p><em>This tool predicts whether an individual's annual income exceeds $50K, helping us target our fundraising efforts more effectively.</em></p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/charity.png", width=100)
    st.header("Impact Calculator")
    
    # Calculate potential savings
    st.subheader("Potential Impact")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Cost per Letter", "$0.65")
    with col_s2:
        st.metric("California Population", "15M")
    
    if 'prediction' in st.session_state:
        st.info(f"""
        **Current Prediction:** 
        {'Potential Donor' if st.session_state.prediction == 1 else 'Not Likely Donor'}
        """)
    
    st.header("About")
    st.info("""
    **CharityML Donor Predictor**
    
    This tool uses machine learning to identify individuals 
    whose income exceeds $50K/year - our most likely donors.
    
    **Model:** XGBoost Classifier
    **Accuracy:** 87.4%
    **F1 Score:** 0.73
    
    By targeting only likely donors, we can:
    * Reduce mailing costs by 75%
    * Increase donation yield
    * Focus resources effectively
    """)
    
    st.header("Feature Importance")
    st.markdown("""
    Top factors that indicate a potential donor:
    1. **Capital Gains** - Investment income
    2. **Married** - Married-civ-spouse
    3. **Age** - Older individuals
    4. **Education** - Higher education
    5. **Relationship** - Husband/Wife
    """)

# Load model function
@st.cache_resource
def load_model():
    """Load the saved model from saved_models directory"""
    try:
        model_path = 'best_xgb_model.pkl'
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            st.sidebar.success(" Model loaded successfully!")
            return model
        else:
            st.sidebar.warning(" Model file not found. Using demo mode.")
            return None
    except Exception as e:
        st.sidebar.error(f" Error loading model: {str(e)}")
        return None

# Load model
model = load_model()

# Main content - Two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Potential Donor Information Form")
    st.markdown("*Fill out the form below to determine if this person is likely to be a donor (income > $50K)*")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Demographics", "Employment", "Financial"])
    
    with tab1:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            age = st.number_input("Age", min_value=17, max_value=90, value=35, 
                                 help="Age of the individual (17-90 years)")
            
            education_num = st.number_input("Years of Education", min_value=1, max_value=16, value=10,
                                          help="Number of years of education completed (1-16)")
            
            marital_status = st.selectbox("Marital Status", 
                ["Married-civ-spouse", "Never-married", "Divorced", 
                 "Separated", "Widowed", "Married-spouse-absent", 
                 "Married-AF-spouse"],
                help="Current marital status - Married individuals are more likely to be donors")
        
        with col_d2:
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            
            race = st.selectbox("Race", 
                ["White", "Black", "Asian-Pac-Islander", 
                 "Amer-Indian-Eskimo", "Other"])
            
            relationship = st.selectbox("Relationship", 
                ["Husband", "Not-in-family", "Wife", "Own-child", 
                 "Unmarried", "Other-relative"],
                help="Relationship status - Husband/Wife are strong donor indicators")
    
    with tab2:
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            workclass = st.selectbox("Workclass", 
                ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                 "Local-gov", "State-gov", "Without-pay", "Never-worked"],
                help="Type of employer/employment")
            
            occupation = st.selectbox("Occupation", 
                ["Tech-support", "Craft-repair", "Other-service", "Sales", 
                 "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                 "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                 "Transport-moving", "Priv-house-serv", "Protective-serv", 
                 "Armed-Forces"],
                help="Current occupation - Executive/Professional roles indicate higher income")
        
        with col_e2:
            hours_per_week = st.number_input("Hours per week", min_value=1, max_value=99, value=40,
                                           help="Average hours worked per week - More hours often mean higher income")
            
            education_level = st.selectbox("Education Level",
                ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
                 "Assoc-acdm", "7th-8th", "Doctorate", "Assoc-voc", "Prof-school",
                 "5th-6th", "10th", "Preschool", "12th", "1st-4th"],
                help="Highest education level achieved - Higher education = higher income potential")
            
            native_country = st.selectbox("Native Country", 
                ["United-States", "Mexico", "Philippines", "Germany", "Canada", 
                 "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", 
                 "Jamaica", "South", "China", "Italy", "Dominican-Republic", 
                 "Japan", "Guatemala", "Poland", "Portugal", "Columbia", 
                 "Haiti", "Iran", "Peru", "France", "Ecuador", "Ireland", 
                 "Thailand", "Nicaragua", "Peru", "Vietnam", "Trinadad&Tobago", 
                 "Hong", "Honduras", "Hungary", "Greece", "Yugoslavia", 
                 "Laos", "Scotland", "Outlying-US(Guam-USVI-etc)", "Cambodia", 
                 "Taiwan", "Holand-Netherlands"],
                help="Country of origin")
    
    with tab3:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            capital_gain = st.number_input("Capital Gain ($)", min_value=0, max_value=100000, value=0,
                                         help="Capital gains from investments - Strongest predictor of high income")
        
        with col_f2:
            capital_loss = st.number_input("Capital Loss ($)", min_value=0, max_value=5000, value=0,
                                         help="Capital losses from investments")

# Preprocessing function
def preprocess_input(input_df):
    """Apply preprocessing steps to match training data"""
    df = input_df.copy()
    
    # Log-transform skewed features
    skewed = ['capital-gain', 'capital-loss']
    df[skewed] = df[skewed].apply(lambda x: np.log(x + 1))
    
    # One-hot encode categorical features
    df = pd.get_dummies(df)
    
    return df

# Mock prediction for demo
def mock_prediction(age, hours, capital_gain, education_num, marital_status):
    """Simple rule-based prediction for demo mode"""
    score = 0
    reasons = []
    
    if age > 40:
        score += 20
        reasons.append("Age > 40")
    if hours > 40:
        score += 20
        reasons.append("Works > 40 hours/week")
    if capital_gain > 1000:
        score += 30
        reasons.append("Has capital gains")
    if education_num > 12:
        score += 20
        reasons.append("Higher education")
    if marital_status == "Married-civ-spouse":
        score += 10
        reasons.append("Married")
    
    probability = score / 100
    if probability > 0.5:
        return 1, [1-probability, probability], reasons
    else:
        return 0, [probability, 1-probability], reasons

# Prediction function
def predict_income(input_df):
    """Make prediction using trained model"""
    if model is not None:
        try:
            processed_df = preprocess_input(input_df)
            prediction = model.predict(processed_df)[0]
            probability = model.predict_proba(processed_df)[0]
            return prediction, probability, []
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None, []
    else:
        # Demo mode
        return mock_prediction(age, hours_per_week, capital_gain, 
                              education_num, marital_status)

# Prediction button
with col1:
    if st.button("Identify Donor Potential", type="primary", use_container_width=True):
        
        # Create input dataframe
        input_data = pd.DataFrame([[
            age, workclass, education_level, education_num, marital_status,
            occupation, relationship, race, sex, capital_gain,
            capital_loss, hours_per_week, native_country
        ]], columns=['age', 'workclass', 'education_level', 'education-num', 'marital-status',
                     'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                     'capital-loss', 'hours-per-week', 'native-country'])
        
        # Make prediction
        with st.spinner('Analyzing donor potential...'):
            prediction, probability, reasons = predict_income(input_data)
            st.session_state.prediction = prediction
        
        if prediction is not None:
            # Display prediction with charity context
            st.markdown("### Donor Potential Analysis")
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-box high-income">
                        <strong>LIKELY DONOR</strong><br>
                        <span style="font-size: 1rem;">Income > $50K - Should receive fundraising letter</span><br>
                        <span style="font-size: 0.9rem;">Confidence: {probability[1]:.1%}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # Calculate potential savings
                st.markdown("""
                <div class="stats-box">
                    <h4>Impact Analysis</h4>
                    <p> This person should receive a fundraising letter</p>
                    <p> By targeting only likely donors, we save $0.65 per avoided letter</p>
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
                
                # Calculate potential savings
                st.markdown("""
                <div class="stats-box">
                    <h4>Savings Analysis</h4>
                    <p> By skipping this person, we save <strong>$0.65</strong> in mailing costs</p>
                    <p> Our targeting algorithm prevents wasting resources on non-donors</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show confidence level
            confidence = max(probability) * 100
            st.progress(int(confidence) / 100, text=f"Prediction Confidence: {confidence:.1f}%")
            
            # Show contributing factors if in demo mode
            if reasons and model is None:
                st.info(f"**Key Factors:** {', '.join(reasons)}")

with col2:
    st.subheader("Campaign Impact Calculator")
    
    # Interactive impact calculator
    st.markdown("### Calculate Your Savings")
    num_letters = st.number_input("Number of letters to send", min_value=1000, max_value=1000000, value=10000, step=1000)
    
    cost_per_letter = 0.65
    traditional_cost = num_letters * cost_per_letter
    
    # Assume our model identifies 25% as potential donors
    targeted_letters = int(num_letters * 0.25)
    targeted_cost = targeted_letters * cost_per_letter
    savings = traditional_cost - targeted_cost
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Traditional Cost", f"${traditional_cost:,.0f}")
    with col_m2:
        st.metric("Targeted Cost", f"${targeted_cost:,.0f}", delta=f"-${savings:,.0f}")
    with col_m3:
        st.metric("Letters Saved", f"{num_letters - targeted_letters:,}")
    
    st.markdown(f"### You save **${savings:,.0f}** by targeting only likely donors!")
    
    # Model performance
    st.markdown("### Model Performance")
    st.markdown("""
    <div class="feature-importance">
        <h4>Donor Prediction Metrics</h4>
        <ul>
            <li><strong>Accuracy:</strong> 87.4% - Correctly identifies donors/non-donors</li>
            <li><strong>Precision:</strong> 81% - Of those predicted as donors, 81% actually are</li>
            <li><strong>Recall:</strong> 67% - Finds 67% of all actual donors</li>
            <li><strong>F1 Score:</strong> 0.73 - Balance between precision and recall</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance
    st.markdown("""
    <div class="feature-importance">
        <h4>Top Donor Indicators</h4>
        <div style="margin: 10px 0;">
            <div>Capital Gains <span style="float: right;">14.0%</span></div>
            <progress value="14" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        <div style="margin: 10px 0;">
            <div>Married <span style="float: right;">11.6%</span></div>
            <progress value="11.6" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        <div style="margin: 10px 0;">
            <div>Age <span style="float: right;">10.2%</span></div>
            <progress value="10.2" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        <div style="margin: 10px 0;">
            <div>Education Years <span style="float: right;">9.2%</span></div>
            <progress value="9.2" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
        <div style="margin: 10px 0;">
            <div>Relationship (Husband) <span style="float: right;">7.5%</span></div>
            <progress value="7.5" max="100" style="width: 100%; height: 10px;"></progress>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer with charity message
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        <p><strong>CharityML</strong> - Helping people learn machine learning since 2024</p>
        <p>By using this tool, we can reduce mailing costs by 75% while maintaining donation yield.</p>
        <p>Every dollar saved goes directly to funding machine learning education!</p>
    </div>
""", unsafe_allow_html=True)

# Debug section (hidden by default)
with st.sidebar.expander("Technical Debug"):
    st.write("Model loaded:", model is not None)
    st.write("Model path:", 'best_xgb_model.pkl')
    if model is not None:
        st.write("Model type:", type(model).__name__)
