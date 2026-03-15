import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Income Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the UI
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
    
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Form section styling */
    .form-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    .form-section h3 {
        color: #333;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Info box styling */
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError {
        padding: 1rem;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Selectbox styling */
    .stSelectbox label, .stNumberInput label {
        font-weight: 500;
        color: #333;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💰 Income Prediction App</h1>
    <p>Predict whether an individual earns more than $50K using machine learning</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - MODEL INFO
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/money--v1.png", width=100)
    st.markdown("## 🎯 About the Model")
    
    st.markdown("""
    <div class="card">
        <h4>Model: XGBoost Classifier</h4>
        <p>This model predicts whether an individual earns more than $50,000 per year based on demographic and employment data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "87.4%", "+2.3%")
        st.metric("Precision", "80.7%", "")
    with col2:
        st.metric("Recall", "66.5%", "")
        st.metric("F1 Score", "0.73", "")
    
    st.markdown("### 🔑 Top Features")
    st.info("""
    1. Capital Gain
    2. Marital Status
    3. Age
    4. Education Level
    5. Hours per Week
    """)
    
    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    st.markdown("""
    - **Source:** UCI Adult Dataset
    - **Records:** 45,222
    - **Features:** 14
    - **Target:** Income >$50K
    """)

# ============================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================
@st.cache_resource
def load_model():
    """Load the saved XGBoost model"""
    try:
        model = pickle.load(open("best_xgboost_model.pkl", "rb"))
        return model
    except:
        # Create a default model if file not found
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=1,
            random_state=42
        )
        return model

@st.cache_data
def load_feature_names():
    """Load feature names used during training"""
    try:
        feature_names = pickle.load(open("feature_names.pkl", "rb"))
        return feature_names
    except:
        return None

@st.cache_resource
def load_scaler():
    """Load the scaler used during training"""
    try:
        scaler = pickle.load(open("scaler.pkl", "rb"))
        return scaler
    except:
        return None

# Load model and preprocessing objects
model = load_model()
feature_names = load_feature_names()
scaler = load_scaler()

# ============================================
# PREPROCESSING FUNCTIONS
# ============================================
def log_transform_skewed(df):
    """Apply log transformation to capital gain and loss"""
    skewed = ['capital-gain', 'capital-loss']
    df[skewed] = df[skewed].apply(lambda x: np.log(x + 1))
    return df

def one_hot_encode(df):
    """Apply one-hot encoding to categorical features"""
    return pd.get_dummies(df)

def align_features(df, target_features):
    """Align dataframe columns with training features"""
    if target_features is None:
        return df
    
    # Create empty dataframe with all required columns
    result = pd.DataFrame(0, index=[0], columns=target_features)
    
    # Fill in values that exist in input
    for col in df.columns:
        if col in result.columns:
            result[col] = df[col].values
    return result

def scale_features(df, scaler, num_cols):
    """Scale numerical features"""
    if scaler is not None:
        df[num_cols] = scaler.transform(df[num_cols])
    else:
        # Approximate standardization if scaler not available
        mean_dict = {
            'age': 38.5, 'education-num': 10.1,
            'capital-gain': 0.5, 'capital-loss': 0.2, 'hours-per-week': 40.9
        }
        std_dict = {
            'age': 13.2, 'education-num': 2.55,
            'capital-gain': 1.5, 'capital-loss': 1.0, 'hours-per-week': 12.0
        }
        for col in num_cols:
            df[col] = (df[col] - mean_dict[col]) / std_dict[col]
    return df

# ============================================
# MAIN CONTENT
# ============================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="form-section">
        <h3>👤 Enter Individual Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input form with columns
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        age = st.number_input("Age", min_value=17, max_value=90, value=35, step=1)
        
        workclass = st.selectbox(
            "Work Class",
            ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
             "Local-gov", "State-gov", "Without-pay", "Never-worked"]
        )
        
        education = st.selectbox(
            "Education Level",
            ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th",
             "11th", "12th", "HS-grad", "Some-college", "Assoc-voc",
             "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"]
        )
        
        education_num = st.number_input("Education Years", min_value=1, max_value=16, value=10, step=1)
    
    with input_col2:
        marital_status = st.selectbox(
            "Marital Status",
            ["Never-married", "Married-civ-spouse", "Divorced",
             "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"]
        )
        
        occupation = st.selectbox(
            "Occupation",
            ["Tech-support", "Craft-repair", "Other-service", "Sales",
             "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
             "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
             "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
        )
        
        relationship = st.selectbox(
            "Relationship",
            ["Husband", "Not-in-family", "Wife", "Own-child", "Unmarried", "Other-relative"]
        )
        
        race = st.selectbox(
            "Race",
            ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
        )
    
    with input_col3:
        sex = st.selectbox("Sex", ["Male", "Female"])
        
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, step=100)
        hours_per_week = st.slider("Hours per Week", min_value=1, max_value=99, value=40)
        
        native_country = st.selectbox(
            "Native Country",
            ["United-States", "Mexico", "Philippines", "Germany", "Canada",
             "Puerto-Rico", "El-Salvador", "India", "Cuba", "England", "Jamaica",
             "South", "China", "Italy", "Dominican-Republic", "Japan", "Guatemala",
             "Poland", "Columbia", "Taiwan", "Vietnam", "Haiti", "Portugal",
             "Iran", "Nicaragua", "Peru", "France", "Ireland", "Ecuador", "Thailand",
             "Cambodia", "Hong", "Greece", "Trinadad&Tobago", "Laos", "Outlying-US(Guam-USVI-etc)",
             "Yugoslavia", "Hungary", "Scotland", "Honduras", "Holand-Netherlands"]
        )
    
    # Prediction button
    if st.button("🔮 Predict Income Level", use_container_width=True):
        with st.spinner("Analyzing data... Making prediction..."):
            try:
                # Create input dataframe with leading spaces (as in notebook)
                input_data = pd.DataFrame({
                    'age': [age],
                    'workclass': [f" {workclass}"],
                    'education_level': [f" {education}"],
                    'education-num': [education_num],
                    'marital-status': [f" {marital_status}"],
                    'occupation': [f" {occupation}"],
                    'relationship': [f" {relationship}"],
                    'race': [f" {race}"],
                    'sex': [f" {sex}"],
                    'capital-gain': [capital_gain],
                    'capital-loss': [capital_loss],
                    'hours-per-week': [hours_per_week],
                    'native-country': [f" {native_country}"]
                })
                
                # Apply preprocessing steps
                # Step 1: Log transform skewed features
                input_data = log_transform_skewed(input_data)
                
                # Step 2: One-hot encode
                input_encoded = one_hot_encode(input_data)
                
                # Define numerical columns for scaling
                num_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
                
                # Step 3: Align features with training data
                if feature_names is not None:
                    input_final = align_features(input_encoded, feature_names)
                else:
                    input_final = input_encoded
                
                # Step 4: Scale numerical features
                input_final = scale_features(input_final, scaler, num_cols)
                
                # Step 5: Make prediction
                if model is not None:
                    prediction = model.predict(input_final)[0]
                    probability = model.predict_proba(input_final)[0]
                else:
                    # Fallback to rule-based prediction if model not available
                    income_score = 0
                    if capital_gain > 5000:
                        income_score += 3
                    elif capital_gain > 1000:
                        income_score += 1
                    if education_num >= 13:
                        income_score += 2
                    elif education_num >= 10:
                        income_score += 1
                    if 35 <= age <= 55:
                        income_score += 1
                    if marital_status == 'Married-civ-spouse':
                        income_score += 2
                    if hours_per_week >= 45:
                        income_score += 1
                    if occupation in ['Exec-managerial', 'Prof-specialty', 'Sales']:
                        income_score += 1
                    
                    prediction = 1 if income_score >= 5 else 0
                    probability = [0.9, 0.1] if prediction == 0 else [0.1, 0.9]
                
                # Display results
                st.markdown("---")
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown("""
                    <div class="info-box">
                        <h4>📋 Input Summary</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    summary_df = pd.DataFrame({
                        'Feature': ['Age', 'Education', 'Hours/Week', 'Capital Gain'],
                        'Value': [age, education, hours_per_week, f"${capital_gain:,}"]
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                with result_col2:
                    if prediction == 1:
                        st.success(f"""
                        ### 🎯 Likely Donor
                        **Income:** >$50K
                        **Confidence:** {probability[1]*100:.1f}%
                        """)
                        st.balloons()
                    else:
                        st.error(f"""
                        ### ❌ Not a Donor
                        **Income:** ≤$50K
                        **Confidence:** {probability[0]*100:.1f}%
                        """)
                
                # Display probability gauge
                st.markdown("### Prediction Confidence")
                confidence = float(max(probability))
                st.progress(confidence)
                
                # Show key factors
                st.markdown("""
                <div class="info-box">
                    <strong>📊 Key Factors in This Prediction:</strong>
                </div>
                """, unsafe_allow_html=True)
                
                factors = []
                if capital_gain > 5000:
                    factors.append("✅ High capital gain")
                if education_num >= 13:
                    factors.append("✅ Advanced education")
                if 35 <= age <= 55:
                    factors.append("✅ Prime earning age")
                if marital_status == 'Married-civ-spouse':
                    factors.append("✅ Married with spouse present")
                if occupation in ['Exec-managerial', 'Prof-specialty', 'Sales']:
                    factors.append("✅ High-income occupation")
                if hours_per_week >= 45:
                    factors.append("✅ Works many hours")
                
                for factor in factors:
                    st.write(factor)
                
                if not factors:
                    st.write("No strong positive indicators")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

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
    <p>Built with ❤️ using Streamlit | Income Prediction App</p>
    <p style="font-size: 0.8rem;">Model trained on UCI Adult Census Income dataset</p>
</div>
""", unsafe_allow_html=True)
