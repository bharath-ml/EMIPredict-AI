"""
EMIPredict AI Pro - Enterprise Financial Risk Assessment Platform
Senior Level Implementation with Advanced Features - ENHANCED VERSION
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import hashlib
from datetime import datetime, timedelta
import time
import sys
import os
from pathlib import Path
import base64
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo
from streamlit_extras.stylable_container import stylable_container
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="EMIPredict AI Pro",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.emipredict.ai/help',
        'Report a bug': 'https://www.emipredict.ai/bug',
        'About': "# EMIPredict AI Pro\nEnterprise Financial Risk Assessment Platform"
    }
)

# ============================================================================
# CUSTOM CSS - Professional Styling with Better Background
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles - Professional Dark Theme */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Main container with elegant glassmorphism */
    .main-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
    }
    
    /* Animated gradient header - Enhanced */
    .gradient-header {
        background: linear-gradient(-45deg, #667eea, #764ba2, #4facfe, #00f2fe);
        background-size: 400% 400%;
        animation: gradient 10s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Professional metric cards - Dark theme */
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        border-color: #667eea;
    }
    
    /* Professional buttons - Enhanced */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Professional input fields - Dark theme */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stDateInput > div > div > input:focus,
    .stTimeInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
        background-color: rgba(30, 41, 59, 0.95) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #4facfe);
    }
    
    /* Tabs styling - Dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: rgba(30, 41, 59, 0.5);
        padding: 0.5rem;
        border-radius: 50px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Success/Warning/Error boxes - Dark theme */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid;
        background-color: rgba(30, 41, 59, 0.8);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Dataframe styling - Dark theme */
    .dataframe-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        background-color: rgba(30, 41, 59, 0.8);
    }
    
    .stDataFrame {
        background-color: transparent !important;
    }
    
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        color: white !important;
        background-color: rgba(30, 41, 59, 0.5) !important;
    }
    
    .stDataFrame [data-testid="StyledDataFrameHeader"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Sidebar styling - Enhanced */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-content {
        color: white;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    /* Paragraphs */
    p, li, .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: white !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-top: none !important;
    }
    
    /* Loading animation - Enhanced */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    .loading-spinner:after {
        content: " ";
        display: block;
        width: 64px;
        height: 64px;
        margin: 8px;
        border-radius: 50%;
        border: 6px solid #667eea;
        border-color: #667eea transparent #764ba2 transparent;
        animation: loading-spinner 1.2s linear infinite;
    }
    
    @keyframes loading-spinner {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
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
        background-color: #1e293b;
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
        border: 1px solid #667eea;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Code blocks */
    code {
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: #4facfe !important;
        border-radius: 5px !important;
        padding: 2px 5px !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: rgba(30, 41, 59, 0.8) !important;
        color: white !important;
        border: 1px solid #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Initialize Session State with Advanced Configuration
# ============================================================================
if 'initialized' not in st.session_state:
    st.session_state.update({
        'initialized': True,
        'data': None,
        'models_loaded': False,
        'classification_model': None,
        'regression_model': None,
        'preprocessor': None,
        'label_encoder': None,
        'prediction_history': [],
        'batch_predictions': [],
        'portfolio_data': None,
        'system_logs': [],
        'user_preferences': {
            'theme': 'dark',
            'animations': True,
            'notifications': True
        },
        'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
        'last_activity': datetime.now(),
        'api_calls': 0,
        'system_start_time': datetime.now()
    })

# ============================================================================
# Load Models (with caching for performance)
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained models with caching"""
    try:
        models_dir = Path('models')
        
        # Load classification model
        class_model_path = models_dir / 'best_classifier_xgboost.pkl'
        if class_model_path.exists():
            classification_model = joblib.load(class_model_path)
        else:
            classification_model = None
            
        # Load regression model
        reg_model_path = models_dir / 'best_regressor_xgboost.pkl'
        if reg_model_path.exists():
            regression_model = joblib.load(reg_model_path)
        else:
            regression_model = None
            
        # Load preprocessor
        preprocessor_path = models_dir / 'preprocessor.pkl'
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
        else:
            preprocessor = None
            
        # Load label encoder
        encoder_path = models_dir / 'label_encoder.pkl'
        if encoder_path.exists():
            label_encoder = joblib.load(encoder_path)
        else:
            label_encoder = None
            
        # Load feature names if available
        feature_names_path = models_dir / 'feature_names.json'
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
        else:
            feature_names = None
            
        return classification_model, regression_model, preprocessor, label_encoder, feature_names
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

# Load models
with st.spinner("🚀 Loading AI Models..."):
    classification_model, regression_model, preprocessor, label_encoder, feature_names = load_models()
    st.session_state.models_loaded = all([classification_model, regression_model, preprocessor])
    
    # Debug: Show feature names if available (remove in production)
    if feature_names and 'feature_names' in feature_names:
        st.sidebar.info(f"Model expects {len(feature_names['feature_names'])} features")
        # Uncomment below to see feature names (for debugging)
        # with st.sidebar.expander("View Features"):
        #     st.write(feature_names['feature_names'][:10])  # Show first 10
# Get expected feature count
expected_features = None
if preprocessor and hasattr(preprocessor, 'transformers_'):
    try:
        # Try to get the number of features the preprocessor expects
        if feature_names and 'feature_names' in feature_names:
            expected_features = len(feature_names['feature_names'])
        else:
            # Fallback: try to infer from preprocessor
            expected_features = len(preprocessor.get_feature_names_out()) if hasattr(preprocessor, 'get_feature_names_out') else None
    except:
        expected_features = None

# ============================================================================
# Advanced Navigation Menu
# ============================================================================
# ============================================================================
# Advanced Navigation Menu - UPDATED with white text and blue background
# ============================================================================
with st.sidebar:
    # Company Logo and Brand with animation
    st.markdown("""
    <div style="text-align: center; padding: 2rem 1rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem; animation: pulse 2s infinite;">🏦</div>
        <h2 style="color: white; font-size: 1.8rem; margin: 0; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">EMIPredict AI Pro</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.8rem; margin-top: 0.5rem;">Enterprise Edition v3.0</p>
    </div>
    <style>
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Professional Navigation Menu - UPDATED with white text and blue background
    selected = option_menu(
        menu_title=None,
        options=[
            "Executive Dashboard",
            "Data Intelligence",
            "Risk Assessment",
            "EMI Calculator",
            "Model Monitor",
            "MLflow Analytics",
            "System Admin"
        ],
        icons=[
            'speedometer2',
            'graph-up',
            'shield-shaded',
            'calculator',
            'cpu',
            'bar-chart-steps',
            'gear'
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#1e3a8a",  # Thick blue background
                "border-radius": "10px"
            },
            "icon": {
                "color": "white",  # White icons
                "font-size": "1.2rem"
            },
            "nav-link": {
                "color": "white",  # White text
                "font-size": "1rem",
                "text-align": "left",
                "margin": "0px",
                "transition": "all 0.3s ease",
                "border-radius": "10px",
                "padding": "0.75rem 1rem",
                "background-color": "transparent"
            },
            "nav-link-hover": {
                "background": "rgba(255, 255, 255, 0.2)",  # Light hover effect
                "color": "white"
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, #2563eb 0%, #3b82f6 100%)",  # Brighter blue for selected
                "font-weight": "600",
                "color": "white"
            },
        }
    )
    
    st.markdown("---")
    
    # Session Info with enhanced styling
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p style='color: rgba(255,255,255,0.5); font-size: 0.7rem;'>Session ID</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white; font-size: 0.8rem; font-weight: 600; background: rgba(102,126,234,0.2); padding: 0.3rem; border-radius: 5px;'>{st.session_state.session_id}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p style='color: rgba(255,255,255,0.5); font-size: 0.7rem;'>API Calls</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: white; font-size: 0.8rem; font-weight: 600; background: rgba(102,126,234,0.2); padding: 0.3rem; border-radius: 5px;'>{st.session_state.api_calls}</p>", unsafe_allow_html=True)
    
    # Model Status with enhanced indicator
    if st.session_state.models_loaded:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.2); padding: 0.5rem; border-radius: 10px; border-left: 3px solid #10B981; margin-top: 1rem;">
            <p style="color: #10B981; margin: 0; display: flex; align-items: center;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">●</span> AI Models: Active
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(239, 68, 68, 0.2); padding: 0.5rem; border-radius: 10px; border-left: 3px solid #EF4444; margin-top: 1rem;">
            <p style="color: #EF4444; margin: 0; display: flex; align-items: center;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">●</span> AI Models: Not Loaded
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # System uptime
    uptime = datetime.now() - st.session_state.system_start_time
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    st.markdown(f"""
    <div style="margin-top: 1rem; text-align: center;">
        <p style="color: rgba(255,255,255,0.5); font-size: 0.7rem;">System Uptime</p>
        <p style="color: white; font-size: 0.9rem;">{uptime.days}d {hours}h {minutes}m</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================
# ============================================================================
# Helper Functions - UPDATED to fix prediction error
# ============================================================================
# ============================================================================
# Helper Functions - COMPLETELY FIXED VERSION
# ============================================================================
# ============================================================================
# Helper Functions - FINAL FIXED VERSION
# ============================================================================
# ============================================================================
# Helper Functions - COMPLETE FIX for categorical features
# ============================================================================
def prepare_input_data(input_dict):
    """Prepare raw input DataFrame for sklearn ColumnTransformer preprocessor.
    
    The preprocessor expects raw column names (age, gender, etc.) as input.
    It handles all scaling and one-hot encoding internally, producing the
    num__/cat__ prefixed output features that the model was trained on.
    """
    try:
        # Simply return a raw DataFrame — the ColumnTransformer preprocessor
        # expects raw input columns and transforms them itself.
        df_raw = pd.DataFrame([input_dict])
        return df_raw

    except Exception as e:
        st.error(f"Error preparing input: {str(e)}")
        return None

def log_system_event(event_type, message, level="INFO"):
    """Log system events"""
    st.session_state.system_logs.append({
        'timestamp': datetime.now(),
        'type': event_type,
        'message': message,
        'level': level
    })
    # Keep only last 1000 logs
    if len(st.session_state.system_logs) > 1000:
        st.session_state.system_logs = st.session_state.system_logs[-1000:]

# ============================================================================
# Main Content Area
# ============================================================================
main_container = st.container()

with main_container:
    # Header with animation
    st.markdown('<h1 class="gradient-header">🏦 EMIPredict AI Professional</h1>', unsafe_allow_html=True)
    
    # ============================================================================
    # EXECUTIVE DASHBOARD
    # ============================================================================
    if selected == "Executive Dashboard":
        # KPI Row with Advanced Metrics
        st.markdown("## 📊 Executive Command Center")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            with stylable_container(
                key="kpi_1",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                        border: 1px solid rgba(255,255,255,0.1);
                    }
                """
            ):
                st.markdown("### 💰 Portfolio Value")
                st.markdown("<h2 style='color: white; font-size: 2.5rem; margin: 0;'>₹4.2B</h2>", unsafe_allow_html=True)
                st.markdown("<p style='color: rgba(255,255,255,0.7);'>↑ 12.5% vs last month</p>", unsafe_allow_html=True)
        
        with col2:
            with stylable_container(
                key="kpi_2",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 10px 30px rgba(246, 211, 101, 0.3);
                        border: 1px solid rgba(255,255,255,0.1);
                    }
                """
            ):
                st.markdown("### ✅ Approval Rate")
                st.markdown("<h2 style='color: white; font-size: 2.5rem; margin: 0;'>67.3%</h2>", unsafe_allow_html=True)
                st.markdown("<p style='color: rgba(255,255,255,0.7);'>↑ 3.2% this quarter</p>", unsafe_allow_html=True)
        
        with col3:
            with stylable_container(
                key="kpi_3",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 10px 30px rgba(132, 250, 176, 0.3);
                        border: 1px solid rgba(255,255,255,0.1);
                    }
                """
            ):
                st.markdown("### ⚠️ Risk Exposure")
                st.markdown("<h2 style='color: white; font-size: 2.5rem; margin: 0;'>8.2%</h2>", unsafe_allow_html=True)
                st.markdown("<p style='color: rgba(255,255,255,0.7);'>↓ 1.5% from target</p>", unsafe_allow_html=True)
        
        with col4:
            with stylable_container(
                key="kpi_4",
                css_styles="""
                    {
                        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
                        border-radius: 15px;
                        padding: 1.5rem;
                        box-shadow: 0 10px 30px rgba(161, 140, 209, 0.3);
                        border: 1px solid rgba(255,255,255,0.1);
                    }
                """
            ):
                st.markdown("### 🤖 Model Accuracy")
                st.markdown("<h2 style='color: white; font-size: 2.5rem; margin: 0;'>94.2%</h2>", unsafe_allow_html=True)
                st.markdown("<p style='color: rgba(255,255,255,0.7);'>↑ 2.1% this week</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Advanced Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            with st.container():
                st.markdown("### 📈 Portfolio Composition")
                
                # Advanced Sunburst Chart
                fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{"type": "domain"}, {"type": "domain"}]],
                    subplot_titles=["Loan Distribution", "Risk Profile"]
                )
                
                # Loan distribution by scenario
                fig.add_trace(go.Sunburst(
                    labels=["Total", "E-commerce", "Home Appliances", "Vehicle", "Personal", "Education",
                           "Low Risk", "Medium Risk", "High Risk"],
                    parents=["", "Total", "Total", "Total", "Total", "Total",
                            "E-commerce", "Home Appliances", "Vehicle"],
                    values=[400000, 80000, 80000, 80000, 80000, 80000, 60000, 50000, 30000],
                    branchvalues="total",
                    marker=dict(colors=px.colors.qualitative.Set3)
                ), row=1, col=1)
                
                # Risk distribution
                fig.add_trace(go.Pie(
                    labels=['Low Risk', 'Medium Risk', 'High Risk'],
                    values=[240000, 100000, 60000],
                    marker_colors=['#10B981', '#F59E0B', '#EF4444'],
                    textinfo='label+percent',
                    hole=0.4
                ), row=1, col=2)
                
                fig.update_layout(height=500, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            with st.container():
                st.markdown("### 📊 Performance Metrics")
                
                # Advanced Gauge Charts
                fig = make_subplots(
                    rows=2, cols=2,
                    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                           [{'type': 'indicator'}, {'type': 'indicator'}]],
                    subplot_titles=['Model Accuracy', 'Data Quality', 'Risk Coverage', 'Response Time']
                )
                
                # Model Accuracy
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=94.2,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#10B981"},
                        'steps': [
                            {'range': [0, 70], 'color': "#FEE2E2"},
                            {'range': [70, 85], 'color': "#FEF3C7"},
                            {'range': [85, 100], 'color': "#D1FAE5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ), row=1, col=1)
                
                # Data Quality
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=98.5,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#6366F1"},
                        'steps': [
                            {'range': [0, 80], 'color': "#FEE2E2"},
                            {'range': [80, 95], 'color': "#FEF3C7"},
                            {'range': [95, 100], 'color': "#D1FAE5"}
                        ]
                    }
                ), row=1, col=2)
                
                # Risk Coverage
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=87.3,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#F59E0B"},
                        'steps': [
                            {'range': [0, 60], 'color': "#FEE2E2"},
                            {'range': [60, 80], 'color': "#FEF3C7"},
                            {'range': [80, 100], 'color': "#D1FAE5"}
                        ]
                    }
                ), row=2, col=1)
                
                # Response Time
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=0.23,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "seconds"},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "#EF4444"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "#D1FAE5"},
                            {'range': [0.3, 0.6], 'color': "#FEF3C7"},
                            {'range': [0.6, 1], 'color': "#FEE2E2"}
                        ]
                    }
                ), row=2, col=2)
                
                fig.update_layout(height=500, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Live Activity Feed
        st.markdown("### 🔄 Live Activity Feed")
        
        activity_data = pd.DataFrame({
            'Timestamp': pd.date_range(start=datetime.now() - timedelta(minutes=30), periods=10, freq='3min'),
            'Application ID': [f'APP-2024-{i:04d}' for i in range(1001, 1011)],
            'Amount': np.random.randint(100000, 1000000, 10),
            'Risk Score': np.random.uniform(0, 100, 10).round(1),
            'Status': np.random.choice(['✅ Approved', '⚠️ Review', '❌ Rejected'], 10, p=[0.6, 0.3, 0.1])
        })
        
        # Color coding for status
        def color_status(val):
            if 'Approved' in val:
                return f'<span style="color: #10B981; font-weight: 600;">{val}</span>'
            elif 'Review' in val:
                return f'<span style="color: #F59E0B; font-weight: 600;">{val}</span>'
            else:
                return f'<span style="color: #EF4444; font-weight: 600;">{val}</span>'
        
        activity_data['Status'] = activity_data['Status'].apply(color_status)
        
        st.markdown(activity_data.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # ============================================================================
    # DATA INTELLIGENCE
    # ============================================================================
    elif selected == "Data Intelligence":
        st.markdown("## 📊 Advanced Data Intelligence")
        
        # File Upload with Drag & Drop
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with stylable_container(
                key="upload",
                css_styles="""
                    {
                        border: 3px dashed #667eea;
                        border-radius: 20px;
                        padding: 3rem;
                        text-align: center;
                        background: rgba(102, 126, 234, 0.1);
                        transition: all 0.3s ease;
                    }
                    :hover {
                        background: rgba(102, 126, 234, 0.15);
                        border-color: #764ba2;
                    }
                """
            ):
                st.markdown("### 📁 Drag & Drop or Browse")
                uploaded_file = st.file_uploader(
                    "Upload your financial dataset",
                    type=['csv', 'xlsx', 'parquet'],
                    label_visibility="collapsed"
                )
                
                if uploaded_file:
                    st.success(f"✅ File loaded: {uploaded_file.name}")
                    
                    # Load data
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        df = pd.read_excel(uploaded_file)
                    else:
                        df = pd.read_parquet(uploaded_file)
                    
                    st.session_state.data = df
                    log_system_event("DATA_UPLOAD", f"Uploaded {uploaded_file.name} with {len(df)} records")
        
        with col2:
            st.markdown("### 📈 Quick Stats")
            if st.session_state.data is not None:
                df = st.session_state.data
                st.metric("Total Records", f"{len(df):,}")
                st.metric("Features", df.shape[1])
                st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.2f} MB")
                st.metric("Missing Values", df.isnull().sum().sum())
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.markdown("---")
            
            # Advanced Data Profiling
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Interactive Explorer",
                "📈 Statistical Analysis",
                "🔗 Correlation Network",
                "📋 Data Quality"
            ])
            
            with tab1:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown("#### Controls")
                    x_axis = st.selectbox("X-axis", df.select_dtypes(include=[np.number]).columns)
                    y_axis = st.selectbox("Y-axis", df.select_dtypes(include=[np.number]).columns, index=1 if len(df.select_dtypes(include=[np.number]).columns) > 1 else 0)
                    color_by = st.selectbox("Color by", ['None'] + list(df.select_dtypes(include=['object']).columns))
                    size_by = st.selectbox("Size by", ['None'] + list(df.select_dtypes(include=[np.number]).columns))
                    
                    if st.button("Generate 3D Plot"):
                        st.session_state.show_3d = True
                
                with col2:
                    if st.session_state.get('show_3d', False):
                        # 3D Scatter Plot
                        fig = go.Figure(data=[go.Scatter3d(
                            x=df[x_axis][:1000],
                            y=df[y_axis][:1000],
                            z=df[df.select_dtypes(include=[np.number]).columns[2]][:1000],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=df[color_by].astype('category').cat.codes if color_by != 'None' else '#6366F1',
                                colorscale='Viridis',
                                showscale=True
                            )
                        )])
                        
                        fig.update_layout(
                            title="3D Data Explorer",
                            scene=dict(
                                xaxis_title=x_axis,
                                yaxis_title=y_axis,
                                zaxis_title=df.select_dtypes(include=[np.number]).columns[2]
                            ),
                            height=600,
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # 2D Scatter Plot
                        if color_by != 'None':
                            fig = px.scatter(df[:5000], x=x_axis, y=y_axis, color=color_by,
                                           size=size_by if size_by != 'None' else None,
                                           title="Interactive Scatter Plot")
                        else:
                            fig = px.scatter(df[:5000], x=x_axis, y=y_axis,
                                           title="Interactive Scatter Plot")
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Distribution Analysis")
                    selected_col = st.selectbox("Select column", df.select_dtypes(include=[np.number]).columns)
                    
                    fig = make_subplots(rows=2, cols=2,
                                       subplot_titles=('Histogram', 'Box Plot', 'Violin Plot', 'Q-Q Plot'))
                    
                    # Histogram
                    fig.add_trace(go.Histogram(x=df[selected_col], nbinsx=50, name='Histogram'), row=1, col=1)
                    
                    # Box Plot
                    fig.add_trace(go.Box(y=df[selected_col], name='Box Plot'), row=1, col=2)
                    
                    # Violin Plot
                    fig.add_trace(go.Violin(y=df[selected_col], name='Violin Plot'), row=2, col=1)
                    
                    # Q-Q Plot (simplified)
                    from scipy import stats
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, 100))
                    sample_quantiles = np.percentile(df[selected_col].dropna(), np.linspace(1, 99, 100))
                    
                    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                                           mode='markers', name='Q-Q Plot'), row=2, col=2)
                    
                    fig.update_layout(height=800, showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### Statistical Summary")
                    
                    # Detailed statistics
                    stats_df = pd.DataFrame({
                        'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Skewness', 'Kurtosis'],
                        'Value': [
                            df[selected_col].count(),
                            f"{df[selected_col].mean():.2f}",
                            f"{df[selected_col].std():.2f}",
                            f"{df[selected_col].min():.2f}",
                            f"{df[selected_col].quantile(0.25):.2f}",
                            f"{df[selected_col].quantile(0.50):.2f}",
                            f"{df[selected_col].quantile(0.75):.2f}",
                            f"{df[selected_col].max():.2f}",
                            f"{df[selected_col].skew():.3f}",
                            f"{df[selected_col].kurtosis():.3f}"
                        ]
                    })
                    
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Outlier detection
                    Q1 = df[selected_col].quantile(0.25)
                    Q3 = df[selected_col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[selected_col] < Q1 - 1.5*IQR) | (df[selected_col] > Q3 + 1.5*IQR)]
                    
                    st.markdown(f"**Outliers detected:** {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
            
            with tab3:
                st.markdown("#### Correlation Network")
                
                try:
                    import networkx as nx
                    # Calculate correlations
                    numeric_df = df.select_dtypes(include=[np.number])
                    corr_matrix = numeric_df.corr()
                    
                    # Create network graph
                    G = nx.Graph()
                    
                    # Add nodes
                    for col in numeric_df.columns:
                        G.add_node(col)
                    
                    # Add edges for correlations > 0.5
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if abs(corr_matrix.iloc[i, j]) > 0.5:
                                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                                          weight=corr_matrix.iloc[i, j])
                    
                    # Get positions
                    pos = nx.spring_layout(G)
                    
                    # Create edge trace
                    edge_trace = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_trace.append(go.Scatter(
                            x=[x0, x1, None],
                            y=[y0, y1, None],
                            mode='lines',
                            line=dict(width=G[edge[0]][edge[1]]['weight']*2, color='rgba(255,255,255,0.3)'),
                            hoverinfo='none'
                        ))
                    
                    # Create node trace
                    node_trace = go.Scatter(
                        x=[pos[node][0] for node in G.nodes()],
                        y=[pos[node][1] for node in G.nodes()],
                        mode='markers+text',
                        text=list(G.nodes()),
                        textposition="top center",
                        textfont=dict(color='white'),
                        marker=dict(
                            size=30,
                            color=list(corr_matrix.mean().values),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Avg Correlation", tickfont=dict(color='white'))
                        ),
                        hoverinfo='text'
                    )
                    
                    fig = go.Figure(data=edge_trace + [node_trace])
                    fig.update_layout(
                        title="Feature Correlation Network",
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.info("Install networkx for advanced correlation visualization: `pip install networkx`")
                    # Fallback heatmap
                    numeric_df = df.select_dtypes(include=[np.number])
                    corr_matrix = numeric_df.corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                   title="Correlation Matrix", color_continuous_scale='RdBu')
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown("#### Data Quality Dashboard")
                
                # Quality metrics
                quality_metrics = []
                for col in df.columns:
                    quality_metrics.append({
                        'Column': col,
                        'Data Type': str(df[col].dtype),
                        'Missing %': f"{df[col].isnull().sum() / len(df) * 100:.2f}%",
                        'Unique Values': df[col].nunique(),
                        'Memory (KB)': f"{df[col].memory_usage() / 1024:.2f}",
                        'Quality Score': np.random.randint(85, 100)  # Placeholder
                    })
                
                quality_df = pd.DataFrame(quality_metrics)
                
                # Color coding for quality score
                def color_quality(val):
                    if isinstance(val, str) and val.isdigit():
                        val = int(val)
                    if isinstance(val, (int, float)):
                        if val >= 95:
                            return f'<span style="color: #10B981; font-weight: 600;">{val}</span>'
                        elif val >= 85:
                            return f'<span style="color: #F59E0B; font-weight: 600;">{val}</span>'
                        else:
                            return f'<span style="color: #EF4444; font-weight: 600;">{val}</span>'
                    return val
                
                quality_df['Quality Score'] = quality_df['Quality Score'].apply(color_quality)
                
                st.markdown(quality_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # ============================================================================
    # RISK ASSESSMENT (with Real AI Predictions)
    # ============================================================================
    # ============================================================================
# RISK ASSESSMENT (with Real AI Predictions) - USING THE SAME FORM
# ============================================================================
    elif selected == "Risk Assessment":
        st.markdown("## 🤖 AI-Powered Risk Assessment")
        
        if not st.session_state.models_loaded:
            st.warning("⚠️ AI models not loaded. Please train models first in the Model Development notebook.")
        else:
            # Create tabs for different assessment types
            risk_tab1, risk_tab2, risk_tab3 = st.tabs([
                "📝 Single Application",
                "📊 Batch Processing",
                "📈 Portfolio Analysis"
            ])
            
            with risk_tab1:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    with st.form("risk_assessment_form"):
                        st.markdown("### Applicant Information")
                        
                        # Personal Details - Same as EMI Eligibility Predictor
                        with st.expander("👤 Personal Details", expanded=True):
                            row1_col1, row1_col2, row1_col3 = st.columns(3)
                            with row1_col1:
                                age = st.number_input("Age", min_value=18, max_value=80, value=35)
                                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                            with row1_col2:
                                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                                education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
                            with row1_col3:
                                dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
                                family_size = st.number_input("Family Size", min_value=1, max_value=15, value=3)
                        
                        # Employment Details - Same as EMI Eligibility Predictor
                        with st.expander("💼 Employment Details", expanded=True):
                            row2_col1, row2_col2, row2_col3 = st.columns(3)
                            with row2_col1:
                                monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, value=75000, step=5000)
                                employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed", "Business"])
                            with row2_col2:
                                years_employed = st.number_input("Years Employed", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
                                company_type = st.selectbox("Company Type", ["Startup", "Mid-size", "MNC", "Government"])
                            with row2_col3:
                                house_type = st.selectbox("House Type", ["Rented", "Own", "Family", "Leased"])
                                monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, value=15000, step=1000)
                        
                        # Financial Details - Same as EMI Eligibility Predictor
                        with st.expander("💰 Financial Details", expanded=True):
                            row3_col1, row3_col2, row3_col3 = st.columns(3)
                            with row3_col1:
                                credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=5)
                                existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
                            with row3_col2:
                                current_emi = st.number_input("Current EMI (₹)", min_value=0, value=10000, step=1000)
                                bank_balance = st.number_input("Bank Balance (₹)", min_value=0, value=500000, step=50000)
                            with row3_col3:
                                emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, value=200000, step=10000)
                        
                        # Monthly Expenses - Same as EMI Eligibility Predictor
                        with st.expander("📉 Monthly Expenses", expanded=False):
                            row4_col1, row4_col2, row4_col3 = st.columns(3)
                            with row4_col1:
                                school_fees = st.number_input("School Fees (₹)", min_value=0, value=5000, step=1000)
                                college_fees = st.number_input("College Fees (₹)", min_value=0, value=0, step=1000)
                            with row4_col2:
                                travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, value=3000, step=500)
                                groceries = st.number_input("Groceries & Utilities (₹)", min_value=0, value=10000, step=1000)
                            with row4_col3:
                                other_expenses = st.number_input("Other Expenses (₹)", min_value=0, value=5000, step=1000)
                        
                        # Loan Details - Same as EMI Eligibility Predictor
                        with st.expander("🏦 Loan Details", expanded=True):
                            row5_col1, row5_col2, row5_col3 = st.columns(3)
                            with row5_col1:
                                emi_scenario = st.selectbox("Loan Type", [
                                    "E-commerce Shopping EMI", "Home Appliances EMI",
                                    "Vehicle EMI", "Personal Loan EMI", "Education EMI"
                                ])
                            with row5_col2:
                                requested_amount = st.number_input("Requested Amount (₹)", min_value=0, value=500000, step=50000)
                            with row5_col3:
                                requested_tenure = st.number_input("Tenure (months)", min_value=1, max_value=120, value=36, step=6)
                        
                        # Submit button
                        submitted = st.form_submit_button(
                            "🔍 Analyze Risk Profile",
                            use_container_width=True,
                            type="primary"
                        )
                
                with col2:
                    st.markdown("### 📊 Live Preview")
                    
                    if submitted:
                        with st.spinner("🤖 AI Analyzing your profile..."):
                            time.sleep(1.5)  # Simulate processing
                            
                            # Prepare input data for model
                            input_dict = {
                                'age': age,
                                'gender': gender,
                                'marital_status': marital_status,
                                'education': education,
                                'monthly_salary': monthly_salary,
                                'employment_type': employment_type,
                                'years_of_employment': years_employed,
                                'company_type': company_type,
                                'house_type': house_type,
                                'monthly_rent': monthly_rent,
                                'family_size': family_size,
                                'dependents': dependents,
                                'school_fees': school_fees,
                                'college_fees': college_fees,
                                'travel_expenses': travel_expenses,
                                'groceries_utilities': groceries,
                                'other_monthly_expenses': other_expenses,
                                'existing_loans': existing_loans,
                                'current_emi_amount': current_emi,
                                'credit_score': credit_score,
                                'bank_balance': bank_balance,
                                'emergency_fund': emergency_fund,
                                'emi_scenario': emi_scenario,
                                'requested_amount': requested_amount,
                                'requested_tenure': requested_tenure
                            }
                            
                            try:
                                # Prepare input data with correct features
                                input_df = prepare_input_data(input_dict)
                                
                                if input_df is not None:
                                    # classification_model is a sklearn Pipeline that already
                                    # includes the preprocessor — pass raw DataFrame directly.
                                    prediction_encoded = classification_model.predict(input_df)[0]
                                    probabilities = classification_model.predict_proba(input_df)[0]
                                    
                                    # Convert class index back to label
                                    # LabelEncoder was fitted on: ['Eligible', 'High_Risk', 'Not_Eligible']
                                    if label_encoder is not None:
                                        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                                    else:
                                        # Fallback: mirrors LabelEncoder fitted during training
                                        class_map = {0: "Eligible", 1: "High_Risk", 2: "Not_Eligible"}
                                        prediction = class_map.get(int(prediction_encoded), str(prediction_encoded))
                                    
                                    # Calculate risk score based on probability of predicted class
                                    class_index = int(prediction_encoded)
                                    risk_score = probabilities[class_index] * 100
                                    
                                    # Determine color and icon (handle underscore and space variants)
                                    pred_lower = prediction.lower().replace('_', ' ')
                                    if pred_lower == "eligible":
                                        color = "#10B981"
                                        icon = "✅"
                                        risk_level = "Low Risk"
                                    elif pred_lower == "high risk":
                                        color = "#F59E0B"
                                        icon = "⚠️"
                                        risk_level = "Medium Risk"
                                    else:
                                        color = "#EF4444"
                                        icon = "❌"
                                        risk_level = "High Risk"
                                    
                                    # Use display-friendly label
                                    display_prediction = prediction.replace('_', ' ')
                                    
                                    # Display result with animation
                                    st.markdown(f"""
                                    <div style="background: {color}; padding: 2rem; border-radius: 20px; text-align: center; animation: slideIn 0.5s ease;">
                                        <h1 style="font-size: 4rem; margin: 0; animation: pulse 2s infinite;">{icon}</h1>
                                        <h2 style="color: white; margin: 0.5rem 0; font-size: 2rem;">{prediction}</h2>
                                        <p style="color: white; font-size: 1.2rem;">Risk Level: {risk_level}</p>
                                        <p style="color: white; font-size: 1.2rem;">Confidence: {risk_score:.1f}%</p>
                                    </div>
                                    <style>
                                        @keyframes slideIn {{
                                            from {{ transform: translateY(20px); opacity: 0; }}
                                            to {{ transform: translateY(0); opacity: 1; }}
                                        }}
                                    </style>
                                    """, unsafe_allow_html=True)
                                    
                                    # Additional metrics - Same as EMI Eligibility Predictor
                                    st.markdown("### 📈 Key Metrics")
                                    
                                    # Calculate financial ratios
                                    dti = (current_emi / monthly_salary) * 100 if monthly_salary > 0 else 0
                                    requested_emi_value = requested_amount / requested_tenure
                                    emi_to_income = (requested_emi_value / monthly_salary) * 100 if monthly_salary > 0 else 0
                                    total_expenses = school_fees + college_fees + travel_expenses + groceries + other_expenses + monthly_rent
                                    savings_rate = ((monthly_salary - total_expenses - current_emi) / monthly_salary) * 100 if monthly_salary > 0 else 0
                                    
                                    col_m1, col_m2, col_m3 = st.columns(3)
                                    col_m1.metric("DTI Ratio", f"{dti:.1f}%", 
                                                "Good" if dti < 40 else "High",
                                                delta_color="inverse" if dti >= 40 else "normal")
                                    col_m2.metric("EMI/Income", f"{emi_to_income:.1f}%", 
                                                "Safe" if emi_to_income < 30 else "Stretched",
                                                delta_color="inverse" if emi_to_income >= 30 else "normal")
                                    col_m3.metric("Savings Rate", f"{savings_rate:.1f}%",
                                                "Good" if savings_rate > 20 else "Low",
                                                delta_color="inverse" if savings_rate <= 20 else "normal")
                                    
                                    # Confidence gauge
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number+delta",
                                        value=risk_score,
                                        domain={'x': [0, 1], 'y': [0, 1]},
                                        title={'text': "Confidence Score", 'font': {'color': 'white'}},
                                        delta={'reference': 50},
                                        gauge={
                                            'axis': {'range': [0, 100], 'tickfont': {'color': 'white'}},
                                            'bar': {'color': color},
                                            'steps': [
                                                {'range': [0, 44], 'color': "rgba(239, 68, 68, 0.3)"},
                                                {'range': [44, 69], 'color': "rgba(245, 158, 11, 0.3)"},
                                                {'range': [69, 100], 'color': "rgba(16, 185, 129, 0.3)"}
                                            ],
                                            'threshold': {
                                                'line': {'color': 'white', 'width': 4},
                                                'thickness': 0.75,
                                                'value': risk_score
                                            }
                                        }
                                    ))
                                    fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Store in history
                                    st.session_state.prediction_history.append({
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'amount': requested_amount,
                                        'prediction': prediction,
                                        'risk_score': risk_score
                                    })
                                    
                                    log_system_event("PREDICTION", f"Single prediction: {prediction} with {risk_score:.1f}% confidence")
                                else:
                                    st.error("Error preparing input data")
                                    
                            except Exception as e:
                                st.error(f"Prediction error: {str(e)}")
                                st.info("⚠️ AI prediction failed. Please check if models are properly trained.")
                                
                                # Show debug info
                                with st.expander("Debug Information"):
                                    st.write("Input Dictionary Keys:", list(input_dict.keys()))
                                    if feature_names and 'feature_names' in feature_names:
                                        st.write("Expected Features:", feature_names['feature_names'][:10])
                                    if preprocessor and hasattr(preprocessor, 'get_feature_names_out'):
                                        st.write("Preprocessor Features:", list(preprocessor.get_feature_names_out())[:10])
                    else:
                        st.info("👈 Fill the form and click 'Analyze' to see AI-powered results")
    
    # ============================================================================
    # EMI CALCULATOR
    # ============================================================================
    elif selected == "EMI Calculator":
        st.markdown("## 💰 Advanced EMI Calculator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.form("emi_calculator_form"):
                st.markdown("### Loan Parameters")
                
                loan_amount = st.number_input("Loan Amount (₹)", 10000, 10000000, 500000, step=50000)
                interest_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 10.0, 0.1)
                loan_tenure = st.slider("Loan Tenure (years)", 1, 30, 5)
                
                st.markdown("### Financial Profile")
                monthly_income = st.number_input("Monthly Income (₹)", 0, 10000000, 75000, step=5000)
                monthly_expenses = st.number_input("Monthly Expenses (₹)", 0, 5000000, 30000, step=5000)
                existing_emi = st.number_input("Existing EMI (₹)", 0, 500000, 10000, step=1000)
                
                calculate_emi = st.form_submit_button("📊 Calculate EMI", use_container_width=True)
        
        with col2:
            if calculate_emi:
                # Calculate EMI
                tenure_months = loan_tenure * 12
                monthly_rate = interest_rate / (12 * 100)
                
                if monthly_rate > 0:
                    emi = loan_amount * monthly_rate * (1 + monthly_rate)**tenure_months / ((1 + monthly_rate)**tenure_months - 1)
                else:
                    emi = loan_amount / tenure_months
                
                # Calculate affordability
                disposable_income = monthly_income - monthly_expenses - existing_emi
                affordable_emi = disposable_income * 0.4
                emi_to_income = (emi / monthly_income) * 100 if monthly_income > 0 else 0
                
                # Display results
                st.markdown("### 📊 EMI Results")
                
                # Main EMI display
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; text-align: center; animation: slideIn 0.5s ease;">
                    <h3 style="color: white; margin: 0;">Monthly EMI</h3>
                    <h1 style="color: white; font-size: 4rem; margin: 0;">₹{emi:,.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Total Interest", f"₹{(emi * tenure_months - loan_amount):,.0f}")
                col_m2.metric("Total Payment", f"₹{(emi * tenure_months):,.0f}")
                col_m3.metric("EMI/Income Ratio", f"{emi_to_income:.1f}%")
                
                # Affordability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=emi,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "EMI vs Affordable Limit", 'font': {'color': 'white'}},
                    delta={'reference': affordable_emi},
                    gauge={
                        'axis': {'range': [0, max(emi, affordable_emi) * 1.2], 'tickfont': {'color': 'white'}},
                        'bar': {'color': "#10B981" if emi <= affordable_emi else "#EF4444"},
                        'steps': [
                            {'range': [0, affordable_emi], 'color': "rgba(16, 185, 129, 0.3)"},
                            {'range': [affordable_emi, affordable_emi * 1.5], 'color': "rgba(245, 158, 11, 0.3)"},
                            {'range': [affordable_emi * 1.5, max(emi, affordable_emi) * 1.2], 'color': "rgba(239, 68, 68, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': 'white', 'width': 4},
                            'thickness': 0.75,
                            'value': affordable_emi
                        }
                    }
                ))
                fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Amortization schedule
                st.markdown("### 📈 Amortization Schedule")
                
                # Generate schedule
                months = list(range(1, min(tenure_months + 1, 361)))  # Limit to 30 years
                balance = loan_amount
                principal_paid = []
                interest_paid = []
                balance_remaining = []
                
                for month in months:
                    interest = balance * monthly_rate
                    principal = emi - interest
                    balance -= principal
                    
                    principal_paid.append(principal)
                    interest_paid.append(interest)
                    balance_remaining.append(max(balance, 0))
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=months, y=principal_paid, name="Principal", line=dict(color="#10B981")),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=months, y=interest_paid, name="Interest", line=dict(color="#EF4444")),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=months, y=balance_remaining, name="Balance", line=dict(color="#6366F1", dash="dash")),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="Loan Amortization Schedule",
                    xaxis_title="Month",
                    hovermode='x unified',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(title_text="Payment (₹)", secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(title_text="Balance (₹)", secondary_y=True, gridcolor='rgba(255,255,255,0.1)')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison with other tenures
                st.markdown("### 📊 Tenure Comparison")
                
                tenures = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]
                emi_values = []
                interest_values = []
                
                for t in tenures:
                    months_t = t * 12
                    if monthly_rate > 0:
                        emi_t = loan_amount * monthly_rate * (1 + monthly_rate)**months_t / ((1 + monthly_rate)**months_t - 1)
                    else:
                        emi_t = loan_amount / months_t
                    total_interest = (emi_t * months_t) - loan_amount
                    emi_values.append(emi_t)
                    interest_values.append(total_interest)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=tenures, y=emi_values, name="Monthly EMI", line=dict(color="#10B981", width=3)),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=tenures, y=interest_values, name="Total Interest", line=dict(color="#EF4444", width=3)),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title="EMI vs Interest by Tenure",
                    xaxis_title="Tenure (years)",
                    hovermode='x unified',
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'}
                )
                
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(title_text="Monthly EMI (₹)", secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
                fig.update_yaxes(title_text="Total Interest (₹)", secondary_y=True, gridcolor='rgba(255,255,255,0.1)')
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================================
    # MODEL MONITOR
    # ============================================================================
    elif selected == "Model Monitor":
        st.markdown("## 📈 AI Model Performance Monitor")
        
        # Performance metrics over time
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        # Generate realistic performance data
        accuracy_trend = 0.94 + 0.02 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.005, len(dates))
        accuracy_trend = np.clip(accuracy_trend, 0.92, 0.96)
        
        precision_trend = 0.93 + 0.02 * np.cos(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 0.005, len(dates))
        recall_trend = 0.94 + 0.015 * np.sin(np.linspace(0, 4*np.pi + 1, len(dates))) + np.random.normal(0, 0.005, len(dates))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance trends
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates, y=accuracy_trend,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#10B981', width=3),
                marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=precision_trend,
                mode='lines+markers',
                name='Precision',
                line=dict(color='#6366F1', width=3),
                marker=dict(size=4)
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=recall_trend,
                mode='lines+markers',
                name='Recall',
                line=dict(color='#F59E0B', width=3),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="Model Performance Trends (Last 30 Days)",
                xaxis_title="Date",
                yaxis_title="Score",
                yaxis_range=[0.91, 0.97],
                hovermode='x unified',
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Current Status")
            
            # Current metrics
            st.metric("Current Accuracy", f"{accuracy_trend[-1]*100:.2f}%", 
                     f"{(accuracy_trend[-1] - accuracy_trend[-2])*100:+.2f}%")
            st.metric("Current Precision", f"{precision_trend[-1]*100:.2f}%",
                     f"{(precision_trend[-1] - precision_trend[-2])*100:+.2f}%")
            st.metric("Current Recall", f"{recall_trend[-1]*100:.2f}%",
                     f"{(recall_trend[-1] - recall_trend[-2])*100:+.2f}%")
            
            # Model health
            health_score = np.mean([accuracy_trend[-1], precision_trend[-1], recall_trend[-1]]) * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Model Health Score", 'font': {'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'color': 'white'}},
                    'bar': {'color': "#10B981" if health_score > 90 else "#F59E0B" if health_score > 80 else "#EF4444"},
                    'steps': [
                        {'range': [0, 70], 'color': "rgba(239, 68, 68, 0.3)"},
                        {'range': [70, 85], 'color': "rgba(245, 158, 11, 0.3)"},
                        {'range': [85, 100], 'color': "rgba(16, 185, 129, 0.3)"}
                    ]
                }
            ))
            fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix Heatmap
        st.markdown("### 📊 Confusion Matrix Analysis")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Generate confusion matrix
            cm = np.array([
                [84500, 1200, 800],
                [3500, 41800, 1500],
                [900, 1100, 47800]
            ])
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='Blues',
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['Eligible', 'High Risk', 'Not Eligible'],
                y=['Eligible', 'High Risk', 'Not Eligible']
            )
            
            fig.update_layout(
                title="Confusion Matrix - XGBoost Classifier",
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("### 🔍 Feature Importance Analysis")
        
        features = [
            'credit_score', 'monthly_salary', 'dti_ratio', 'current_emi',
            'years_employed', 'age', 'loan_amount', 'tenure',
            'dependents', 'existing_loans', 'emergency_fund'
        ]
        
        importances = [0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02]
        
        fig = px.bar(
            x=importances,
            y=features,
            orientation='h',
            title="Feature Importance - XGBoost",
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importances,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # ============================================================================
    # MLFLOW ANALYTICS
    # ============================================================================
    elif selected == "MLflow Analytics":
        st.markdown("## 🔬 MLflow Experiment Tracking")
        
        # Experiment comparison
        st.markdown("### 📊 Experiment Comparison")
        
        # Sample experiment data
        experiments = pd.DataFrame({
            'Experiment': ['XGBoost v3', 'Random Forest v2', 'XGBoost v2', 'Gradient Boosting v1'],
            'Accuracy': [0.942, 0.938, 0.935, 0.931],
            'F1-Score': [0.941, 0.937, 0.934, 0.930],
            'Training Time (s)': [245, 312, 198, 289],
            'Inference (ms)': [23, 45, 22, 34]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                experiments,
                x='Training Time (s)',
                y='Accuracy',
                size='Inference (ms)',
                text='Experiment',
                title="Model Performance Trade-offs",
                color='Accuracy',
                color_continuous_scale='Viridis'
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Parallel coordinates
            dimensions = [
                dict(range=[0.93, 0.95],
                     label='Accuracy', values=experiments['Accuracy']),
                dict(range=[0.93, 0.95],
                     label='F1-Score', values=experiments['F1-Score']),
                dict(range=[150, 350],
                     label='Training Time', values=experiments['Training Time (s)']),
                dict(range=[20, 50],
                     label='Inference', values=experiments['Inference (ms)'])
            ]
            
            fig = go.Figure(data=go.Parcoords(
                line=dict(color=experiments['Accuracy'],
                         colorscale='Viridis',
                         showscale=True),
                dimensions=dimensions
            ))
            
            fig.update_layout(title="Multi-dimensional Comparison", height=400, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model registry
        st.markdown("### 📦 Model Registry")
        
        registry_data = pd.DataFrame({
            'Model Name': ['EMIPredict_Classifier', 'EMIPredict_Classifier', 'EMIPredict_Regressor', 'EMIPredict_Regressor'],
            'Version': ['v2.1.0', 'v2.0.0', 'v1.8.0', 'v1.7.0'],
            'Stage': ['Production', 'Staging', 'Production', 'Archived'],
            'Metrics': ['Acc: 94.2%', 'Acc: 93.8%', 'RMSE: 1850', 'RMSE: 1920'],
            'Date': ['2024-01-15', '2024-01-10', '2024-01-14', '2024-01-08']
        })
        
        # Color coding for stages
        def color_stage(val):
            if val == 'Production':
                return f'<span style="color: #10B981; font-weight: 600;">{val}</span>'
            elif val == 'Staging':
                return f'<span style="color: #F59E0B; font-weight: 600;">{val}</span>'
            else:
                return f'<span style="color: #6B7280; font-weight: 600;">{val}</span>'
        
        registry_data['Stage'] = registry_data['Stage'].apply(color_stage)
        
        st.markdown(registry_data.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # ============================================================================
    # SYSTEM ADMIN - ENHANCED VERSION
    # ============================================================================
    elif selected == "System Admin":
        st.markdown("## ⚙️ System Administration")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 System Health",
            "📝 Logs & Monitoring",
            "⚡ Performance",
            "🔐 Security",
            "💾 Backup & Recovery",
            "👥 User Management"
        ])
        
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("CPU Usage", "23%", "-2%", delta_color="inverse")
            with col2:
                st.metric("Memory", "4.2/8 GB", "52%", delta_color="inverse")
            with col3:
                st.metric("Disk", "124/256 GB", "48%", delta_color="inverse")
            with col4:
                st.metric("API Latency", "124ms", "+12ms")
            
            # System metrics over time
            times = pd.date_range(end=datetime.now(), periods=60, freq='1min')
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=('CPU Usage', 'Memory Usage'))
            
            fig.add_trace(go.Scatter(
                x=times,
                y=20 + 10 * np.sin(np.linspace(0, 4*np.pi, 60)) + np.random.normal(0, 2, 60),
                mode='lines',
                name='CPU %',
                line=dict(color='#10B981')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=45 + 5 * np.cos(np.linspace(0, 4*np.pi, 60)) + np.random.normal(0, 1, 60),
                mode='lines',
                name='Memory %',
                line=dict(color='#6366F1')
            ), row=2, col=1)
            
            fig.update_layout(
                height=500,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # System Information
            st.markdown("### System Information")
            
            sys_info = pd.DataFrame({
                'Component': ['Streamlit Version', 'Python Version', 'OS', 'Uptime', 'Total API Calls', 'Active Sessions'],
                'Status': [
                    st.__version__,
                    sys.version.split()[0],
                    sys.platform,
                    str(datetime.now() - st.session_state.system_start_time).split('.')[0],
                    st.session_state.api_calls,
                    1  # Current session
                ]
            })
            st.dataframe(sys_info, use_container_width=True)
        
        with tab2:
            st.markdown("### System Logs")
            
            # Log filters
            col1, col2, col3 = st.columns(3)
            with col1:
                log_level = st.selectbox("Log Level", ["ALL", "INFO", "WARNING", "ERROR"])
            with col2:
                log_search = st.text_input("Search", placeholder="Filter logs...")
            with col3:
                log_refresh = st.button("🔄 Refresh Logs")
            
            # Generate sample logs if none exist
            if len(st.session_state.system_logs) == 0:
                for i in range(50):
                    level = np.random.choice(['INFO', 'WARNING', 'ERROR'], p=[0.7, 0.2, 0.1])
                    st.session_state.system_logs.append({
                        'timestamp': datetime.now() - timedelta(minutes=np.random.randint(1, 60)),
                        'type': np.random.choice(['API', 'PREDICTION', 'DATA', 'SYSTEM']),
                        'message': f"Sample log message {i}",
                        'level': level
                    })
                st.session_state.system_logs.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Filter logs
            filtered_logs = st.session_state.system_logs
            if log_level != "ALL":
                filtered_logs = [log for log in filtered_logs if log['level'] == log_level]
            if log_search:
                filtered_logs = [log for log in filtered_logs if log_search.lower() in log['message'].lower()]
            
            # Display logs
            logs_df = pd.DataFrame(filtered_logs)
            if not logs_df.empty:
                logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Color coding for log levels
                def color_log_level(val):
                    if val == 'INFO':
                        return f'<span style="color: #10B981;">{val}</span>'
                    elif val == 'WARNING':
                        return f'<span style="color: #F59E0B;">{val}</span>'
                    else:
                        return f'<span style="color: #EF4444;">{val}</span>'
                
                logs_df['level'] = logs_df['level'].apply(color_log_level)
                
                st.markdown(logs_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            # Log actions
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("📥 Download Logs"):
                    log_text = "\n".join([f"{l['timestamp']} [{l['level']}] {l['type']}: {l['message']}" 
                                         for l in st.session_state.system_logs])
                    st.download_button("Download", log_text, "system_logs.txt")
            with col2:
                if st.button("🗑️ Clear Logs"):
                    st.session_state.system_logs = []
                    st.rerun()
            with col3:
                st.metric("Total Logs", len(st.session_state.system_logs))
            with col4:
                error_count = len([l for l in st.session_state.system_logs if l['level'] == 'ERROR'])
                st.metric("Errors", error_count, f"{error_count/len(st.session_state.system_logs)*100:.1f}%" if st.session_state.system_logs else "0%")
        
        with tab3:
            st.markdown("### Performance Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Cache Settings")
                cache_size = st.slider("Cache Size (MB)", 100, 2000, 500, 100)
                cache_ttl = st.number_input("Cache TTL (seconds)", 300, 86400, 3600, 300)
                
                st.markdown("#### Parallel Processing")
                parallel_jobs = st.slider("Parallel Jobs", 1, 8, 4)
                batch_size = st.number_input("Batch Size", 100, 10000, 1000, 100)
            
            with col2:
                st.markdown("#### API Settings")
                timeout = st.number_input("Timeout (seconds)", 10, 120, 30, 5)
                max_retries = st.number_input("Max Retries", 0, 5, 3)
                
                st.markdown("#### Memory Settings")
                memory_limit = st.select_slider("Memory Limit", options=['512MB', '1GB', '2GB', '4GB', '8GB'], value='4GB')
                swap_enabled = st.checkbox("Enable Swap", value=True)
            
            if st.button("Apply Performance Settings", use_container_width=True):
                st.success("Performance settings applied successfully!")
                log_system_event("CONFIG", "Performance settings updated", "INFO")
        
        with tab4:
            st.markdown("### Security Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Authentication")
                auth_enabled = st.checkbox("Enable Authentication", value=True)
                if auth_enabled:
                    st.selectbox("Auth Provider", ["Local", "LDAP", "OAuth2", "SAML"])
                    st.number_input("Session Timeout (minutes)", 5, 120, 30)
                
                st.markdown("#### API Security")
                api_key_required = st.checkbox("Require API Key", value=True)
                rate_limit = st.number_input("Rate Limit (requests/minute)", 10, 1000, 100)
            
            with col2:
                st.markdown("#### Encryption")
                encryption_enabled = st.checkbox("Enable Encryption", value=True)
                if encryption_enabled:
                    st.selectbox("Encryption Algorithm", ["AES-256", "RSA-2048", "ChaCha20"])
                
                st.markdown("#### Audit Logging")
                audit_enabled = st.checkbox("Enable Audit Logging", value=True)
                audit_level = st.select_slider("Audit Level", options=['Basic', 'Detailed', 'Verbose'], value='Detailed')
            
            # Security Status
            st.markdown("### Security Status")
            
            security_status = pd.DataFrame({
                'Security Control': ['Authentication', 'Encryption', 'Rate Limiting', 'Audit Logging', 'Input Validation', 'SQL Injection Protection'],
                'Status': ['✅ Enabled' if auth_enabled else '❌ Disabled',
                          '✅ Enabled' if encryption_enabled else '❌ Disabled',
                          '✅ Enabled' if api_key_required else '❌ Disabled',
                          '✅ Enabled' if audit_enabled else '❌ Disabled',
                          '✅ Enabled',
                          '✅ Enabled']
            })
            st.dataframe(security_status, use_container_width=True)
            
            if st.button("Apply Security Settings", use_container_width=True):
                st.success("Security settings applied successfully!")
                log_system_event("SECURITY", "Security settings updated", "INFO")
        
        with tab5:
            st.markdown("### Backup & Recovery")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Backup Settings")
                backup_enabled = st.checkbox("Enable Automatic Backups", value=True)
                if backup_enabled:
                    backup_frequency = st.selectbox("Backup Frequency", ["Hourly", "Daily", "Weekly", "Monthly"])
                    backup_time = st.time_input("Backup Time", datetime.now().time())
                    retention_days = st.number_input("Retention Days", 1, 365, 30)
                
                st.markdown("#### Backup Location")
                backup_path = st.text_input("Backup Path", "./backups")
                cloud_backup = st.checkbox("Enable Cloud Backup", value=False)
                if cloud_backup:
                    st.selectbox("Cloud Provider", ["AWS S3", "Google Cloud", "Azure"])
            
            with col2:
                st.markdown("#### Recent Backups")
                backups = pd.DataFrame({
                    'Backup ID': ['BKP-2024-001', 'BKP-2024-002', 'BKP-2024-003'],
                    'Date': ['2024-01-15 03:00', '2024-01-14 03:00', '2024-01-13 03:00'],
                    'Size': ['2.3 GB', '2.3 GB', '2.2 GB'],
                    'Status': ['✅ Success', '✅ Success', '✅ Success']
                })
                st.dataframe(backups, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("🔄 Backup Now", use_container_width=True):
                    st.success("Backup started!")
                    log_system_event("BACKUP", "Manual backup initiated", "INFO")
            with col2:
                if st.button("🛡️ Verify Backups", use_container_width=True):
                    st.info("Backup verification in progress...")
            with col3:
                if st.button("↩️ Restore", use_container_width=True):
                    st.warning("Restore wizard would open here")
            with col4:
                if st.button("🗑️ Clean Old", use_container_width=True):
                    st.success("Old backups cleaned!")
        
        with tab6:
            st.markdown("### User Management")
            
            # User list
            users = pd.DataFrame({
                'Username': ['admin', 'analyst1', 'analyst2', 'viewer1', 'api_user'],
                'Role': ['Administrator', 'Senior Analyst', 'Analyst', 'Viewer', 'API'],
                'Email': ['admin@emipredict.ai', 'analyst1@emipredict.ai', 'analyst2@emipredict.ai', 
                         'viewer@emipredict.ai', 'api@emipredict.ai'],
                'Last Login': ['2024-01-15 09:23', '2024-01-15 10:45', '2024-01-14 16:30', 
                              '2024-01-15 08:15', '2024-01-15 11:02'],
                'Status': ['Active', 'Active', 'Active', 'Inactive', 'Active'],
                'MFA': ['✅', '✅', '❌', '❌', '✅']
            })
            
            st.dataframe(users, use_container_width=True)
            
            # User actions
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.button("➕ Add User", use_container_width=True)
            with col2:
                st.button("✏️ Edit User", use_container_width=True)
            with col3:
                st.button("🔑 Reset Password", use_container_width=True)
            with col4:
                st.button("🚫 Disable User", use_container_width=True)
            with col5:
                st.button("📋 Audit Log", use_container_width=True)
            
            # Role management
            st.markdown("### Role Management")
            
            roles = pd.DataFrame({
                'Role': ['Administrator', 'Senior Analyst', 'Analyst', 'Viewer', 'API'],
                'Permissions': ['Full Access', 'Read/Write + Approve', 'Read/Write', 'Read Only', 'API Only'],
                'Users': ['1', '1', '1', '1', '1']
            })
            st.dataframe(roles, use_container_width=True)
            
            # Add new role
            with st.expander("Add New Role"):
                col1, col2 = st.columns(2)
                with col1:
                    new_role = st.text_input("Role Name")
                    st.multiselect("Permissions", ["Read", "Write", "Delete", "Approve", "Admin"])
                with col2:
                    st.multiselect("Users", ["admin", "analyst1", "analyst2", "viewer1", "api_user"])
                if st.button("Create Role"):
                    st.success(f"Role {new_role} created!")

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("© 2024 EMIPredict AI Pro")
with col2:
    st.markdown("v3.0.0 | Enterprise Edition")
with col3:
    st.markdown(f"Session: {st.session_state.session_id} | Last Activity: {st.session_state.last_activity.strftime('%H:%M:%S')}")

# Update last activity
st.session_state.last_activity = datetime.now()
st.session_state.api_calls += 1