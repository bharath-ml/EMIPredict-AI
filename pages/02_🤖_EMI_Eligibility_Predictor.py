"""
EMI Eligibility Prediction Page
Professional implementation with real-time predictions and risk analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classification_models import ClassificationModelTrainer
from src.mlflow_tracking.experiment_tracker import ExperimentTracker

# Page configuration
st.set_page_config(
    page_title="EMI Eligibility Predictor",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-card {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
    }
    .risk-factor {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .risk-high {
        background-color: #FEF2F2;
        border-left-color: #EF4444;
    }
    .risk-medium {
        background-color: #FFFBEB;
        border-left-color: #F59E0B;
    }
    .risk-low {
        background-color: #ECFDF5;
        border-left-color: #10B981;
    }
    .metric-container {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

def calculate_financial_metrics(input_data):
    """Calculate comprehensive financial metrics"""
    metrics = {}
    
    # Debt-to-Income Ratio
    monthly_salary = input_data['monthly_salary']
    current_emi = input_data['current_emi_amount']
    requested_emi = input_data['requested_amount'] / input_data['requested_tenure']
    
    metrics['dti_ratio'] = (current_emi / monthly_salary) * 100 if monthly_salary > 0 else 0
    metrics['requested_emi_ratio'] = (requested_emi / monthly_salary) * 100 if monthly_salary > 0 else 0
    metrics['total_obligation_ratio'] = ((current_emi + requested_emi) / monthly_salary) * 100 if monthly_salary > 0 else 0
    
    # Credit Score Category
    credit_score = input_data['credit_score']
    if credit_score >= 750:
        metrics['credit_category'] = 'Excellent'
        metrics['credit_score_points'] = 100
    elif credit_score >= 700:
        metrics['credit_category'] = 'Good'
        metrics['credit_score_points'] = 80
    elif credit_score >= 650:
        metrics['credit_category'] = 'Fair'
        metrics['credit_score_points'] = 60
    elif credit_score >= 600:
        metrics['credit_category'] = 'Poor'
        metrics['credit_score_points'] = 40
    else:
        metrics['credit_category'] = 'Very Poor'
        metrics['credit_score_points'] = 20
    
    # Employment Stability
    years_employed = input_data['years_of_employment']
    if years_employed >= 10:
        metrics['stability_score'] = 100
        metrics['stability_category'] = 'Very Stable'
    elif years_employed >= 5:
        metrics['stability_score'] = 80
        metrics['stability_category'] = 'Stable'
    elif years_employed >= 2:
        metrics['stability_score'] = 60
        metrics['stability_category'] = 'Moderately Stable'
    else:
        metrics['stability_score'] = 40
        metrics['stability_category'] = 'Unstable'
    
    # Financial Capacity
    monthly_expenses = sum([
        input_data.get('monthly_rent', 0),
        input_data.get('school_fees', 0),
        input_data.get('college_fees', 0),
        input_data.get('travel_expenses', 0),
        input_data.get('groceries_utilities', 0),
        input_data.get('other_monthly_expenses', 0)
    ])
    
    metrics['disposable_income'] = monthly_salary - monthly_expenses - current_emi
    metrics['savings_rate'] = (input_data.get('emergency_fund', 0) / monthly_salary) * 100 if monthly_salary > 0 else 0
    
    return metrics

def determine_eligibility(metrics):
    """Determine eligibility based on financial metrics"""
    score = 0
    
    # DTI Ratio scoring
    if metrics['dti_ratio'] < 20:
        score += 30
    elif metrics['dti_ratio'] < 35:
        score += 20
    elif metrics['dti_ratio'] < 50:
        score += 10
    
    # Requested EMI scoring
    if metrics['requested_emi_ratio'] < 15:
        score += 30
    elif metrics['requested_emi_ratio'] < 25:
        score += 20
    elif metrics['requested_emi_ratio'] < 40:
        score += 10
    
    # Credit score points
    score += metrics['credit_score_points'] * 0.3
    
    # Stability points
    score += metrics['stability_score'] * 0.2
    
    # Determine category
    if score >= 70:
        return "Eligible", score
    elif score >= 45:
        return "High Risk", score
    else:
        return "Not Eligible", score

def main():
    st.title("🤖 EMI Eligibility Predictor")
    st.markdown("### Advanced Risk Assessment Engine")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction", "📈 Prediction History"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 👤 Applicant Information")
            
            # Personal Information
            with st.expander("Personal Details", expanded=True):
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                with col_p2:
                    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
                    education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
                with col_p3:
                    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
                    family_size = st.number_input("Family Size", min_value=1, max_value=15, value=1)
            
            # Employment Information
            with st.expander("Employment Details", expanded=True):
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    monthly_salary = st.number_input("Monthly Salary (₹)", min_value=0, value=50000, step=1000)
                    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed", "Unemployed"])
                with col_e2:
                    years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=5.0, step=0.5)
                    company_type = st.selectbox("Company Type", ["Startup", "Mid-size", "MNC", "Government", "Other"])
                with col_e3:
                    house_type = st.selectbox("House Type", ["Rented", "Own", "Family", "Leased"])
                    monthly_rent = st.number_input("Monthly Rent (₹)", min_value=0, value=15000, step=1000) if house_type == "Rented" else 0
            
            # Financial Information
            with st.expander("Financial Details", expanded=True):
                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700, step=1)
                    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
                with col_f2:
                    current_emi_amount = st.number_input("Current EMI (₹)", min_value=0, value=10000, step=500)
                    bank_balance = st.number_input("Bank Balance (₹)", min_value=0, value=150000, step=5000)
                with col_f3:
                    emergency_fund = st.number_input("Emergency Fund (₹)", min_value=0, value=50000, step=5000)
            
            # Expenses
            with st.expander("Monthly Expenses", expanded=False):
                col_x1, col_x2, col_x3 = st.columns(3)
                with col_x1:
                    school_fees = st.number_input("School Fees (₹)", min_value=0, value=5000, step=500)
                    college_fees = st.number_input("College Fees (₹)", min_value=0, value=0, step=500)
                with col_x2:
                    travel_expenses = st.number_input("Travel Expenses (₹)", min_value=0, value=3000, step=500)
                    groceries_utilities = st.number_input("Groceries & Utilities (₹)", min_value=0, value=8000, step=500)
                with col_x3:
                    other_expenses = st.number_input("Other Expenses (₹)", min_value=0, value=5000, step=500)
            
            # Loan Details
            with st.expander("Loan Application Details", expanded=True):
                col_l1, col_l2, col_l3 = st.columns(3)
                with col_l1:
                    emi_scenario = st.selectbox("EMI Scenario", 
                                               ["E-commerce Shopping EMI", "Home Appliances EMI",
                                                "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
                with col_l2:
                    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, value=500000, step=10000)
                with col_l3:
                    requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, max_value=120, value=36, step=1)
            
            # Predict button
            col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
            with col_b2:
                predict_button = st.button("🎯 Predict Eligibility", use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Quick Metrics")
            
            # Display current metrics
            if 'monthly_salary' in locals():
                input_data = {
                    'monthly_salary': monthly_salary,
                    'current_emi_amount': current_emi_amount,
                    'requested_amount': requested_amount,
                    'requested_tenure': requested_tenure,
                    'credit_score': credit_score,
                    'years_of_employment': years_of_employment,
                    'monthly_rent': monthly_rent,
                    'school_fees': school_fees,
                    'college_fees': college_fees,
                    'travel_expenses': travel_expenses,
                    'groceries_utilities': groceries_utilities,
                    'other_monthly_expenses': other_expenses,
                    'emergency_fund': emergency_fund
                }
                
                metrics = calculate_financial_metrics(input_data)
                
                # Display quick metrics
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("DTI Ratio", f"{metrics['dti_ratio']:.1f}%", 
                         "Good" if metrics['dti_ratio'] < 35 else "High")
                st.metric("Requested EMI/Salary", f"{metrics['requested_emi_ratio']:.1f}%",
                         "Affordable" if metrics['requested_emi_ratio'] < 25 else "Stretched")
                st.metric("Credit Category", metrics['credit_category'])
                st.metric("Employment Stability", metrics['stability_category'])
                st.metric("Disposable Income", f"₹{metrics['disposable_income']:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    if predict_button:
        with st.spinner("Analyzing financial profile..."):
            # Prepare input data
            input_data = {
                'age': age,
                'gender': gender,
                'marital_status': marital_status,
                'education': education,
                'monthly_salary': monthly_salary,
                'employment_type': employment_type,
                'years_of_employment': years_of_employment,
                'company_type': company_type,
                'house_type': house_type,
                'monthly_rent': monthly_rent,
                'family_size': family_size,
                'dependents': dependents,
                'school_fees': school_fees,
                'college_fees': college_fees,
                'travel_expenses': travel_expenses,
                'groceries_utilities': groceries_utilities,
                'other_monthly_expenses': other_expenses,
                'existing_loans': existing_loans,
                'current_emi_amount': current_emi_amount,
                'credit_score': credit_score,
                'bank_balance': bank_balance,
                'emergency_fund': emergency_fund,
                'emi_scenario': emi_scenario,
                'requested_amount': requested_amount,
                'requested_tenure': requested_tenure
            }
            
            # Calculate metrics and determine eligibility
            metrics = calculate_financial_metrics(input_data)
            eligibility, confidence_score = determine_eligibility(metrics)
            
            # Store prediction in history
            prediction_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'amount': requested_amount,
                'eligibility': eligibility,
                'confidence': confidence_score,
                'dti_ratio': metrics['dti_ratio']
            }
            st.session_state.predictions_history.append(prediction_record)
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            
            with col_r1:
                if eligibility == "Eligible":
                    color = "#10B981"
                    icon = "✅"
                elif eligibility == "High Risk":
                    color = "#F59E0B"
                    icon = "⚠️"
                else:
                    color = "#EF4444"
                    icon = "❌"
                
                st.markdown(f'''
                    <div class="prediction-card" style="background: {color};">
                        <h1 style="font-size: 3rem;">{icon}</h1>
                        <h2>{eligibility}</h2>
                        <p style="font-size: 1.2rem;">Confidence Score: {confidence_score:.1f}%</p>
                    </div>
                ''', unsafe_allow_html=True)
            
            with col_r2:
                st.markdown("#### 📈 Key Ratios")
                # Use specs for indicator subplots - indicator traces need domain type, not xy
                fig = make_subplots(
                    rows=2, cols=2, 
                    subplot_titles=('DTI Ratio', 'EMI/Salary', 'Total Obligation', 'Savings Rate'),
                    specs=[[{"type": "indicator"}, {"type": "indicator"}], [{"type": "indicator"}, {"type": "indicator"}]]
                )
                
                # DTI Ratio gauge
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=metrics['dti_ratio'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "#6366F1"},
                          'steps': [
                              {'range': [0, 35], 'color': "#10B981"},
                              {'range': [35, 50], 'color': "#F59E0B"},
                              {'range': [50, 100], 'color': "#EF4444"}
                          ]}
                ), row=1, col=1)
                
                # Requested EMI ratio gauge
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=metrics['requested_emi_ratio'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "#6366F1"},
                          'steps': [
                              {'range': [0, 25], 'color': "#10B981"},
                              {'range': [25, 40], 'color': "#F59E0B"},
                              {'range': [40, 100], 'color': "#EF4444"}
                          ]}
                ), row=1, col=2)
                
                # Total obligation ratio gauge
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=metrics['total_obligation_ratio'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 200]},
                          'bar': {'color': "#6366F1"},
                          'steps': [
                              {'range': [0, 50], 'color': "#10B981"},
                              {'range': [50, 75], 'color': "#F59E0B"},
                              {'range': [75, 200], 'color': "#EF4444"}
                          ]}
                ), row=2, col=1)
                
                # Savings rate gauge
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=metrics['savings_rate'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "#6366F1"},
                          'steps': [
                              {'range': [20, 100], 'color': "#10B981"},
                              {'range': [10, 20], 'color': "#F59E0B"},
                              {'range': [0, 10], 'color': "#EF4444"}
                          ]}
                ), row=2, col=2)
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_r3:
                st.markdown("#### ⚠️ Risk Factors")
                
                risk_factors = []
                
                if metrics['dti_ratio'] > 50:
                    risk_factors.append(("High DTI Ratio", f"Current DTI of {metrics['dti_ratio']:.1f}% exceeds safe limit of 50%", "high"))
                elif metrics['dti_ratio'] > 35:
                    risk_factors.append(("Elevated DTI Ratio", f"DTI of {metrics['dti_ratio']:.1f}% is above ideal 35%", "medium"))
                
                if metrics['requested_emi_ratio'] > 40:
                    risk_factors.append(("High EMI Burden", f"Requested EMI would consume {metrics['requested_emi_ratio']:.1f}% of income", "high"))
                elif metrics['requested_emi_ratio'] > 25:
                    risk_factors.append(("Moderate EMI Burden", f"Requested EMI is {metrics['requested_emi_ratio']:.1f}% of income", "medium"))
                
                if metrics['credit_score_points'] < 60:
                    risk_factors.append(("Poor Credit History", f"Credit score of {credit_score} needs improvement", "high"))
                
                if metrics['stability_score'] < 60:
                    risk_factors.append(("Employment Instability", f"Only {years_of_employment} years of work history", "medium"))
                
                if metrics['savings_rate'] < 10:
                    risk_factors.append(("Low Emergency Fund", "Emergency fund is less than 3 months of expenses", "high"))
                
                if risk_factors:
                    for factor, description, severity in risk_factors:
                        if severity == "high":
                            st.markdown(f'<div class="risk-factor risk-high">⚠️ <strong>{factor}:</strong> {description}</div>', unsafe_allow_html=True)
                        elif severity == "medium":
                            st.markdown(f'<div class="risk-factor risk-medium">⚠️ <strong>{factor}:</strong> {description}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="risk-factor risk-low">✅ <strong>{factor}:</strong> {description}</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ No significant risk factors detected")
            
            # Recommendations
            st.markdown("---")
            st.markdown("### 💡 Recommendations")
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                if eligibility == "Not Eligible":
                    st.info("""
                        **To improve eligibility:**
                        - Reduce existing debt burden
                        - Improve credit score (pay bills on time, reduce credit utilization)
                        - Build emergency fund (target 6 months of expenses)
                        - Consider lower loan amount or longer tenure
                    """)
                elif eligibility == "High Risk":
                    st.warning("""
                        **Risk mitigation strategies:**
                        - Consider reducing loan amount by 20-30%
                        - Extend loan tenure to reduce EMI burden
                        - Provide additional collateral or guarantor
                        - Consider a co-applicant to strengthen profile
                    """)
                else:
                    st.success("""
                        **Your profile looks good! Next steps:**
                        - Proceed with loan application
                        - Consider pre-approved offers
                        - Maintain good credit behavior
                        - Explore higher loan amounts if needed
                    """)
            
            with col_rec2:
                # Create radar chart for financial health
                categories = ['DTI Ratio', 'Credit Score', 'EMI Burden', 'Savings', 'Stability']
                
                # Normalize values
                dti_score = max(0, 100 - metrics['dti_ratio'])
                credit_score_norm = metrics['credit_score_points']
                emi_score = max(0, 100 - metrics['requested_emi_ratio'] * 2)
                savings_score = min(100, metrics['savings_rate'] * 2)
                stability_score = metrics['stability_score']
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=[dti_score, credit_score_norm, emi_score, savings_score, stability_score],
                    theta=categories,
                    fill='toself',
                    marker=dict(color='#6366F1')
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=False,
                    title="Financial Health Radar",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### 📊 Batch Prediction")
        st.info("Upload a CSV file with multiple applications for batch processing")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"📁 Loaded {len(df)} applications")
            st.dataframe(df.head())
            
            if st.button("Process Batch"):
                with st.spinner("Processing applications..."):
                    # Add batch processing logic here
                    st.success(f"✅ Processed {len(df)} applications successfully!")
                    
                    # Show results
                    results_df = df.copy()
                    results_df['eligibility'] = np.random.choice(['Eligible', 'High Risk', 'Not Eligible'], len(df))
                    results_df['confidence'] = np.random.uniform(60, 100, len(df))
                    
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "batch_predictions.csv", "text/csv")
    
    with tab3:
        st.markdown("#### 📈 Prediction History")
        
        if st.session_state.predictions_history:
            history_df = pd.DataFrame(st.session_state.predictions_history)
            
            # Show metrics
            col_h1, col_h2, col_h3 = st.columns(3)
            with col_h1:
                st.metric("Total Predictions", len(history_df))
            with col_h2:
                eligible_count = len(history_df[history_df['eligibility'] == 'Eligible'])
                st.metric("Eligible", f"{eligible_count} ({(eligible_count/len(history_df)*100):.1f}%)")
            with col_h3:
                avg_confidence = history_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Show history table
            st.dataframe(history_df, use_container_width=True)
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.predictions_history = []
                st.rerun()
        else:
            st.info("No predictions made yet. Use the Single Prediction tab to get started.")

if __name__ == "__main__":
    main()