"""
Maximum EMI Calculator Page
Professional implementation for calculating safe EMI limits
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Max EMI Calculator",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .calculator-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
    }
    .calculation-step {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #6366F1;
    }
    </style>
""", unsafe_allow_html=True)

def calculate_max_emi(input_data):
    """
    Calculate maximum safe EMI based on financial profile
    Using professional lending standards
    """
    monthly_salary = input_data['monthly_salary']
    monthly_expenses = input_data['monthly_expenses']
    current_emi = input_data.get('current_emi', 0)
    credit_score = input_data['credit_score']
    dependents = input_data.get('dependents', 0)
    employment_type = input_data['employment_type']
    years_employed = input_data.get('years_employed', 0)
    existing_loans = input_data.get('existing_loans', 'No')
    emergency_fund = input_data.get('emergency_fund', 0)
    
    # Step 1: Calculate disposable income
    disposable_income = monthly_salary - monthly_expenses - current_emi
    
    # Step 2: Base EMI capacity (40% of disposable income)
    base_emi_capacity = disposable_income * 0.4
    
    # Step 3: Apply credit score multiplier
    if credit_score >= 750:
        credit_multiplier = 1.2
        credit_category = "Excellent"
    elif credit_score >= 700:
        credit_multiplier = 1.1
        credit_category = "Good"
    elif credit_score >= 650:
        credit_multiplier = 1.0
        credit_category = "Fair"
    elif credit_score >= 600:
        credit_multiplier = 0.8
        credit_category = "Poor"
    else:
        credit_multiplier = 0.6
        credit_category = "Very Poor"
    
    # Step 4: Apply employment multiplier
    if employment_type == "Government":
        emp_multiplier = 1.2
    elif employment_type == "Private":
        if years_employed >= 5:
            emp_multiplier = 1.1
        elif years_employed >= 2:
            emp_multiplier = 1.0
        else:
            emp_multiplier = 0.8
    elif employment_type == "Self-employed":
        if years_employed >= 5:
            emp_multiplier = 1.0
        else:
            emp_multiplier = 0.7
    else:
        emp_multiplier = 0.5
    
    # Step 5: Apply dependents multiplier
    if dependents == 0:
        dependent_multiplier = 1.1
    elif dependents <= 2:
        dependent_multiplier = 1.0
    elif dependents <= 4:
        dependent_multiplier = 0.9
    else:
        dependent_multiplier = 0.8
    
    # Step 6: Apply existing loans penalty
    if existing_loans == "Yes":
        loan_multiplier = 0.8
    else:
        loan_multiplier = 1.0
    
    # Step 7: Apply emergency fund bonus
    required_emergency_fund = monthly_expenses * 6  # 6 months of expenses
    if emergency_fund >= required_emergency_fund:
        emergency_multiplier = 1.1
    elif emergency_fund >= required_emergency_fund * 0.5:
        emergency_multiplier = 1.05
    else:
        emergency_multiplier = 1.0
    
    # Calculate final EMI capacity
    max_emi = base_emi_capacity * credit_multiplier * emp_multiplier * dependent_multiplier * loan_multiplier * emergency_multiplier
    
    # Apply absolute limits
    max_emi = max(0, min(max_emi, monthly_salary * 0.5))  # Cannot exceed 50% of salary
    
    # Calculate recommended loan amount for different tenures
    loan_amounts = {}
    for tenure in [12, 24, 36, 48, 60, 84]:
        # Simple interest calculation (assuming 10% interest rate)
        monthly_rate = 0.10 / 12
        if monthly_rate > 0:
            loan_amount = max_emi * ((1 - (1 + monthly_rate) ** -tenure) / monthly_rate)
        else:
            loan_amount = max_emi * tenure
        loan_amounts[f"{tenure} months"] = round(loan_amount, -3)
    
    return {
        'max_emi': round(max_emi, -2),
        'disposable_income': disposable_income,
        'base_capacity': base_emi_capacity,
        'credit_multiplier': credit_multiplier,
        'credit_category': credit_category,
        'emp_multiplier': emp_multiplier,
        'dependent_multiplier': dependent_multiplier,
        'loan_multiplier': loan_multiplier,
        'emergency_multiplier': emergency_multiplier,
        'loan_amounts': loan_amounts,
        'emi_to_income_ratio': (max_emi / monthly_salary * 100) if monthly_salary > 0 else 0
    }

def main():
    st.title("💰 Maximum EMI Calculator")
    st.markdown("### Calculate Your Safe EMI Limit")
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📋 Financial Profile")
        
        # Income Section
        with st.expander("Income Details", expanded=True):
            monthly_salary = st.number_input("Monthly Salary (₹)", 
                                            min_value=0, value=75000, step=5000,
                                            help="Your gross monthly income")
            
            employment_type = st.selectbox("Employment Type", 
                                          ["Private", "Government", "Self-employed", "Business", "Other"],
                                          help="Type of employment")
            
            years_employed = st.number_input("Years in Current Employment", 
                                            min_value=0.0, value=5.0, step=0.5,
                                            help="Number of years in current job")
        
        # Expenses Section
        with st.expander("Monthly Expenses", expanded=True):
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                housing_expense = st.number_input("Housing (Rent/Mortgage) (₹)", 
                                                 min_value=0, value=20000, step=1000)
                utilities = st.number_input("Utilities (₹)", 
                                           min_value=0, value=5000, step=500)
                food_expense = st.number_input("Food & Groceries (₹)", 
                                              min_value=0, value=10000, step=1000)
            
            with col_exp2:
                transport_expense = st.number_input("Transportation (₹)", 
                                                   min_value=0, value=3000, step=500)
                insurance_expense = st.number_input("Insurance (₹)", 
                                                   min_value=0, value=2000, step=500)
                other_expenses = st.number_input("Other Expenses (₹)", 
                                                min_value=0, value=5000, step=500)
        
        # Liabilities Section
        with st.expander("Liabilities & Savings", expanded=True):
            col_lia1, col_lia2 = st.columns(2)
            
            with col_lia1:
                current_emi = st.number_input("Existing EMI Payments (₹)", 
                                             min_value=0, value=15000, step=1000,
                                             help="Total of all current EMIs")
                
                existing_loans = st.selectbox("Have Other Loans?", 
                                             ["Yes", "No"],
                                             help="Do you have any other active loans?")
            
            with col_lia2:
                credit_score = st.number_input("Credit Score", 
                                              min_value=300, max_value=850, value=720, step=5,
                                              help="Your CIBIL or equivalent credit score")
                
                emergency_fund = st.number_input("Emergency Fund (₹)", 
                                                min_value=0, value=100000, step=10000,
                                                help="Total savings for emergencies")
        
        # Family Section
        with st.expander("Family Details", expanded=False):
            col_fam1, col_fam2 = st.columns(2)
            
            with col_fam1:
                dependents = st.number_input("Number of Dependents", 
                                            min_value=0, max_value=10, value=2,
                                            help="People financially dependent on you")
            
            with col_fam2:
                marital_status = st.selectbox("Marital Status", 
                                             ["Single", "Married", "Divorced", "Widowed"])
    
    with col2:
        st.markdown("#### 💡 EMI Calculation")
        
        # Calculate button
        if st.button("Calculate Maximum EMI", use_container_width=True):
            with st.spinner("Analyzing your financial profile..."):
                
                # Calculate total monthly expenses
                monthly_expenses = housing_expense + utilities + food_expense + transport_expense + insurance_expense + other_expenses
                
                input_data = {
                    'monthly_salary': monthly_salary,
                    'monthly_expenses': monthly_expenses,
                    'current_emi': current_emi,
                    'credit_score': credit_score,
                    'dependents': dependents,
                    'employment_type': employment_type,
                    'years_employed': years_employed,
                    'existing_loans': existing_loans,
                    'emergency_fund': emergency_fund
                }
                
                result = calculate_max_emi(input_data)
                
                # Display result card
                st.markdown(f'''
                    <div class="calculator-card">
                        <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem;">Your Maximum Safe EMI</h2>
                        <h1 style="font-size: 3.5rem; margin: 0;">₹{result['max_emi']:,.0f}</h1>
                        <p style="font-size: 1.1rem; margin-top: 0.5rem;">per month</p>
                        <p style="font-size: 0.9rem;">EMI to Income Ratio: {result['emi_to_income_ratio']:.1f}%</p>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Calculation steps
                st.markdown("#### 📊 Calculation Breakdown")
                
                st.markdown(f'''
                    <div class="calculation-step">
                        <strong>Step 1: Disposable Income</strong><br>
                        Monthly Salary: ₹{monthly_salary:,.0f}<br>
                        Monthly Expenses: ₹{monthly_expenses:,.0f}<br>
                        Current EMI: ₹{current_emi:,.0f}<br>
                        <strong>Disposable Income: ₹{result['disposable_income']:,.0f}</strong>
                    </div>
                    
                    <div class="calculation-step">
                        <strong>Step 2: Base EMI Capacity</strong><br>
                        40% of Disposable Income<br>
                        <strong>Base Capacity: ₹{result['base_capacity']:,.0f}</strong>
                    </div>
                    
                    <div class="calculation-step">
                        <strong>Step 3: Risk Multipliers</strong><br>
                        Credit Score ({input_data['credit_score']}): {result['credit_multiplier']:.1f}x ({result['credit_category']})<br>
                        Employment: {result['emp_multiplier']:.1f}x<br>
                        Dependents ({dependents}): {result['dependent_multiplier']:.1f}x<br>
                        Existing Loans: {result['loan_multiplier']:.1f}x<br>
                        Emergency Fund: {result['emergency_multiplier']:.1f}x<br>
                        <strong>Total Multiplier: {(result['credit_multiplier'] * result['emp_multiplier'] * result['dependent_multiplier'] * result['loan_multiplier'] * result['emergency_multiplier']):.2f}x</strong>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Loan amount calculator
                st.markdown("#### 🏦 Recommended Loan Amounts")
                st.markdown("Based on your EMI capacity, here are recommended loan amounts for different tenures (at 10% interest):")
                
                loan_df = pd.DataFrame([
                    {"Tenure": tenure, "Max Loan Amount": f"₹{amount:,.0f}"}
                    for tenure, amount in result['loan_amounts'].items()
                ])
                st.table(loan_df)
                
                # Visualization
                st.markdown("#### 📈 Financial Health Dashboard")
                
                # Create gauges for key metrics
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('EMI to Income Ratio', 'Credit Health', 'Debt Burden', 'Emergency Fund'),
                    specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                           [{'type': 'indicator'}, {'type': 'indicator'}]]
                )
                
                # EMI to Income Ratio
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=result['emi_to_income_ratio'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 60]},
                        'bar': {'color': "#10B981"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ECFDF5"},
                            {'range': [30, 45], 'color': "#FEF3C7"},
                            {'range': [45, 60], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 40
                        }
                    }
                ), row=1, col=1)
                
                # Credit Health
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=input_data['credit_score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [300, 850]},
                        'bar': {'color': "#6366F1"},
                        'steps': [
                            {'range': [300, 600], 'color': "#FEE2E2"},
                            {'range': [600, 700], 'color': "#FEF3C7"},
                            {'range': [700, 750], 'color': "#D1FAE5"},
                            {'range': [750, 850], 'color': "#A7F3D0"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 750
                        }
                    }
                ), row=1, col=2)
                
                # Debt Burden
                debt_ratio = (current_emi / monthly_salary * 100) if monthly_salary > 0 else 0
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=debt_ratio,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#F59E0B"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ECFDF5"},
                            {'range': [30, 50], 'color': "#FEF3C7"},
                            {'range': [50, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ), row=2, col=1)
                
                # Emergency Fund
                required_emergency = monthly_expenses * 6
                emergency_ratio = (emergency_fund / required_emergency * 100) if required_emergency > 0 else 0
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=min(emergency_ratio, 200),
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 200]},
                        'bar': {'color': "#10B981"},
                        'steps': [
                            {'range': [0, 50], 'color': "#FEE2E2"},
                            {'range': [50, 100], 'color': "#FEF3C7"},
                            {'range': [100, 200], 'color': "#ECFDF5"}
                        ],
                        'threshold': {
                            'line': {'color': "green", 'width': 4},
                            'thickness': 0.75,
                            'value': 100
                        }
                    }
                ), row=2, col=2)
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("#### 💡 Recommendations")
                
                if result['emi_to_income_ratio'] > 40:
                    st.warning("⚠️ Your EMI would exceed 40% of your income. Consider reducing the loan amount or extending the tenure.")
                
                if input_data['credit_score'] < 700:
                    st.warning("📉 Your credit score could be improved. Consider building credit before taking large loans.")
                
                if emergency_ratio < 100:
                    st.warning(f"💰 Your emergency fund covers only {emergency_ratio:.0f}% of recommended 6 months. Build this to 100% for better financial security.")
                
                if debt_ratio > 50:
                    st.error("🚨 Your current debt burden is very high. Consider reducing existing debt before taking new loans.")
                
                if result['emi_to_income_ratio'] <= 30 and input_data['credit_score'] >= 700 and emergency_ratio >= 100:
                    st.success("✅ Your financial profile is excellent! You're in a good position to take this loan.")

if __name__ == "__main__":
    main()