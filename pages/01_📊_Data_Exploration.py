import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

st.title("📊 Data Exploration Dashboard")
st.markdown("Explore the financial dataset with interactive visualizations")

# Check if data exists in session state
if 'data' not in st.session_state:
    st.session_state.data = None

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # Load sample data
    if st.button("Load Sample Data", use_container_width=True):
        # Create larger sample dataset
        import numpy as np
        np.random.seed(42)
        n_samples = 50000
        
        sample_data = pd.DataFrame({
            'age': np.random.randint(22, 65, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'education': np.random.choice(['High School', 'Graduate', 'Post Graduate', 'Professional'], n_samples),
            'monthly_salary': np.random.randint(15000, 300000, n_samples),
            'employment_type': np.random.choice(['Private', 'Government', 'Self-employed', 'Unemployed'], n_samples),
            'years_of_employment': np.random.uniform(0, 35, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'existing_loans': np.random.choice(['Yes', 'No'], n_samples),
            'current_emi_amount': np.random.randint(0, 80000, n_samples),
            'emi_scenario': np.random.choice(['E-commerce Shopping EMI', 'Home Appliances EMI', 
                                             'Vehicle EMI', 'Personal Loan EMI', 'Education EMI'], n_samples),
            'requested_amount': np.random.randint(10000, 2000000, n_samples),
            'requested_tenure': np.random.randint(3, 84, n_samples),
            'emi_eligibility': np.random.choice(['Eligible', 'High Risk', 'Not Eligible'], n_samples, 
                                               p=[0.6, 0.25, 0.15]),
            'max_monthly_emi': np.random.randint(500, 120000, n_samples)
        })
        
        st.session_state.data = sample_data
        st.success(f"✅ Loaded {n_samples:,} records!")

if st.session_state.data is not None:
    df = st.session_state.data
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.1f} MB")
    
    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_scenarios = st.multiselect("EMI Scenario", 
                                           df['emi_scenario'].unique(),
                                           default=df['emi_scenario'].unique()[:2])
    with col2:
        selected_eligibility = st.multiselect("Eligibility Status",
                                             df['emi_eligibility'].unique(),
                                             default=df['emi_eligibility'].unique())
    with col3:
        salary_range = st.slider("Monthly Salary Range",
                                float(df['monthly_salary'].min()),
                                float(df['monthly_salary'].max()),
                                (float(df['monthly_salary'].min()), 
                                 float(df['monthly_salary'].max())))
    
    # Apply filters
    filtered_df = df[
        (df['emi_scenario'].isin(selected_scenarios)) &
        (df['emi_eligibility'].isin(selected_eligibility)) &
        (df['monthly_salary'].between(salary_range[0], salary_range[1]))
    ]
    
    st.info(f"Showing {len(filtered_df):,} records after filtering")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distributions", "🔗 Relationships", "📊 Statistical Summary", "📋 Raw Data"])
    
    with tab1:
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Numerical distributions
            num_col = st.selectbox("Select Numerical Feature", 
                                   df.select_dtypes(include=[np.number]).columns)
            fig = px.histogram(filtered_df, x=num_col, nbins=50, 
                              title=f'Distribution of {num_col}',
                              color_discrete_sequence=['#6366F1'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Categorical distributions
                        # Categorical distributions
            cat_col = st.selectbox("Select Categorical Feature",
                                   df.select_dtypes(include=['object']).columns)
            
            # FIX: Properly handle value counts for bar chart
            value_counts = filtered_df[cat_col].value_counts().reset_index()
            value_counts.columns = [cat_col, 'count']
            
            fig = px.bar(value_counts, 
                        x=cat_col, 
                        y='count',
                        title=f'Distribution of {cat_col}',
                        color_discrete_sequence=['#10B981'])
            st.plotly_chart(fig, use_container_width=True)
            
    
    with tab2:
        st.subheader("Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            x_col = st.selectbox("X-axis", df.select_dtypes(include=[np.number]).columns, index=0)
            y_col = st.selectbox("Y-axis", df.select_dtypes(include=[np.number]).columns, index=4)
            color_col = st.selectbox("Color by", ['None'] + list(df.select_dtypes(include=['object']).columns))
            
            if color_col == 'None':
                fig = px.scatter(filtered_df.sample(min(1000, len(filtered_df))), 
                               x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
            else:
                fig = px.scatter(filtered_df.sample(min(1000, len(filtered_df))), 
                               x=x_col, y=y_col, color=color_col, 
                               title=f'{x_col} vs {y_col} colored by {color_col}')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation heatmap
            numeric_df = filtered_df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Correlation Heatmap",
                           color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Statistical Summary")
        
        # Numerical features statistics
        st.write("**Numerical Features**")
        st.dataframe(filtered_df.describe(), use_container_width=True)
        
        # Categorical features statistics
        st.write("**Categorical Features**")
        cat_stats = []
        for col in filtered_df.select_dtypes(include=['object']).columns:
            cat_stats.append({
                'Feature': col,
                'Unique Values': filtered_df[col].nunique(),
                'Most Common': filtered_df[col].mode()[0] if not filtered_df[col].mode().empty else 'N/A',
                'Frequency': filtered_df[col].value_counts().iloc[0] if not filtered_df[col].value_counts().empty else 0
            })
        st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)
    
    with tab4:
        st.subheader("Raw Data")
        
        # Row selection
        rows_to_show = st.slider("Rows to display", 10, 1000, 100)
        st.dataframe(filtered_df.head(rows_to_show), use_container_width=True)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button("📥 Download Filtered Data", csv, "filtered_data.csv", "text/csv")

else:
    st.info("👈 Please load sample data from the sidebar to begin exploration")