"""
Data Management Page
Complete CRUD operations for financial data management
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import io
from datetime import datetime
import sys
import os

# Helper function to ensure dataframe is Arrow-compatible
def prepare_df_for_display(df):
    """Convert dataframe to Arrow-compatible format"""
    df_display = df.copy()
    for col in df_display.select_dtypes(include=['number']).columns:
        df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
    for col in df_display.select_dtypes(include=['object']).columns:
        df_display[col] = df_display[col].astype(str)
    return df_display

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Data Management",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .data-card {
        background-color: #F9FAFB;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        margin: 0.5rem 0;
    }
    .stat-box {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .upload-area {
        border: 2px dashed #6366F1;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background-color: #F9FAFB;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for data management
if 'data_management' not in st.session_state:
    st.session_state.data_management = {
        'current_dataset': None,
        'dataset_name': None,
        'upload_history': [],
        'transformations': [],
        'backups': []
    }

def generate_sample_data(n_rows=1000):
    """Generate sample financial data"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(22, 65, n_rows),
        'gender': np.random.choice(['Male', 'Female'], n_rows),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_rows),
        'education': np.random.choice(['High School', 'Graduate', 'Post Graduate', 'Professional'], n_rows),
        'monthly_salary': np.random.randint(15000, 300000, n_rows),
        'employment_type': np.random.choice(['Private', 'Government', 'Self-employed', 'Unemployed'], n_rows),
        'years_of_employment': np.random.uniform(0, 35, n_rows).round(1),
        'company_type': np.random.choice(['Startup', 'Mid-size', 'MNC', 'Government', 'Other'], n_rows),
        'house_type': np.random.choice(['Rented', 'Own', 'Family', 'Leased'], n_rows),
        'monthly_rent': np.random.randint(0, 50000, n_rows),
        'family_size': np.random.randint(1, 8, n_rows),
        'dependents': np.random.randint(0, 5, n_rows),
        'school_fees': np.random.randint(0, 20000, n_rows),
        'college_fees': np.random.randint(0, 50000, n_rows),
        'travel_expenses': np.random.randint(0, 15000, n_rows),
        'groceries_utilities': np.random.randint(5000, 30000, n_rows),
        'other_monthly_expenses': np.random.randint(0, 20000, n_rows),
        'existing_loans': np.random.choice(['Yes', 'No'], n_rows),
        'current_emi_amount': np.random.randint(0, 80000, n_rows),
        'credit_score': np.random.randint(300, 850, n_rows),
        'bank_balance': np.random.randint(0, 1000000, n_rows),
        'emergency_fund': np.random.randint(0, 500000, n_rows),
        'emi_scenario': np.random.choice(['E-commerce Shopping EMI', 'Home Appliances EMI', 
                                         'Vehicle EMI', 'Personal Loan EMI', 'Education EMI'], n_rows),
        'requested_amount': np.random.randint(10000, 2000000, n_rows),
        'requested_tenure': np.random.randint(3, 84, n_rows)
    }
    
    # Add target variables
    data['emi_eligibility'] = np.random.choice(['Eligible', 'High Risk', 'Not Eligible'], n_rows, p=[0.6, 0.25, 0.15])
    data['max_monthly_emi'] = np.random.randint(500, 120000, n_rows)
    
    return pd.DataFrame(data)

def validate_data(df):
    """Validate data quality with proper type handling"""
    issues = []
    
    # Make a copy to avoid modifying original
    df_validation = df.copy()
    
    # Convert numeric columns that might be stored as strings
    numeric_candidates = ['age', 'monthly_salary', 'years_of_employment', 'credit_score', 
                         'current_emi_amount', 'requested_amount', 'requested_tenure',
                         'max_monthly_emi', 'monthly_rent', 'family_size', 'dependents',
                         'school_fees', 'college_fees', 'travel_expenses', 
                         'groceries_utilities', 'other_monthly_expenses', 'bank_balance',
                         'emergency_fund']
    
    for col in numeric_candidates:
        if col in df_validation.columns:
            try:
                df_validation[col] = pd.to_numeric(df_validation[col], errors='coerce')
            except:
                pass
    
    # Check for missing values
    missing_cols = df_validation.columns[df_validation.isnull().any()].tolist()
    if missing_cols:
        issues.append(f"⚠️ Missing values in columns: {', '.join(missing_cols)}")
    
    # Check for negative values in numeric columns
    for col in numeric_candidates:
        if col in df_validation.columns and pd.api.types.is_numeric_dtype(df_validation[col]):
            if (df_validation[col] < 0).any():
                issues.append(f"⚠️ Negative values found in: {col}")
    
    # Check data types for critical columns
    if 'age' in df_validation.columns:
        if pd.api.types.is_numeric_dtype(df_validation['age']):
            if (df_validation['age'] < 18).any() or (df_validation['age'] > 100).any():
                issues.append("⚠️ Age values outside reasonable range (18-100)")
        else:
            issues.append("⚠️ Age column is not numeric")
    
    if 'credit_score' in df_validation.columns:
        if pd.api.types.is_numeric_dtype(df_validation['credit_score']):
            if (df_validation['credit_score'] < 300).any() or (df_validation['credit_score'] > 850).any():
                issues.append("⚠️ Credit score values outside range (300-850)")
        else:
            issues.append("⚠️ Credit score column is not numeric")
    
    if 'monthly_salary' in df_validation.columns:
        if pd.api.types.is_numeric_dtype(df_validation['monthly_salary']):
            if (df_validation['monthly_salary'] < 0).any():
                issues.append("⚠️ Negative salary values found")
    
    return issues

def get_data_summary(df):
    """Get comprehensive data summary"""
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage': f"{df.memory_usage().sum() / 1024**2:.2f} MB",
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=['object']).columns)
    }
    return summary

def main():
    st.title("⚙️ Data Management")
    st.markdown("### Complete Data Lifecycle Management")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload Data",
        "🔄 Transform Data",
        "📊 Data Quality",
        "📥 Export Data",
        "📋 Data Dictionary"
    ])
    
    with tab1:
        st.subheader("Upload Dataset")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            st.markdown("### 📁 Drag and drop or browse files")
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded successfully: {uploaded_file.name}")
                    st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    
                    # Store in session state
                    st.session_state.data_management['current_dataset'] = df
                    st.session_state.data_management['dataset_name'] = uploaded_file.name
                    
                    # Add to history
                    st.session_state.data_management['upload_history'].append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': uploaded_file.name,
                        'rows': df.shape[0],
                        'columns': df.shape[1]
                    })
                    
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Generate Sample Data")
            st.markdown("Create a sample dataset for testing")
            
            n_samples = st.slider("Number of samples", 100, 10000, 1000, step=100)
            
            if st.button("🎲 Generate Sample", use_container_width=True):
                sample_df = generate_sample_data(n_samples)
                st.session_state.data_management['current_dataset'] = sample_df
                st.session_state.data_management['dataset_name'] = f"sample_data_{n_samples}.csv"
                st.success(f"✅ Generated {n_samples} sample records!")
                st.rerun()
        
        # Show upload history
        if st.session_state.data_management['upload_history']:
            st.subheader("📋 Upload History")
            history_df = pd.DataFrame(st.session_state.data_management['upload_history'])
            st.dataframe(history_df, use_container_width=True)
    
    with tab2:
        st.subheader("Data Transformations")
        
        if st.session_state.data_management['current_dataset'] is not None:
            df = st.session_state.data_management['current_dataset'].copy()
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Available Transformations")
                
                transformation = st.selectbox(
                    "Select Transformation",
                    [
                        "Handle Missing Values",
                        "Remove Duplicates",
                        "Remove Outliers",
                        "Normalize/Scale Data",
                        "Encode Categorical Variables",
                        "Create Derived Features",
                        "Filter Rows",
                        "Select Columns"
                    ]
                )
                
                if transformation == "Handle Missing Values":
                    method = st.radio("Method", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
                    columns = st.multiselect("Apply to columns", df.columns)
                    
                    if st.button("Apply", use_container_width=True):
                        with st.spinner("Processing..."):
                            if method == "Drop rows":
                                df_clean = df.dropna(subset=columns if columns else None)
                            elif method == "Fill with mean":
                                for col in columns or df.select_dtypes(include=[np.number]).columns:
                                    df[col].fillna(df[col].mean(), inplace=True)
                                df_clean = df
                            elif method == "Fill with median":
                                for col in columns or df.select_dtypes(include=[np.number]).columns:
                                    df[col].fillna(df[col].median(), inplace=True)
                                df_clean = df
                            else:  # mode
                                for col in columns or df.select_dtypes(include=['object']).columns:
                                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                                df_clean = df
                            
                            st.session_state.data_management['current_dataset'] = df_clean
                            st.success("✅ Transformation applied!")
                
                elif transformation == "Remove Duplicates":
                    subset = st.multiselect("Consider these columns for duplicates (optional)", df.columns)
                    keep = st.radio("Keep", ["First", "Last", "None"])
                    
                    if st.button("Apply", use_container_width=True):
                        with st.spinner("Processing..."):
                            keep_map = {"First": "first", "Last": "last", "None": False}
                            before = len(df)
                            df_clean = df.drop_duplicates(subset=subset if subset else None, keep=keep_map[keep])
                            after = len(df_clean)
                            
                            st.session_state.data_management['current_dataset'] = df_clean
                            st.success(f"✅ Removed {before - after} duplicates!")
                
                elif transformation == "Remove Outliers":
                    method = st.radio("Method", ["IQR", "Z-Score"])
                    threshold = st.slider("Threshold", 1.0, 5.0, 3.0)
                    columns = st.multiselect("Apply to columns", df.select_dtypes(include=[np.number]).columns)
                    
                    if st.button("Apply", use_container_width=True):
                        with st.spinner("Processing..."):
                            df_clean = df.copy()
                            outliers_removed = 0
                            
                            for col in columns:
                                if method == "IQR":
                                    Q1 = df[col].quantile(0.25)
                                    Q3 = df[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - threshold * IQR
                                    upper_bound = Q3 + threshold * IQR
                                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                                    outliers_removed += len(outliers)
                                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                                else:  # Z-Score
                                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                                    outliers = df[z_scores > threshold]
                                    outliers_removed += len(outliers)
                                    df_clean = df_clean[z_scores <= threshold]
                            
                            st.session_state.data_management['current_dataset'] = df_clean
                            st.success(f"✅ Removed approximately {outliers_removed} outliers!")
            
            with col2:
                st.markdown("#### Data Preview")
                st.dataframe(df.head(100), use_container_width=True)
                
                # Show transformation history
                if st.session_state.data_management['transformations']:
                    st.markdown("#### Transformation History")
                    trans_df = pd.DataFrame(st.session_state.data_management['transformations'])
                    st.dataframe(trans_df, use_container_width=True)
        
        else:
            st.info("Please upload a dataset first")
    
    with tab3:
        st.subheader("Data Quality Assessment")
        
        if st.session_state.data_management['current_dataset'] is not None:
            df = st.session_state.data_management['current_dataset']
            
            # Data summary
            summary = get_data_summary(df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.metric("Total Rows", f"{summary['total_rows']:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.metric("Total Columns", summary['total_columns'])
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.metric("Missing Values", f"{summary['missing_values']:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="stat-box">', unsafe_allow_html=True)
                st.metric("Duplicates", f"{summary['duplicates']:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Data validation
            st.subheader("🔍 Data Validation")
            issues = validate_data(df)
            
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("✅ No data quality issues detected!")
            
            # Column details
            st.subheader("📋 Column Details")
            
            col_info = []
            for col in df.columns:
                col_info.append({
                    'Column': str(col),
                    'Type': str(df[col].dtype),
                    'Non-Null Count': int(df[col].count()),
                    'Null Count': int(df[col].isnull().sum()),
                    'Unique Values': int(df[col].nunique()),
                    'Sample Values': ', '.join(str(x) for x in df[col].dropna().unique()[:3])
                })
            
            col_df = pd.DataFrame(col_info)
            # Ensure Arrow-compatible types
            col_df['Non-Null Count'] = col_df['Non-Null Count'].astype('int64')
            col_df['Null Count'] = col_df['Null Count'].astype('int64')
            col_df['Unique Values'] = col_df['Unique Values'].astype('int64')
            st.dataframe(col_df, use_container_width=True)
            
            # Data statistics
            st.subheader("📊 Statistical Summary")
            
            tab_stats1, tab_stats2 = st.tabs(["Numerical Features", "Categorical Features"])
            
            with tab_stats1:
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    st.dataframe(numeric_df.describe(), use_container_width=True)
                else:
                    st.info("No numerical features found")
            
            with tab_stats2:
                categorical_df = df.select_dtypes(include=['object'])
                if not categorical_df.empty:
                    stats = []
                    for col in categorical_df.columns:
                        value_counts = df[col].value_counts()
                        stats.append({
                            'Column': col,
                            'Most Common': value_counts.index[0] if not value_counts.empty else 'N/A',
                            'Frequency': value_counts.iloc[0] if not value_counts.empty else 0,
                            'Unique Values': len(value_counts)
                        })
                    st.dataframe(pd.DataFrame(stats), use_container_width=True)
                else:
                    st.info("No categorical features found")
        
        else:
            st.info("Please upload a dataset first")
    
    with tab4:
        st.subheader("Export Data")
        
        if st.session_state.data_management['current_dataset'] is not None:
            df = st.session_state.data_management['current_dataset']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Export Format")
                
                export_format = st.radio(
                    "Select format",
                    ["CSV", "Excel", "JSON", "Parquet"]
                )
                
                include_index = st.checkbox("Include index", value=False)
                
                if export_format == "CSV":
                    csv_data = df.to_csv(index=include_index)
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                elif export_format == "Excel":
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=include_index, sheet_name='Data')
                    
                    st.download_button(
                        "📥 Download Excel",
                        output.getvalue(),
                        f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                elif export_format == "JSON":
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        json_data,
                        f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        use_container_width=True
                    )
                
                else:  # Parquet
                    parquet_data = df.to_parquet(index=include_index)
                    st.download_button(
                        "📥 Download Parquet",
                        parquet_data,
                        f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                        "application/octet-stream",
                        use_container_width=True
                    )
            
            with col2:
                st.markdown("#### Export Options")
                
                st.markdown(f"""
                **Dataset Info:**
                - Rows: {df.shape[0]:,}
                - Columns: {df.shape[1]}
                - Size: {df.memory_usage().sum() / 1024**2:.2f} MB
                """)
                
                # Create backup
                if st.button("💾 Create Backup", use_container_width=True):
                    backup = df.copy()
                    st.session_state.data_management['backups'].append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'data': backup,
                        'rows': len(backup),
                        'columns': len(backup.columns)
                    })
                    st.success("✅ Backup created!")
                
                # Show backups
                if st.session_state.data_management['backups']:
                    st.markdown("#### Available Backups")
                    for i, backup in enumerate(st.session_state.data_management['backups'][-5:]):
                        st.caption(f"{backup['timestamp']} - {backup['rows']} rows, {backup['columns']} cols")
        
        else:
            st.info("Please upload a dataset first")
    
    with tab5:
        st.subheader("Data Dictionary")
        
        # Define data dictionary
        data_dict = {
            'Feature': [
                'age', 'gender', 'marital_status', 'education', 'monthly_salary',
                'employment_type', 'years_of_employment', 'company_type', 'house_type',
                'monthly_rent', 'family_size', 'dependents', 'school_fees', 'college_fees',
                'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
                'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
                'emergency_fund', 'emi_scenario', 'requested_amount', 'requested_tenure',
                'emi_eligibility', 'max_monthly_emi'
            ],
            'Type': [
                'Integer', 'Categorical', 'Categorical', 'Categorical', 'Integer',
                'Categorical', 'Float', 'Categorical', 'Categorical',
                'Integer', 'Integer', 'Integer', 'Integer', 'Integer',
                'Integer', 'Integer', 'Integer',
                'Categorical', 'Integer', 'Integer', 'Integer',
                'Integer', 'Categorical', 'Integer', 'Integer',
                'Categorical', 'Integer'
            ],
            'Description': [
                'Age of the applicant',
                'Gender of the applicant',
                'Marital status',
                'Highest education level',
                'Monthly gross salary',
                'Type of employment',
                'Years in current employment',
                'Type/size of company',
                'Type of housing',
                'Monthly rent payment',
                'Total family size',
                'Number of dependents',
                'Monthly school fees',
                'Monthly college fees',
                'Monthly travel expenses',
                'Monthly groceries and utilities',
                'Other monthly expenses',
                'Has existing loans',
                'Current EMI amount',
                'Credit score (300-850)',
                'Current bank balance',
                'Emergency fund amount',
                'Type of EMI scenario',
                'Requested loan amount',
                'Requested loan tenure',
                'Eligibility status (Target)',
                'Maximum safe EMI (Target)'
            ],
            'Range/Values': [
                '18-80', 'Male/Female/Other', 'Single/Married/Divorced', 'High School/Graduate/Post Graduate/Professional',
                '15,000-300,000', 'Private/Government/Self-employed/Unemployed', '0-40', 'Startup/Mid-size/MNC/Government/Other',
                'Rented/Own/Family/Leased', '0-100,000', '1-10', '0-8', '0-50,000', '0-100,000',
                '0-30,000', '0-50,000', '0-50,000',
                'Yes/No', '0-150,000', '300-850', '0-10,000,000',
                '0-1,000,000', 'E-commerce/Home Appliances/Vehicle/Personal/Education', '10,000-2,000,000', '3-84',
                'Eligible/High Risk/Not Eligible', '500-150,000'
            ]
        }
        
        dict_df = pd.DataFrame(data_dict)
        st.dataframe(dict_df, use_container_width=True)
        
        # Download dictionary
        csv_dict = dict_df.to_csv(index=False)
        st.download_button(
            "📥 Download Data Dictionary",
            csv_dict,
            "data_dictionary.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()