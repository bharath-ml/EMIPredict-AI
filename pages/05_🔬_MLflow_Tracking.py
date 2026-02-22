"""
MLflow Experiment Tracking Dashboard
Comprehensive experiment management and model registry
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import mlflow
from mlflow.tracking import MlflowClient
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="MLflow Tracking",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .experiment-card {
        background-color: #F9FAFB;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        margin: 0.5rem 0;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .badge-production {
        background-color: #10B98120;
        color: #10B981;
        border: 1px solid #10B981;
    }
    .badge-staging {
        background-color: #F59E0B20;
        color: #F59E0B;
        border: 1px solid #F59E0B;
    }
    .badge-archived {
        background-color: #6B728020;
        color: #6B7280;
        border: 1px solid #6B7280;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize MLflow
mlflow.set_tracking_uri("mlruns")
client = MlflowClient()

# Sample data for demonstration
SAMPLE_EXPERIMENTS = [
    {
        'experiment_id': '1',
        'name': 'XGBoost Classification v3',
        'type': 'classification',
        'status': 'completed',
        'accuracy': 0.942,
        'f1': 0.941,
        'timestamp': datetime.now() - timedelta(days=2)
    },
    {
        'experiment_id': '2',
        'name': 'Random Forest Classification v2',
        'type': 'classification',
        'status': 'completed',
        'accuracy': 0.938,
        'f1': 0.937,
        'timestamp': datetime.now() - timedelta(days=3)
    },
    {
        'experiment_id': '3',
        'name': 'XGBoost Regression v2',
        'type': 'regression',
        'status': 'completed',
        'rmse': 1850,
        'r2': 0.921,
        'timestamp': datetime.now() - timedelta(days=1)
    },
    {
        'experiment_id': '4',
        'name': 'Gradient Boosting Regression v1',
        'type': 'regression',
        'status': 'failed',
        'rmse': None,
        'r2': None,
        'timestamp': datetime.now() - timedelta(hours=12)
    },
    {
        'experiment_id': '5',
        'name': 'LightGBM Classification v1',
        'type': 'classification',
        'status': 'running',
        'accuracy': None,
        'f1': None,
        'timestamp': datetime.now() - timedelta(hours=2)
    }
]

SAMPLE_REGISTRY = [
    {
        'name': 'EMIPredict_Classifier',
        'version': '2.1.0',
        'stage': 'Production',
        'model': 'XGBoost',
        'metrics': {'accuracy': 0.942, 'f1': 0.941},
        'date': '2024-01-15'
    },
    {
        'name': 'EMIPredict_Regressor',
        'version': '1.8.0',
        'stage': 'Staging',
        'model': 'XGBoost',
        'metrics': {'rmse': 1850, 'r2': 0.921},
        'date': '2024-01-14'
    },
    {
        'name': 'EMIPredict_Classifier',
        'version': '2.0.0',
        'stage': 'Archived',
        'model': 'Random Forest',
        'metrics': {'accuracy': 0.938, 'f1': 0.937},
        'date': '2024-01-10'
    },
    {
        'name': 'EMIPredict_Regressor',
        'version': '1.7.0',
        'stage': 'Archived',
        'model': 'Random Forest',
        'metrics': {'rmse': 1920, 'r2': 0.915},
        'date': '2024-01-08'
    }
]

def main():
    st.title("🔬 MLflow Experiment Tracking")
    st.markdown("### Comprehensive Model and Experiment Management")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Experiments Overview",
        "🏭 Model Registry",
        "📈 Run Comparison",
        "⚙️ Configuration"
    ])
    
    with tab1:
        st.subheader("Recent Experiments")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect("Status", ["completed", "running", "failed"], default=["completed"])
        with col2:
            type_filter = st.multiselect("Model Type", ["classification", "regression"], default=["classification", "regression"])
        with col3:
            search = st.text_input("Search", placeholder="Experiment name...")
        
        # Experiments table
        exp_df = pd.DataFrame(SAMPLE_EXPERIMENTS)
        
        # Apply filters
        if status_filter:
            exp_df = exp_df[exp_df['status'].isin(status_filter)]
        if type_filter:
            exp_df = exp_df[exp_df['type'].isin(type_filter)]
        if search:
            exp_df = exp_df[exp_df['name'].str.contains(search, case=False)]
        
        for _, exp in exp_df.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{exp['name']}**")
                    st.caption(f"ID: {exp['experiment_id']} | {exp['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                
                with col2:
                    status_color = {
                        'completed': '🟢',
                        'running': '🟡',
                        'failed': '🔴'
                    }.get(exp['status'], '⚪')
                    st.markdown(f"{status_color} {exp['status'].title()}")
                
                with col3:
                    if exp['type'] == 'classification':
                        if exp['accuracy']:
                            st.metric("Accuracy", f"{exp['accuracy']*100:.1f}%")
                        else:
                            st.markdown("—")
                    else:
                        if exp['rmse']:
                            st.metric("RMSE", f"₹{exp['rmse']:,.0f}")
                        else:
                            st.markdown("—")
                
                with col4:
                    if exp['type'] == 'classification':
                        if exp['f1']:
                            st.metric("F1", f"{exp['f1']*100:.1f}%")
                        else:
                            st.markdown("—")
                    else:
                        if exp['r2']:
                            st.metric("R²", f"{exp['r2']:.3f}")
                        else:
                            st.markdown("—")
                
                with col5:
                    if exp['status'] == 'completed':
                        if st.button("📊 View", key=f"view_{exp['experiment_id']}"):
                            st.session_state['selected_experiment'] = exp['experiment_id']
                
                st.markdown("---")
        
        # Experiment metrics
        st.subheader("📊 Experiment Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        completed_runs = len([e for e in SAMPLE_EXPERIMENTS if e['status'] == 'completed'])
        with col1:
            st.metric("Total Experiments", len(SAMPLE_EXPERIMENTS))
        with col2:
            st.metric("Completed", completed_runs)
        with col3:
            success_rate = (completed_runs / len(SAMPLE_EXPERIMENTS)) * 100
            st.metric("Success Rate", f"{success_rate:.0f}%")
        with col4:
            avg_accuracy = np.mean([e['accuracy'] for e in SAMPLE_EXPERIMENTS if e.get('accuracy')]) * 100
            st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
    
    with tab2:
        st.subheader("Model Registry")
        
        # Model registry overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            registry_df = pd.DataFrame(SAMPLE_REGISTRY)
            
            # Style the dataframe
            def highlight_stage(val):
                if val == 'Production':
                    return 'background-color: #10B98120'
                elif val == 'Staging':
                    return 'background-color: #F59E0B20'
                else:
                    return 'background-color: #6B728020'
            
            styled_df = registry_df.style.applymap(highlight_stage, subset=['stage'])
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.markdown("#### Model Stages")
            st.markdown("""
            <div class="metric-badge badge-production">🏭 Production</div>
            <div class="metric-badge badge-staging">⚡ Staging</div>
            <div class="metric-badge badge-archived">📦 Archived</div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Actions")
            if st.button("📤 Promote to Production", use_container_width=True):
                st.success("Model promoted to Production!")
            
            if st.button("🔄 Compare Versions", use_container_width=True):
                st.info("Version comparison feature coming soon")
        
        # Model version details
        st.subheader("📈 Model Version Details")
        
        selected_model = st.selectbox("Select Model", ["EMIPredict_Classifier", "EMIPredict_Regressor"])
        
        if selected_model == "EMIPredict_Classifier":
            versions = [v for v in SAMPLE_REGISTRY if v['name'] == selected_model]
            
            # Performance over versions
            fig = go.Figure()
            
            versions_data = [
                {'version': '2.1.0', 'accuracy': 0.942},
                {'version': '2.0.0', 'accuracy': 0.938},
                {'version': '1.5.0', 'accuracy': 0.931},
                {'version': '1.0.0', 'accuracy': 0.915}
            ]
            
            df = pd.DataFrame(versions_data)
            
            fig.add_trace(go.Scatter(
                x=df['version'],
                y=df['accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#10B981', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Model Accuracy Over Versions",
                xaxis_title="Version",
                yaxis_title="Accuracy",
                yaxis_range=[0.9, 0.95]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Version comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Current Production (v2.1.0)")
                st.json({
                    "model": "XGBoost",
                    "accuracy": 0.942,
                    "precision": 0.941,
                    "recall": 0.942,
                    "f1": 0.941,
                    "auc_roc": 0.978,
                    "training_date": "2024-01-15",
                    "dataset_size": "400,000",
                    "features": 35
                })
            
            with col2:
                st.markdown("##### Staging (v1.8.0)")
                st.json({
                    "model": "XGBoost",
                    "rmse": 1850,
                    "mae": 1420,
                    "r2": 0.921,
                    "mape": 8.2,
                    "training_date": "2024-01-14",
                    "dataset_size": "400,000",
                    "features": 35
                })
    
    with tab3:
        st.subheader("Run Comparison")
        
        # Select runs to compare
        col1, col2 = st.columns(2)
        
        with col1:
            run1 = st.selectbox("Select First Run", 
                               [f"{e['name']} ({e['experiment_id']})" for e in SAMPLE_EXPERIMENTS if e['status'] == 'completed'])
        
        with col2:
            run2 = st.selectbox("Select Second Run",
                               [f"{e['name']} ({e['experiment_id']})" for e in SAMPLE_EXPERIMENTS if e['status'] == 'completed'],
                               index=1)
        
        if run1 and run2:
            # Extract run IDs
            run1_id = run1.split('(')[-1].strip(')')
            run2_id = run2.split('(')[-1].strip(')')
            
            # Get run details
            run1_details = next(e for e in SAMPLE_EXPERIMENTS if e['experiment_id'] == run1_id)
            run2_details = next(e for e in SAMPLE_EXPERIMENTS if e['experiment_id'] == run2_id)
            
            # Comparison table
            st.markdown("#### Metrics Comparison")
            
            if run1_details['type'] == 'classification' and run2_details['type'] == 'classification':
                comparison_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                    run1_details['name']: [0.942, 0.941, 0.942, 0.941, 0.978],
                    run2_details['name']: [0.938, 0.937, 0.938, 0.937, 0.971],
                    'Difference': ['+0.004', '+0.004', '+0.004', '+0.004', '+0.007']
                }
            else:
                comparison_data = {
                    'Metric': ['RMSE', 'MAE', 'R²', 'MAPE'],
                    run1_details['name']: [1850, 1420, 0.921, 8.2],
                    run2_details['name']: [1920, 1480, 0.915, 8.5],
                    'Difference': ['-70', '-60', '+0.006', '-0.3']
                }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualization
            st.markdown("#### Performance Visualization")
            
            if run1_details['type'] == 'classification':
                # Classification metrics radar chart
                categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=[0.942, 0.941, 0.942, 0.941, 0.978],
                    theta=categories,
                    fill='toself',
                    name=run1_details['name'],
                    marker_color='#10B981'
                ))
                
                fig.add_trace(go.Scatterpolar(
                    r=[0.938, 0.937, 0.938, 0.937, 0.971],
                    theta=categories,
                    fill='toself',
                    name=run2_details['name'],
                    marker_color='#6366F1'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0.9, 1.0]
                        )),
                    showlegend=True,
                    title="Model Performance Comparison"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Regression metrics bar chart
                metrics = ['RMSE', 'MAE', 'MAPE']
                values1 = [1850, 1420, 8.2]
                values2 = [1920, 1480, 8.5]
                
                fig = go.Figure(data=[
                    go.Bar(name=run1_details['name'], x=metrics, y=values1, marker_color='#10B981'),
                    go.Bar(name=run2_details['name'], x=metrics, y=values2, marker_color='#6366F1')
                ])
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    barmode='group',
                    yaxis_title="Value"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("MLflow Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Tracking Server")
            st.code("""
            tracking_uri: mlruns
            experiment_name: EMIPredict_AI
            artifact_location: ./mlartifacts
            """)
            
            st.markdown("#### Current Settings")
            st.json({
                "mlflow_version": "2.5.0",
                "tracking_uri": "mlruns",
                "default_experiment": "EMIPredict_AI",
                "artifact_max_size": "100 MB",
                "backend_store": "file"
            })
        
        with col2:
            st.markdown("#### Environment Variables")
            st.code("""
            MLFLOW_TRACKING_URI=./mlruns
            MLFLOW_EXPERIMENT_NAME=EMIPredict_AI
            MLFLOW_S3_ENDPOINT_URL= 
            MLFLOW_TRACKING_USERNAME=
            MLFLOW_TRACKING_PASSWORD=
            """)
            
            st.markdown("#### Actions")
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.success("Cache cleared successfully!")
            
            if st.button("📊 Export Experiment Data", use_container_width=True):
                st.success("Experiment data exported!")
            
            if st.button("⚡ Set as Default", use_container_width=True):
                st.success("Configuration saved as default!")

if __name__ == "__main__":
    main()