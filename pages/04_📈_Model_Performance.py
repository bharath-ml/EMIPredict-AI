"""
Model Performance Dashboard
Comprehensive visualization of model metrics and comparisons
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mlflow
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        border-bottom: 4px solid #6366F1;
    }
    .model-selector {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    </style>
""", unsafe_allow_html=True)

# Sample data for demonstration (replace with actual MLflow data)
CLASSIFICATION_MODELS = {
    'XGBoost': {
        'accuracy': 0.942,
        'precision': 0.941,
        'recall': 0.942,
        'f1': 0.941,
        'auc_roc': 0.978,
        'training_time': 245,
        'inference_time': 0.023
    },
    'Random Forest': {
        'accuracy': 0.938,
        'precision': 0.937,
        'recall': 0.938,
        'f1': 0.937,
        'auc_roc': 0.971,
        'training_time': 312,
        'inference_time': 0.045
    },
    'Gradient Boosting': {
        'accuracy': 0.935,
        'precision': 0.934,
        'recall': 0.935,
        'f1': 0.934,
        'auc_roc': 0.969,
        'training_time': 289,
        'inference_time': 0.034
    },
    'Logistic Regression': {
        'accuracy': 0.891,
        'precision': 0.889,
        'recall': 0.891,
        'f1': 0.890,
        'auc_roc': 0.932,
        'training_time': 89,
        'inference_time': 0.012
    }
}

REGRESSION_MODELS = {
    'XGBoost': {
        'rmse': 1850,
        'mae': 1420,
        'r2': 0.921,
        'mape': 8.2,
        'training_time': 198,
        'inference_time': 0.019
    },
    'Random Forest': {
        'rmse': 1920,
        'mae': 1480,
        'r2': 0.915,
        'mape': 8.5,
        'training_time': 267,
        'inference_time': 0.038
    },
    'Gradient Boosting': {
        'rmse': 1980,
        'mae': 1520,
        'r2': 0.912,
        'mape': 8.7,
        'training_time': 234,
        'inference_time': 0.029
    },
    'Linear Regression': {
        'rmse': 3450,
        'mae': 2780,
        'r2': 0.765,
        'mape': 15.2,
        'training_time': 45,
        'inference_time': 0.008
    }
}

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=500,
        height=500
    )
    
    return fig

def plot_roc_curves():
    """Plot ROC curves for multiple models"""
    fig = go.Figure()
    
    # Generate sample ROC curves
    np.random.seed(42)
    for model_name in CLASSIFICATION_MODELS.keys():
        # Generate sample ROC curve data
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.3) + np.random.normal(0, 0.02, 100)
        tpr = np.clip(tpr, 0, 1)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f"{model_name} (AUC={CLASSIFICATION_MODELS[model_name]['auc_roc']:.3f})",
            line=dict(width=2)
        ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title="ROC Curves Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=500,
        legend=dict(x=0.6, y=0.2)
    )
    
    return fig

def plot_feature_importance():
    """Plot feature importance for best model"""
    features = [
        'credit_score', 'monthly_salary', 'dti_ratio', 'current_emi_amount',
        'years_of_employment', 'age', 'requested_amount', 'requested_tenure',
        'dependents', 'existing_loans', 'emergency_fund', 'bank_balance'
    ]
    
    importances = [0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    
    df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(df, x='importance', y='feature', orientation='h',
                 title="Feature Importance (XGBoost)",
                 color='importance', color_continuous_scale='Viridis')
    
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500
    )
    
    return fig

def plot_residuals():
    """Plot residuals for regression models"""
    np.random.seed(42)
    actual = np.random.uniform(5000, 50000, 1000)
    
    fig = go.Figure()
    
    for model_name in ['XGBoost', 'Random Forest', 'Linear Regression']:
        if model_name == 'XGBoost':
            predicted = actual + np.random.normal(0, 1850, 1000)
            color = '#10B981'
        elif model_name == 'Random Forest':
            predicted = actual + np.random.normal(0, 1920, 1000)
            color = '#6366F1'
        else:
            predicted = actual + np.random.normal(0, 3450, 1000)
            color = '#EF4444'
        
        residuals = predicted - actual
        
        fig.add_trace(go.Scatter(
            x=actual, y=residuals,
            mode='markers',
            name=model_name,
            marker=dict(size=4, color=color, opacity=0.5)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="Residual Plot Comparison",
        xaxis_title="Actual EMI (₹)",
        yaxis_title="Residuals (₹)",
        height=500
    )
    
    return fig

def main():
    st.title("📈 Model Performance Dashboard")
    st.markdown("### Comprehensive Model Evaluation and Comparison")
    
    # Model type selector
    model_type = st.radio(
        "Select Model Type",
        ["Classification Models", "Regression Models"],
        horizontal=True
    )
    
    if model_type == "Classification Models":
        models = CLASSIFICATION_MODELS
        
        # Key metrics overview
        st.subheader("🎯 Key Performance Metrics")
        
        cols = st.columns(4)
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                best_model = max(models.items(), key=lambda x: x[1][metric])
                st.markdown(f'''
                    <div class="metric-box">
                        <h3 style="margin:0; color:#6B7280;">{metric.upper()}</h3>
                        <h2 style="margin:0; font-size:2rem;">{best_model[1][metric]*100:.1f}%</h2>
                        <p style="margin:0; color:#6366F1;">{best_model[0]}</p>
                    </div>
                ''', unsafe_allow_html=True)
        
        # Model comparison table
        st.subheader("📊 Model Comparison")
        
        comparison_df = pd.DataFrame(models).T
        comparison_df = comparison_df.round(4)
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='#10B98180'), 
                    use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy comparison
            fig = px.bar(
                x=list(models.keys()),
                y=[models[m]['accuracy'] for m in models],
                title="Model Accuracy Comparison",
                labels={'x': 'Model', 'y': 'Accuracy'},
                color=[models[m]['accuracy'] for m in models],
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Training time vs accuracy
            fig = px.scatter(
                x=[models[m]['training_time'] for m in models],
                y=[models[m]['accuracy'] for m in models],
                text=list(models.keys()),
                title="Training Time vs Accuracy",
                labels={'x': 'Training Time (seconds)', 'y': 'Accuracy'},
                size=[20] * len(models)
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        # Advanced metrics
        st.subheader("📈 Advanced Performance Metrics")
        
        tab1, tab2, tab3 = st.tabs(["ROC Curves", "Confusion Matrix", "Feature Importance"])
        
        with tab1:
            st.plotly_chart(plot_roc_curves(), use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_model = st.selectbox("Select Model", list(models.keys()))
                # Generate sample confusion matrix data
                if selected_model == 'XGBoost':
                    cm_data = [[85000, 1200, 800], [3500, 42000, 1500], [900, 1100, 48000]]
                elif selected_model == 'Random Forest':
                    cm_data = [[84800, 1300, 900], [3600, 41800, 1600], [950, 1150, 47800]]
                else:
                    cm_data = [[84000, 1500, 1100], [3800, 41500, 1700], [1000, 1200, 47500]]
                
                labels = ['Eligible', 'High Risk', 'Not Eligible']
                fig = plot_confusion_matrix([0]*1000, [0]*1000, labels)  # Placeholder
                fig.update(data=[{'z': cm_data}])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.plotly_chart(plot_feature_importance(), use_container_width=True)
    
    else:  # Regression Models
        models = REGRESSION_MODELS
        
        # Key metrics overview
        st.subheader("🎯 Key Performance Metrics")
        
        cols = st.columns(4)
        metrics = ['rmse', 'mae', 'r2', 'mape']
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                if metric in ['rmse', 'mae', 'mape']:
                    best_model = min(models.items(), key=lambda x: x[1][metric])
                    value = best_model[1][metric]
                    if metric == 'mape':
                        display_value = f"{value:.1f}%"
                    else:
                        display_value = f"₹{value:,.0f}"
                else:
                    best_model = max(models.items(), key=lambda x: x[1][metric])
                    display_value = f"{best_model[1][metric]:.3f}"
                
                st.markdown(f'''
                    <div class="metric-box">
                        <h3 style="margin:0; color:#6B7280;">{metric.upper()}</h3>
                        <h2 style="margin:0; font-size:2rem;">{display_value}</h2>
                        <p style="margin:0; color:#6366F1;">{best_model[0]}</p>
                    </div>
                ''', unsafe_allow_html=True)
        
        # Model comparison table
        st.subheader("📊 Model Comparison")
        
        comparison_df = pd.DataFrame(models).T
        comparison_df = comparison_df.round(4)
        st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['rmse', 'mae', 'mape'], color='#10B98180')
                                .highlight_max(axis=0, subset=['r2'], color='#10B98180'), 
                    use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE comparison
            fig = px.bar(
                x=list(models.keys()),
                y=[models[m]['rmse'] for m in models],
                title="Model RMSE Comparison (Lower is Better)",
                labels={'x': 'Model', 'y': 'RMSE (₹)'},
                color=[models[m]['rmse'] for m in models],
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # R² comparison
            fig = px.bar(
                x=list(models.keys()),
                y=[models[m]['r2'] for m in models],
                title="Model R² Comparison (Higher is Better)",
                labels={'x': 'Model', 'y': 'R² Score'},
                color=[models[m]['r2'] for m in models],
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Residual analysis
        st.subheader("📉 Residual Analysis")
        st.plotly_chart(plot_residuals(), use_container_width=True)
        
        # Model performance over time
        st.subheader("📊 Performance Over Time")
        
        # Generate sample time series data
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        performance_data = []
        
        for model in models.keys():
            if model == 'XGBoost':
                base_rmse = 1850
                trend = np.random.normal(0, 50, len(dates))
            elif model == 'Random Forest':
                base_rmse = 1920
                trend = np.random.normal(0, 60, len(dates))
            else:
                base_rmse = 3450
                trend = np.random.normal(0, 150, len(dates))
            
            for i, date in enumerate(dates):
                performance_data.append({
                    'date': date,
                    'model': model,
                    'rmse': base_rmse + trend[i] + i * 0.5  # Slight increasing trend
                })
        
        perf_df = pd.DataFrame(performance_data)
        
        fig = px.line(perf_df, x='date', y='rmse', color='model',
                     title="RMSE Trend Over Time",
                     labels={'rmse': 'RMSE (₹)', 'date': 'Date'})
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()