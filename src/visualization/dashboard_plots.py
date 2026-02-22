"""
Dashboard Visualization Module
Professional plotting utilities for Streamlit dashboard
"""
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DashboardVisualizer:
    """Professional dashboard visualization class"""
    
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
        self.colors = {
            'primary': '#6366F1',
            'success': '#10B981',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'info': '#3B82F6',
            'gray': '#6B7280'
        }
    
    def create_gauge_chart(self, value: float, title: str, 
                          min_val: float = 0, max_val: float = 100,
                          thresholds: Optional[List[Dict]] = None) -> go.Figure:
        """
        Create a gauge chart
        
        Args:
            value: Current value
            title: Chart title
            min_val: Minimum value
            max_val: Maximum value
            thresholds: Threshold definitions
            
        Returns:
            Plotly figure
        """
        if thresholds is None:
            thresholds = [
                {'range': [0, 30], 'color': self.colors['danger']},
                {'range': [30, 70], 'color': self.colors['warning']},
                {'range': [70, 100], 'color': self.colors['success']}
            ]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 24}},
            delta={'reference': (max_val + min_val) / 2},
            gauge={
                'axis': {'range': [min_val, max_val], 'tickwidth': 1},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': t['range'], 'color': t['color']} for t in thresholds
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_metric_card(self, title: str, value: str, 
                          delta: Optional[str] = None,
                          icon: Optional[str] = None) -> go.Figure:
        """
        Create a metric card visualization
        
        Args:
            title: Metric title
            value: Metric value
            delta: Change indicator
            icon: Icon emoji
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="number+delta" if delta else "number",
            value=float(value.replace(',', '').replace('₹', '').replace('%', '')) if value.replace(',', '').replace('₹', '').replace('%', '').replace('.', '').isdigit() else 0,
            number={'prefix': '₹' if '₹' in value else '', 
                   'suffix': '%' if '%' in value else '',
                   'font': {'size': 40}},
            delta={'reference': float(delta.replace(',', '').replace('₹', '').replace('%', '')) if delta and delta.replace(',', '').replace('₹', '').replace('%', '').replace('.', '').replace('-', '').isdigit() else 0,
                   'relative': True} if delta else None,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{icon} {title}" if icon else title, 'font': {'size': 16}}
        ))
        
        fig.update_layout(
            height=200,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_comparison_chart(self, data: pd.DataFrame, 
                               x_col: str, y_cols: List[str],
                               title: str, chart_type: str = "bar") -> go.Figure:
        """
        Create comparison chart
        
        Args:
            data: DataFrame with data
            x_col: Column for x-axis
            y_cols: Columns for y-axis
            title: Chart title
            chart_type: "bar", "line", or "scatter"
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for i, y_col in enumerate(y_cols):
            color = list(self.colors.values())[i % len(self.colors)]
            
            if chart_type == "bar":
                fig.add_trace(go.Bar(
                    x=data[x_col],
                    y=data[y_col],
                    name=y_col,
                    marker_color=color
                ))
            elif chart_type == "line":
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines+markers',
                    name=y_col,
                    line=dict(color=color, width=3),
                    marker=dict(size=8)
                ))
            else:  # scatter
                fig.add_trace(go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    name=y_col,
                    marker=dict(color=color, size=10)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title="Value",
            barmode='group',
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_distribution_chart(self, data: pd.Series, 
                                 title: str, bins: int = 50) -> go.Figure:
        """
        Create distribution histogram
        
        Args:
            data: Data series
            title: Chart title
            bins: Number of bins
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            marker_color=self.colors['primary'],
            opacity=0.75
        ))
        
        # Add mean line
        mean_val = data.mean()
        fig.add_vline(x=mean_val, line_dash="dash", 
                     line_color=self.colors['danger'],
                     annotation_text=f"Mean: {mean_val:.2f}",
                     annotation_position="top")
        
        fig.update_layout(
            title=title,
            xaxis_title=data.name,
            yaxis_title="Frequency",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=False
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame,
                                  title: str = "Correlation Heatmap") -> go.Figure:
        """
        Create correlation heatmap
        
        Args:
            data: DataFrame with numerical columns
            title: Chart title
            
        Returns:
            Plotly figure
        """
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=corr_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_radar_chart(self, categories: List[str], 
                          values: List[float], title: str) -> go.Figure:
        """
        Create radar chart
        
        Args:
            categories: Category names
            values: Values for each category
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the loop
            theta=categories + [categories[0]],
            fill='toself',
            marker=dict(color=self.colors['primary']),
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        fig.update_layout(
            title=title,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]
                )
            ),
            height=500,
            margin=dict(l=80, r=80, t=50, b=50),
            showlegend=False
        )
        
        return fig
    
    def create_time_series(self, data: pd.DataFrame, date_col: str,
                          value_cols: List[str], title: str) -> go.Figure:
        """
        Create time series chart
        
        Args:
            data: DataFrame with time series data
            date_col: Date column name
            value_cols: Value columns to plot
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for i, value_col in enumerate(value_cols):
            color = list(self.colors.values())[i % len(self.colors)]
            
            fig.add_trace(go.Scatter(
                x=data[date_col],
                y=data[value_col],
                mode='lines',
                name=value_col,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='x unified'
        )
        
        return fig
    
    def create_pie_chart(self, labels: List[str], values: List[float],
                        title: str, hole: float = 0.3) -> go.Figure:
        """
        Create pie/donut chart
        
        Args:
            labels: Slice labels
            values: Slice values
            title: Chart title
            hole: Size of center hole (0 for pie, >0 for donut)
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=hole,
            marker=dict(colors=list(self.colors.values())),
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title=title,
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        
        return fig
    
    def create_scatter_plot(self, data: pd.DataFrame, x_col: str,
                           y_col: str, color_col: Optional[str] = None,
                           size_col: Optional[str] = None,
                           title: str = "Scatter Plot") -> go.Figure:
        """
        Create scatter plot
        
        Args:
            data: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Color by column
            size_col: Size by column
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=title,
            color_continuous_scale='Viridis' if color_col else None
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='closest'
        )
        
        return fig
    
    def create_box_plot(self, data: pd.DataFrame, x_col: str,
                       y_col: str, title: str = "Box Plot") -> go.Figure:
        """
        Create box plot
        
        Args:
            data: DataFrame
            x_col: X-axis column (categorical)
            y_col: Y-axis column (numerical)
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = px.box(data, x=x_col, y=y_col, title=title,
                    color=x_col, color_discrete_sequence=list(self.colors.values()))
        
        fig.update_layout(
            height=500,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=False
        )
        
        return fig
    
    def create_3d_scatter(self, data: pd.DataFrame, x_col: str,
                         y_col: str, z_col: str,
                         color_col: Optional[str] = None,
                         title: str = "3D Scatter Plot") -> go.Figure:
        """
        Create 3D scatter plot
        
        Args:
            data: DataFrame
            x_col: X-axis column
            y_col: Y-axis column
            z_col: Z-axis column
            color_col: Color by column
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=data[x_col],
            y=data[y_col],
            z=data[z_col],
            mode='markers',
            marker=dict(
                size=5,
                color=data[color_col] if color_col else self.colors['primary'],
                colorscale='Viridis',
                showscale=True if color_col else False
            ),
            text=data.index
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            ),
            height=600,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def create_parallel_coordinates(self, data: pd.DataFrame,
                                   dimensions: List[str],
                                   color_col: str,
                                   title: str = "Parallel Coordinates Plot") -> go.Figure:
        """
        Create parallel coordinates plot
        
        Args:
            data: DataFrame
            dimensions: Columns to include
            color_col: Column for coloring
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=data[color_col],
                colorscale='Viridis',
                showscale=True
            ),
            dimensions=[
                dict(
                    label=col,
                    values=data[col],
                    range=[data[col].min(), data[col].max()]
                ) for col in dimensions
            ]
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def create_sankey_diagram(self, labels: List[str],
                             source: List[int], target: List[int],
                             values: List[float],
                             title: str = "Sankey Diagram") -> go.Figure:
        """
        Create Sankey diagram
        
        Args:
            labels: Node labels
            source: Source node indices
            target: Target node indices
            values: Flow values
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=self.colors['primary']
            ),
            link=dict(
                source=source,
                target=target,
                value=values
            )
        )])
        
        fig.update_layout(
            title=title,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig