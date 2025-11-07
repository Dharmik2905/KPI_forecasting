import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Forecast Lag Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Enhanced metric styling with gradients */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        border-left: 5px solid #ffd700;
        transition: transform 0.2s;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    div[data-testid="stMetricValue"] {
        color: white;
        font-weight: bold;
        font-size: 1.8rem;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #f0f0f0;
        font-weight: 600;
        font-size: 1rem;
    }
    
    div[data-testid="stMetricDelta"] {
        color: #ffd700;
    }
    
    .tooltip-text {
        font-size: 0.85em;
        color: #666;
        font-style: italic;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    h3 {
        color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# ==================== KPI DEFINITIONS ====================
KPI_DEFINITIONS = {
    'MAE': {
        'name': 'Mean Absolute Error',
        'formula': 'MAE = mean(|Actual - Forecast|)',
        'description': 'Average absolute difference between actual and forecast values. Lower is better. Easy to interpret in original units.',
        'interpretation': 'Measures average forecast error magnitude. A MAE of 50 means forecasts are off by 50 units on average.'
    },
    'RMSE': {
        'name': 'Root Mean Squared Error',
        'formula': 'RMSE = sqrt(mean((Actual - Forecast)²))',
        'description': 'Square root of average squared errors. Penalizes large errors more heavily than MAE. Lower is better.',
        'interpretation': 'Higher sensitivity to outliers than MAE. Use when large errors are particularly costly.'
    },
    'Bias_%': {
        'name': 'Bias Percentage',
        'formula': 'Bias% = (sum(Forecast - Actual) / sum(Actual)) × 100',
        'description': 'Measures systematic over-forecasting (positive) or under-forecasting (negative). Optimal value is 0.',
        'interpretation': 'Bias of +10% means forecasts are consistently 10% too high. Near-zero bias indicates balanced forecasting.'
    },
    'Service_Level_%': {
        'name': 'Service Level Percentage',
        'formula': 'Service Level% = (count(Forecast >= Actual) / Total Periods) × 100',
        'description': 'Percentage of periods where forecast met or exceeded actual demand. Higher is better for inventory planning.',
        'interpretation': '90% service level means demand was met in 9 out of 10 periods. Critical for stockout prevention.'
    },
    'SMAPE': {
        'name': 'Symmetric Mean Absolute Percentage Error',
        'formula': 'SMAPE = mean(200 × |Actual - Forecast| / (|Actual| + |Forecast|))',
        'description': 'Scale-independent percentage error. Bounded between 0-200%. Lower is better. Handles zero values better than MAPE.',
        'interpretation': 'Allows comparison across different SKUs. SMAPE of 20% indicates 20% average relative error.'
    },
    'Tracking_Signal': {
        'name': 'Tracking Signal',
        'formula': 'TS = Cumulative Forecast Error / Mean Absolute Deviation',
        'description': 'Detects forecast bias drift over time. Ideal range: -4 to +4. Outside this range indicates systematic bias.',
        'interpretation': 'Values beyond ±4 suggest forecast model needs recalibration. Positive = over-forecasting trend.'
    }
}

# ==================== DATA LOADING ====================
@st.cache_data
def load_data():
    """Load KPI summary and detailed forecasts"""
    try:
        kpi_df = pd.read_csv('kpi_summary_all.csv')
        detailed_df = pd.read_csv('detailed_forecasts_all.csv')
        
        # Convert date column
        if 'Date' in detailed_df.columns:
            detailed_df['Date'] = pd.to_datetime(detailed_df['Date'])
        
        return kpi_df, detailed_df
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure 'kpi_summary_all.csv' and 'detailed_forecasts_all.csv' are in the same directory as this script.")
        st.stop()

@st.cache_data
def load_monthly_data():
    """Load complete monthly historical data"""
    try:
        monthly_df = pd.read_csv('monthly_data.csv')
        monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
        return monthly_df
    except FileNotFoundError:
        st.warning("monthly_data.csv not found. Historical view will be limited.")
        return None

# ==================== HELPER FUNCTIONS ====================
def create_kpi_tooltip(kpi_name):
    """Create tooltip text for KPI metrics"""
    if kpi_name in KPI_DEFINITIONS:
        kpi_info = KPI_DEFINITIONS[kpi_name]
        return f"""
        **{kpi_info['name']}**
        
        Formula: `{kpi_info['formula']}`
        
        Description: {kpi_info['description']}
        
        Interpretation: {kpi_info['interpretation']}
        """
    return ""

def generate_ai_recommendation(kpi_df, detailed_df, api_key):
    """Generate AI-powered recommendation using Gemini via OpenRouter"""
    
    if not api_key or api_key == "":
        return None
    
    # Prepare context for AI
    summary_stats = kpi_df.groupby(['Lag_Period', 'Model']).agg({
        'MAE': 'mean',
        'RMSE': 'mean',
        'SMAPE': 'mean',
        'Bias_%': 'mean',
        'Service_Level_%': 'mean'
    }).reset_index()
    
    # SKU-level insights
    sku_performance = kpi_df.groupby('SKU_ID').agg({
        'MAE': 'mean',
        'RMSE': 'mean',
        'SMAPE': 'mean'
    }).reset_index()
    
    # Best lag by metric
    best_lags = {
        'MAE': kpi_df.groupby('Lag_Period')['MAE'].mean().idxmin(),
        'RMSE': kpi_df.groupby('Lag_Period')['RMSE'].mean().idxmin(),
        'SMAPE': kpi_df.groupby('Lag_Period')['SMAPE'].mean().idxmin()
    }
    
    context = f"""
You are a demand planning expert. Explain forecast lag analysis in VERY SIMPLE, plain language for business users who may not be technical.

## GENERAL BEST PRACTICES (Use this to guide your recommendations):

The effect of lag depends on the forecasting model. For RMSE and MAE, errors are lowest at 1-month lag and generally increase at 3 and 6-month lags. However, at 12 months, errors often decrease again because the model captures full seasonal patterns.

Growing SKUs (like 1-5, 8-9) perform best with shorter lags (1-3 months) because they capture recent market changes. These SKUs show increasing errors from 1 to 6 months, then improvement at 12 months due to seasonality.

Declining SKUs (like 6, 7, 10) are more stable and predictable. They perform well with longer lags (6-12 months) because their downward trend is less affected by short-term changes.

The key insight: there's no one-size-fits-all. Fast-moving products need frequent updates (1-3 months). Mature or seasonal products do well with longer windows (12 months). Demand planners should compare KPIs across different lags to find the right balance between responsiveness and stability.

## YOUR DATA SUMMARY:
Average KPIs by Lag:
{summary_stats.to_string()}

Best Lag by Metric:
- MAE: {best_lags['MAE']} months
- RMSE: {best_lags['RMSE']} months  
- SMAPE: {best_lags['SMAPE']} months

SKU Performance:
{sku_performance.to_string()}

## YOUR TASK:
Write a SIMPLE recommendation (250-300 words maximum) that:
1. Tells which lag period is best overall in plain English
2. Explains which SKUs need shorter vs longer update frequencies (keep it simple)
3. Gives 2-3 clear action steps

Use everyday language. Avoid jargon. Pretend you're explaining to someone who doesn't know statistics. Use short sentences and bullet points.
"""
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "google/gemini-2.0-flash-exp:free",
                "messages": [
                    {
                        "role": "user",
                        "content": context
                    }
                ]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_kpi_trends(df, kpi_metric, selected_sku, selected_lags):
    """Line chart showing KPI trends across lag periods for ONE SKU"""
    filtered_df = df[
        (df['SKU_ID'] == selected_sku) & 
        (df['Lag_Period'].isin(selected_lags))
    ]
    
    fig = px.line(
        filtered_df,
        x='Lag_Period',
        y=kpi_metric,
        markers=True,
        title=f'{KPI_DEFINITIONS[kpi_metric]["name"]} Trend for SKU {selected_sku}',
        labels={'Lag_Period': 'Lag Period (Months)', kpi_metric: kpi_metric},
        hover_data=['Model']
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified'
    )
    
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    
    return fig

def plot_kpi_heatmap(df, kpi_metric, selected_lags):
    """Heatmap showing KPI values across SKUs and lag periods"""
    filtered_df = df[df['Lag_Period'].isin(selected_lags)]
    
    pivot_df = filtered_df.pivot_table(
        values=kpi_metric,
        index='Lag_Period',
        columns='SKU_ID',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale='RdYlGn_r' if kpi_metric in ['MAE', 'RMSE', 'SMAPE'] else 'RdYlGn',
        text=np.round(pivot_df.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title=kpi_metric)
    ))
    
    fig.update_layout(
        title=f'{KPI_DEFINITIONS[kpi_metric]["name"]} Heatmap: SKUs vs Lag Periods',
        xaxis_title='SKU ID',
        yaxis_title='Lag Period (Months)',
        height=400
    )
    
    return fig

def plot_model_comparison(df, kpi_metric, selected_sku, selected_lags):
    """Bar chart comparing models for ONE SKU"""
    filtered_df = df[
        (df['SKU_ID'] == selected_sku) & 
        (df['Lag_Period'].isin(selected_lags))
    ]
    
    model_avg = filtered_df.groupby('Model')[kpi_metric].mean().reset_index()
    model_avg = model_avg.sort_values(kpi_metric)
    
    fig = px.bar(
        model_avg,
        x='Model',
        y=kpi_metric,
        title=f'Average {KPI_DEFINITIONS[kpi_metric]["name"]} by Model (SKU {selected_sku})',
        color=kpi_metric,
        color_continuous_scale='RdYlGn_r' if kpi_metric in ['MAE', 'RMSE', 'SMAPE'] else 'RdYlGn',
        text=kpi_metric
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def plot_comparative_lag_analysis(df, selected_sku):
    """Multi-panel comparison of all KPIs across lag periods - FIXED LAYOUT"""
    filtered_df = df[df['SKU_ID'] == selected_sku]
    
    metrics = ['MAE', 'RMSE', 'SMAPE', 'Bias_%']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[KPI_DEFINITIONS[m]['name'] for m in metrics],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    for idx, metric in enumerate(metrics):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        lag_avg = filtered_df.groupby('Lag_Period')[metric].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=lag_avg['Lag_Period'],
                y=lag_avg[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=3),
                marker=dict(size=10),
                showlegend=False
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Lag Period (Months)", row=row, col=col)
        fig.update_yaxes(title_text=metric, row=row, col=col)
    
    fig.update_layout(
        height=700,
        title_text=f"Comparative Lag Analysis: All KPIs (SKU {selected_sku})",
        showlegend=False
    )
    
    return fig

def plot_sku_time_series(detailed_df, sku_id, lag_period):
    """Plot actual vs forecast with confidence intervals for a specific SKU"""
    sku_data = detailed_df[
        (detailed_df['SKU_ID'] == sku_id) & 
        (detailed_df['Lag_Period'] == lag_period)
    ].sort_values('Date')
    
    if len(sku_data) == 0:
        return None
    
    fig = go.Figure()
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=sku_data['Date'],
        y=sku_data['CI_Upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=sku_data['Date'],
        y=sku_data['CI_Lower'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.2)',
        fill='tonexty',
        name='95% Confidence Interval',
        hoverinfo='skip'
    ))
    
    # Actual values
    fig.add_trace(go.Scatter(
        x=sku_data['Date'],
        y=sku_data['Actual'],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Forecast values
    fig.add_trace(go.Scatter(
        x=sku_data['Date'],
        y=sku_data['Forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f'SKU {sku_id} - Actual vs Forecast (Lag: {lag_period} months)',
        xaxis_title='Date',
        yaxis_title='Sales Volume',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_rolling_stats(detailed_df, sku_id, window=3):
    """Plot rolling mean and standard deviation for a SKU"""
    sku_data = detailed_df[detailed_df['SKU_ID'] == sku_id].sort_values('Date')
    
    if len(sku_data) == 0:
        return None
    
    # Calculate rolling statistics
    sku_data['Rolling_Mean'] = sku_data['Actual'].rolling(window=window, min_periods=1).mean()
    sku_data['Rolling_Std'] = sku_data['Actual'].rolling(window=window, min_periods=1).std()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Sales with Rolling Mean', 'Rolling Standard Deviation'),
        vertical_spacing=0.15
    )
    
    # Sales and rolling mean
    fig.add_trace(
        go.Scatter(x=sku_data['Date'], y=sku_data['Actual'], 
                   mode='lines', name='Actual Sales', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=sku_data['Date'], y=sku_data['Rolling_Mean'], 
                   mode='lines', name=f'{window}-Month Rolling Mean', 
                   line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Rolling std
    fig.add_trace(
        go.Scatter(x=sku_data['Date'], y=sku_data['Rolling_Std'], 
                   mode='lines', name='Rolling Std Dev', 
                   line=dict(color='orange', width=2), showlegend=True),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sales Volume", row=1, col=1)
    fig.update_yaxes(title_text="Std Deviation", row=2, col=1)
    
    fig.update_layout(height=600, title_text=f"SKU {sku_id} - Rolling Statistics Analysis")
    
    return fig

def plot_complete_history_with_forecast(monthly_df, detailed_df, sku_id, lag_period, model_name):
    """Plot complete historical sales with train/forecast split and overlay forecast"""
    
    if monthly_df is None:
        return None
    
    # Get complete historical data for this SKU
    sku_historical = monthly_df[monthly_df['SKU_ID'] == sku_id].sort_values('Date').copy()
    
    if len(sku_historical) == 0:
        return None
    
    # Get forecast data
    sku_forecast = detailed_df[
        (detailed_df['SKU_ID'] == sku_id) & 
        (detailed_df['Lag_Period'] == lag_period)
    ].sort_values('Date')
    
    if len(sku_forecast) == 0:
        return None
    
    # Get train end date (last date before forecast starts)
    forecast_start = sku_forecast['Date'].min()
    
    # Split historical data into train and actual (forecast period)
    train_data = sku_historical[sku_historical['Date'] < forecast_start]
    actual_forecast_period = sku_historical[sku_historical['Date'] >= forecast_start]
    
    fig = go.Figure()
    
    # Plot training data (gray)
    fig.add_trace(go.Scatter(
        x=train_data['Date'],
        y=train_data['Monthly_Sales'],
        mode='lines',
        name='Train',
        line=dict(color='gray', width=1.5),
        opacity=0.7
    ))
    
    # Plot actual data in forecast period (green)
    if len(actual_forecast_period) > 0:
        fig.add_trace(go.Scatter(
            x=actual_forecast_period['Date'],
            y=actual_forecast_period['Monthly_Sales'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='green', width=2),
            marker=dict(size=8, symbol='circle')
        ))
    
    # Plot forecast (red dashed)
    fig.add_trace(go.Scatter(
        x=sku_forecast['Date'],
        y=sku_forecast['Forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=sku_forecast['Date'],
        y=sku_forecast['CI_Upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=sku_forecast['Date'],
        y=sku_forecast['CI_Lower'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255, 192, 203, 0.3)',
        fill='tonexty',
        name='CI',
        hoverinfo='skip'
    ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=forecast_start,
        line_dash="dot",
        line_color="blue",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    # Calculate MAE for title
    mae = sku_forecast['Abs_Error'].mean()
    
    fig.update_layout(
        title=f'{model_name}: SKU {sku_id} | {lag_period}-Month Lag | MAE: {mae:.2f}',
        xaxis_title='Date',
        yaxis_title='Sales Volume',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def plot_rolling_stats_complete(monthly_df, sku_id, window=3):
    """Plot rolling statistics on complete historical data"""
    
    if monthly_df is None:
        return None
    
    sku_data = monthly_df[monthly_df['SKU_ID'] == sku_id].sort_values('Date').copy()
    
    if len(sku_data) == 0:
        return None
    
    # Calculate rolling statistics on complete historical data
    sku_data['Rolling_Mean'] = sku_data['Monthly_Sales'].rolling(window=window, min_periods=1).mean()
    sku_data['Rolling_Std'] = sku_data['Monthly_Sales'].rolling(window=window, min_periods=1).std()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Complete Sales History with {window}-Month Rolling Mean', 
            f'{window}-Month Rolling Standard Deviation'
        ),
        vertical_spacing=0.15
    )
    
    # Sales and rolling mean
    fig.add_trace(
        go.Scatter(
            x=sku_data['Date'], 
            y=sku_data['Monthly_Sales'], 
            mode='lines', 
            name='Actual Sales', 
            line=dict(color='blue', width=1),
            opacity=0.6
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sku_data['Date'], 
            y=sku_data['Rolling_Mean'], 
            mode='lines', 
            name=f'{window}-Month Rolling Mean', 
            line=dict(color='red', width=2.5)
        ),
        row=1, col=1
    )
    
    # Rolling std
    fig.add_trace(
        go.Scatter(
            x=sku_data['Date'], 
            y=sku_data['Rolling_Std'], 
            mode='lines', 
            name='Rolling Std Dev', 
            line=dict(color='orange', width=2.5),
            showlegend=True
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sales Volume", row=1, col=1)
    fig.update_yaxes(title_text="Std Deviation", row=2, col=1)
    
    fig.update_layout(
        height=650, 
        title_text=f"SKU {sku_id} - Complete Historical Rolling Statistics"
    )
    
    return fig

# ==================== MAIN APP ====================
def main():
    # Load data
    kpi_df, detailed_df = load_data()
    monthly_df = load_monthly_data()  # Load complete historical data
    
    # ==================== SIDEBAR ====================
    st.sidebar.title("Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Model selection - ARIMA as default
    st.sidebar.subheader("Model Selection")
    all_models = sorted(kpi_df['Model'].unique().tolist())
    
    # Set ARIMA as default if available
    default_model_idx = 0
    if 'ARIMA' in all_models:
        default_model_idx = all_models.index('ARIMA')
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        all_models,
        index=default_model_idx,
        help="ARIMA is the baseline model for comparison"
    )
    
    # Filter by model
    kpi_filtered = kpi_df[kpi_df['Model'] == selected_model].copy()
    detailed_filtered = detailed_df[detailed_df['Model'] == selected_model].copy()
    
    # SKU selection - SINGLE SKU ONLY, default to specific SKUs
    st.sidebar.subheader("SKU Selection")
    all_skus = sorted(kpi_filtered['SKU_ID'].unique().tolist())
    
    # Default SKUs: 1, 2, 6, 10, 5
    default_skus = [1, 2, 5, 6, 10]
    available_default_skus = [s for s in default_skus if s in all_skus]
    
    # Single SKU selection
    selected_sku = st.sidebar.selectbox(
        "Select ONE SKU for Analysis",
        all_skus,
        index=all_skus.index(available_default_skus[0]) if available_default_skus else 0,
        help="Select a single SKU to avoid confusing overlapping graphs"
    )
    
    # Lag period selection - All lags selected by default
    st.sidebar.subheader("Lag Period Selection")
    available_lags = sorted(kpi_filtered['Lag_Period'].unique().tolist())
    
    selected_lags = st.sidebar.multiselect(
        "Select Lag Periods (Months)",
        available_lags,
        default=available_lags,  # All lags selected by default
        help="All lags selected by default. Deselect to focus on specific periods."
    )
    
    if not selected_lags:
        st.sidebar.warning("Please select at least one lag period")
        selected_lags = available_lags
    
    # KPI toggle - All selected by default
    st.sidebar.subheader("KPI Selection")
    kpi_options = {
        'MAE': st.sidebar.checkbox('MAE', value=True),
        'RMSE': st.sidebar.checkbox('RMSE', value=True),
        'SMAPE': st.sidebar.checkbox('SMAPE', value=True),
        'Bias_%': st.sidebar.checkbox('Bias %', value=True),
        'Service_Level_%': st.sidebar.checkbox('Service Level %', value=True),
        'Tracking_Signal': st.sidebar.checkbox('Tracking Signal', value=True)
    }
    
    selected_kpis = [k for k, v in kpi_options.items() if v]
    
    st.sidebar.markdown("---")
    st.sidebar.info("Tip: Single SKU selection prevents graph overlap and makes analysis clearer")
    
    # ==================== MAIN CONTENT ====================
    st.title("Forecast Lag Analysis Dashboard")
    st.markdown("### Interactive Analysis of Forecast Performance Across Different Update Frequencies")
    st.markdown("---")
    
    # ==================== LAG SELECTOR FOR MAIN KPIs ====================
    st.markdown("### 📊 Select Lag Period for KPI Display")
    
    col_lag1, col_lag2, col_lag3 = st.columns([1, 2, 2])
    
    with col_lag1:
        # Single lag selector for main KPI display
        display_lag = st.selectbox(
            "Lag Period for Metrics",
            selected_lags,
            index=selected_lags.index(3) if 3 in selected_lags else 0,
            key='display_lag',
            help="Select ONE lag period to display KPI metrics below"
        )
    
    with col_lag2:
        st.info(f"Displaying KPIs for **{display_lag}-month lag period** for SKU {selected_sku}")
    
    with col_lag3:
        st.markdown(f"*Available lags: {', '.join(map(str, selected_lags))} months*")
    
    st.markdown("---")
    
    # Filter data for selected SKU AND SINGLE LAG for display
    display_kpi_single = kpi_filtered[
        (kpi_filtered['SKU_ID'] == selected_sku) & 
        (kpi_filtered['Lag_Period'] == display_lag)
    ]
    
    # Filter for ALL selected lags (used in other tabs)
    display_kpi = kpi_filtered[
        (kpi_filtered['SKU_ID'] == selected_sku) & 
        (kpi_filtered['Lag_Period'].isin(selected_lags))
    ]
    
    display_detailed = detailed_filtered[
        (detailed_filtered['SKU_ID'] == selected_sku) & 
        (detailed_filtered['Lag_Period'].isin(selected_lags))
    ]
    
    # ==================== KEY METRICS ROW ====================
    st.subheader(f"Key Performance Indicators (Lag: {display_lag} months)")
    
    # Check if we have data for this lag
    if len(display_kpi_single) == 0:
        st.warning(f"No data available for SKU {selected_sku} at {display_lag}-month lag")
    else:
        # Get single row values (not averaged)
        kpi_row = display_kpi_single.iloc[0]
        
        # First row: MAE, RMSE, Bias%, Service Level%
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mae_val = kpi_row['MAE']
            st.metric(
                label="MAE",
                value=f"{mae_val:.2f}",
                help=create_kpi_tooltip('MAE')
            )
            with st.expander("About MAE"):
                st.markdown(create_kpi_tooltip('MAE'))
        
        with col2:
            rmse_val = kpi_row['RMSE']
            st.metric(
                label="RMSE",
                value=f"{rmse_val:.2f}",
                help=create_kpi_tooltip('RMSE')
            )
            with st.expander("About RMSE"):
                st.markdown(create_kpi_tooltip('RMSE'))
        
        with col3:
            bias_val = kpi_row['Bias_%']
            st.metric(
                label="Bias %",
                value=f"{bias_val:.2f}%",
                delta=f"{abs(bias_val):.2f}% from zero",
                delta_color="inverse",
                help=create_kpi_tooltip('Bias_%')
            )
            with st.expander("About Bias %"):
                st.markdown(create_kpi_tooltip('Bias_%'))
        
        with col4:
            service_val = kpi_row['Service_Level_%']
            st.metric(
                label="Service Level %",
                value=f"{service_val:.1f}%",
                help=create_kpi_tooltip('Service_Level_%')
            )
            with st.expander("About Service Level"):
                st.markdown(create_kpi_tooltip('Service_Level_%'))
        
        # Second row: SMAPE, Tracking Signal
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            smape_val = kpi_row['SMAPE']
            st.metric(
                label="SMAPE",
                value=f"{smape_val:.2f}%",
                help=create_kpi_tooltip('SMAPE')
            )
            with st.expander("About SMAPE"):
                st.markdown(create_kpi_tooltip('SMAPE'))
        
        with col6:
            ts_val = kpi_row['Tracking_Signal']
            ts_status = "Good" if abs(ts_val) < 4 else "Check Bias"
            st.metric(
                label="Tracking Signal",
                value=f"{ts_val:.2f}",
                delta=ts_status,
                delta_color="normal" if abs(ts_val) < 4 else "inverse",
                help=create_kpi_tooltip('Tracking_Signal')
            )
            with st.expander("About Tracking Signal"):
                st.markdown(create_kpi_tooltip('Tracking_Signal'))
        
        with col7:
            # Best lag across ALL selected lags
            best_lag = display_kpi.groupby('Lag_Period')['RMSE'].mean().idxmin()
            st.metric(
                label="Best Lag Period",
                value=f"{best_lag} months",
                help="Lag period with lowest average RMSE across all selected lags"
            )
        
        with col8:
            st.metric(
                label="Current Display",
                value=f"{display_lag}m lag",
                help=f"Currently displaying metrics for {display_lag}-month lag period"
            )
    
    st.markdown("---")
    
    # ==================== REORDERED TABS ====================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "SKU Time Series",  # First
        "KPI Trends",  # Second
        "Comparative Analysis",  # Third
        "Model Comparison",  # Fourth
        "Recommendations"  # Last
    ])
    
    # ==================== TAB 1: SKU TIME SERIES (FIRST) ====================
    with tab1:
        st.subheader("SKU-Level Time Series Analysis")
        st.markdown("**Purpose**: Examine complete sales history, actual vs forecast performance, and rolling statistics.")
        
        # Lag selector for time series
        selected_lag_ts = st.selectbox(
            "Select Lag Period for Time Series",
            selected_lags,
            key='lag_ts'
        )
        
        # ===== NEW: Complete Historical View with Forecast Overlay =====
        st.markdown("#### Complete Sales History with Train/Forecast Split")
        
        if monthly_df is not None:
            fig_complete = plot_complete_history_with_forecast(
                monthly_df, 
                display_detailed, 
                selected_sku, 
                selected_lag_ts,
                selected_model
            )
            
            if fig_complete:
                st.plotly_chart(fig_complete, use_container_width=True)
                st.markdown("""
                **What This Shows:**
                - **Gray line**: Historical training data
                - **Green line with dots**: Actual sales in forecast period
                - **Red dashed line**: Model forecast
                - **Pink shaded area**: 95% confidence interval
                - **Blue dotted line**: Marks where forecast begins
                
                This view shows how well the model learned from historical patterns and predicted future sales.
                """)
            else:
                st.warning(f"No complete historical data available for SKU {selected_sku}")
        else:
            st.info("Upload monthly_data.csv to see complete historical view")
        
        st.markdown("---")
        
        # ===== Detailed Forecast View (existing) =====
        st.markdown("#### Detailed Forecast Period View")
        fig_ts = plot_sku_time_series(display_detailed, selected_sku, selected_lag_ts)
        if fig_ts:
            st.plotly_chart(fig_ts, use_container_width=True)
            st.markdown("""
            **Zoomed view of forecast period:**
            - Blue line: Actual sales
            - Red dashed line: Forecasted sales
            - Gray shaded area: 95% confidence interval
            """)
        else:
            st.warning(f"No detailed forecast data available for SKU {selected_sku} at lag {selected_lag_ts}")
        
        st.markdown("---")
        
        # ===== Rolling statistics on COMPLETE data =====
        st.markdown("#### Rolling Statistics Analysis (Complete History)")
        window_size = st.slider("Rolling Window Size (Months)", 2, 12, 3, key='rolling_window')
        
        if monthly_df is not None:
            fig_rolling_complete = plot_rolling_stats_complete(monthly_df, selected_sku, window=window_size)
            if fig_rolling_complete:
                st.plotly_chart(fig_rolling_complete, use_container_width=True)
                st.markdown("""
                **What This Shows:**
                - **Top Panel**: Complete sales history with rolling average showing long-term trends
                - **Bottom Panel**: Rolling standard deviation showing volatility over time
                - **High volatility periods** suggest need for more frequent forecast updates
                - **Trend changes** visible in rolling mean indicate market shifts
                """)
            else:
                st.warning(f"No complete historical data available for SKU {selected_sku}")
        else:
            # Fallback to old method if monthly data not available
            st.info("Showing rolling statistics on forecast period only. Upload monthly_data.csv for complete history.")
            fig_rolling = plot_rolling_stats(display_detailed, selected_sku, window=window_size)
            if fig_rolling:
                st.plotly_chart(fig_rolling, use_container_width=True)
    
    # ==================== TAB 2: KPI TRENDS (SECOND) ====================
    with tab2:
        st.subheader("KPI Performance Trends Across Lag Periods")
        st.markdown(f"**Purpose**: Visualize how each KPI changes for SKU {selected_sku} as lag period increases.")
        
        if not selected_kpis:
            st.warning("Please select at least one KPI from the sidebar.")
        else:
            for kpi in selected_kpis:
                if kpi in display_kpi.columns:
                    with st.container():
                        st.markdown(f"#### {KPI_DEFINITIONS[kpi]['name']}")
                        
                        col_a, col_b = st.columns([3, 1])
                        
                        with col_a:
                            fig = plot_kpi_trends(display_kpi, kpi, selected_sku, selected_lags)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_b:
                            st.markdown("**What This Shows:**")
                            st.markdown(KPI_DEFINITIONS[kpi]['description'])
                            st.markdown("**Interpretation:**")
                            st.markdown(KPI_DEFINITIONS[kpi]['interpretation'])
                        
                        st.markdown("---")
    
    # ==================== TAB 3: COMPARATIVE ANALYSIS (THIRD) ====================
    with tab3:
        st.subheader("Comparative Lag Analysis")
        st.markdown(f"**Purpose**: Compare KPIs across different lag periods for SKU {selected_sku}.")
        
        # Lag selection for comparative analysis
        st.markdown("#### Select Lag Periods to Compare")
        
        col_comp1, col_comp2 = st.columns([2, 3])
        
        with col_comp1:
            comparative_lags = st.multiselect(
                "Select Multiple Lags for Comparison",
                available_lags,
                default=selected_lags,
                key='comparative_lags',
                help="Select multiple lag periods to see how KPIs change"
            )
        
        with col_comp2:
            if comparative_lags:
                st.info(f"Comparing {len(comparative_lags)} lag periods: {', '.join(map(str, comparative_lags))} months")
            else:
                st.warning("Please select at least one lag period to compare")
        
        if not comparative_lags:
            comparative_lags = selected_lags  # Fallback to sidebar selection
        
        # Filter data for comparative lags
        display_kpi_comp = kpi_filtered[
            (kpi_filtered['SKU_ID'] == selected_sku) & 
            (kpi_filtered['Lag_Period'].isin(comparative_lags))
        ]
        
        st.markdown("---")
        
        # Multi-KPI comparison - FIXED LAYOUT
        st.markdown("#### All KPIs Across Selected Lag Periods")
        fig_compare = plot_comparative_lag_analysis(display_kpi_comp, selected_sku)
        st.plotly_chart(fig_compare, use_container_width=True)
        
        st.markdown("""
        **Interpretation Guide:**
        - **Top Left (MAE)**: Lower values indicate better absolute accuracy
        - **Top Right (RMSE)**: Similar to MAE but penalizes large errors more
        - **Bottom Left (SMAPE)**: Scale-independent metric for comparing across SKUs
        - **Bottom Right (Bias%)**: Values near zero indicate balanced forecasting
        """)
        
        st.markdown("---")
        
        # KPI Comparison Table
        st.markdown("#### Detailed KPI Comparison Table")
        comparison_table = display_kpi_comp.groupby('Lag_Period').agg({
            'MAE': 'mean',
            'RMSE': 'mean',
            'SMAPE': 'mean',
            'Bias_%': 'mean',
            'Service_Level_%': 'mean',
            'Tracking_Signal': 'mean'
        }).round(2)
        
        # Style the table
        st.dataframe(
            comparison_table.style.highlight_min(
                axis=0, 
                color='lightgreen', 
                subset=['MAE', 'RMSE', 'SMAPE']
            ).highlight_max(
                axis=0,
                color='lightgreen',
                subset=['Service_Level_%']
            ),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Heatmap for selected KPI
        if selected_kpis:
            st.markdown("#### Heatmap: All SKUs vs Lag Period")
            heatmap_kpi = st.selectbox(
                "Select KPI for Heatmap",
                selected_kpis,
                key='heatmap_kpi'
            )
            
            col_heat_a, col_heat_b = st.columns([3, 1])
            
            with col_heat_a:
                fig_heatmap = plot_kpi_heatmap(kpi_filtered, heatmap_kpi, comparative_lags)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with col_heat_b:
                st.markdown("**What This Shows:**")
                st.markdown(f"Color intensity represents {heatmap_kpi} values across all SKUs. ")
                if heatmap_kpi in ['MAE', 'RMSE', 'SMAPE']:
                    st.markdown("🟢 Green = Better (lower error)")
                    st.markdown("🔴 Red = Worse (higher error)")
                else:
                    st.markdown("Color scale shows metric values across SKUs and lags.")
                st.markdown(f"\n**Comparing**: {', '.join(map(str, comparative_lags))} month lags")
    
    # ==================== TAB 4: MODEL COMPARISON (FOURTH) ====================
    with tab4:
        st.subheader("Model Performance Comparison")
        st.markdown(f"**Purpose**: Compare ALL forecasting models for SKU {selected_sku} at **{display_lag}-month lag**.")
        
        st.info(f"Comparing all available models at **{display_lag}-month lag period** for SKU {selected_sku}")
        
        # Get data for ALL models at the selected display lag
        all_models_data = kpi_df[
            (kpi_df['SKU_ID'] == selected_sku) & 
            (kpi_df['Lag_Period'] == display_lag)
        ]
        
        if len(all_models_data) == 0:
            st.warning(f"No data available for SKU {selected_sku} at {display_lag}-month lag across models")
        else:
            if not selected_kpis:
                st.warning("Please select at least one KPI from the sidebar.")
            else:
                for kpi in selected_kpis:
                    if kpi in all_models_data.columns:
                        col_model_a, col_model_b = st.columns([3, 1])
                        
                        with col_model_a:
                            # Create bar chart comparing all models
                            model_comparison = all_models_data[['Model', kpi]].sort_values(kpi)
                            
                            fig = px.bar(
                                model_comparison,
                                x='Model',
                                y=kpi,
                                title=f'{KPI_DEFINITIONS[kpi]["name"]} by Model (SKU {selected_sku}, {display_lag}m lag)',
                                color=kpi,
                                color_continuous_scale='RdYlGn_r' if kpi in ['MAE', 'RMSE', 'SMAPE'] else 'RdYlGn',
                                text=kpi
                            )
                            
                            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                            fig.update_layout(height=400, showlegend=False)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_model_b:
                            st.markdown(f"**{KPI_DEFINITIONS[kpi]['name']}**")
                            st.markdown(KPI_DEFINITIONS[kpi]['description'])
                            
                            # Show best model for this KPI
                            if kpi in ['MAE', 'RMSE', 'SMAPE']:
                                best_model = model_comparison.iloc[0]['Model']
                                best_value = model_comparison.iloc[0][kpi]
                                st.success(f"Best: {best_model} ({best_value:.2f})")
                            elif kpi == 'Service_Level_%':
                                best_model = model_comparison.iloc[-1]['Model']
                                best_value = model_comparison.iloc[-1][kpi]
                                st.success(f"Best: {best_model} ({best_value:.2f}%)")
            
            st.markdown("---")
            
            # Model Comparison Summary Table
            st.markdown("#### Model Comparison Summary Table")
            st.markdown(f"**Shows**: All models compared at {display_lag}-month lag for SKU {selected_sku}")
            
            summary_cols = ['Model', 'MAE', 'RMSE', 'SMAPE', 'Bias_%', 'Service_Level_%', 'Tracking_Signal']
            available_cols = [col for col in summary_cols if col in all_models_data.columns]
            
            summary_table = all_models_data[available_cols].copy()
            
            # Style the table
            st.dataframe(
                summary_table.style.highlight_min(
                    axis=0, 
                    color='lightgreen', 
                    subset=['MAE', 'RMSE', 'SMAPE']
                ).highlight_max(
                    axis=0,
                    color='lightgreen',
                    subset=['Service_Level_%']
                ),
                use_container_width=True
            )
            
            # Export button
            csv = summary_table.to_csv(index=False)
            st.download_button(
                label="Download Model Comparison",
                data=csv,
                file_name=f"model_comparison_sku_{selected_sku}_lag_{display_lag}m.csv",
                mime="text/csv"
            )
    
    # ==================== TAB 5: RECOMMENDATIONS (LAST) ====================
    with tab5:
        st.subheader("Data-Driven Recommendations")
        
        # AI Recommendation Section at Top
        st.markdown("#### AI-Powered Strategic Recommendations")
        
        col_ai1, col_ai2, col_ai3 = st.columns([2, 2, 3])
        
        with col_ai1:
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                help="Enter your OpenRouter API key to enable AI-powered recommendations",
                key='api_key_rec'
            )
        
        with col_ai2:
            generate_rec = st.button(
                "🔮 Generate AI Recommendation",
                disabled=(api_key == "" or api_key is None),
                use_container_width=True,
                type="primary"
            )
        
        with col_ai3:
            if not api_key or api_key == "":
                st.info("Enter API key to generate AI recommendations")
        
        # Generate and display AI recommendation
        if api_key and api_key != "":
            if generate_rec:
                with st.spinner("Generating AI-powered insights..."):
                    recommendation = generate_ai_recommendation(kpi_filtered, detailed_filtered, api_key)
                    
                    if recommendation and not recommendation.startswith("Error") and not recommendation.startswith("API Error"):
                        st.markdown("---")
                        st.markdown(recommendation)
                    else:
                        st.error(recommendation)
            else:
                st.info("Click the 'Generate AI Recommendation' button above to get AI-powered insights.")
        else:
            st.info("""
            **Enable AI Recommendations:**
            
            1. Get a free API key from [OpenRouter](https://openrouter.ai/)
            2. Enter your API key above
            3. Click 'Generate AI Recommendation'
            
            The AI will analyze your data in simple language and provide actionable recommendations.
            """)
        
        st.markdown("---")
        
        # Rule-based recommendations
        st.markdown("#### Automated Analysis Summary")
        
        # Calculate key insights for selected SKU
        lag_performance = display_kpi.groupby('Lag_Period').agg({
            'MAE': 'mean',
            'RMSE': 'mean',
            'SMAPE': 'mean',
            'Bias_%': 'mean'
        }).round(2)
        
        best_lag_mae = lag_performance['MAE'].idxmin()
        best_lag_rmse = lag_performance['RMSE'].idxmin()
        worst_lag_rmse = lag_performance['RMSE'].idxmax()
        
        # Display insights in columns
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("##### Key Findings")
            st.success(f"**Best Overall Lag**: {best_lag_rmse} months (Lowest RMSE: {lag_performance.loc[best_lag_rmse, 'RMSE']:.2f})")
            st.info(f"**Most Accurate (MAE)**: {best_lag_mae} months (MAE: {lag_performance.loc[best_lag_mae, 'MAE']:.2f})")
            
            # SKU-specific insights
            growing_skus = [1, 2, 3, 4, 5, 8, 9]
            declining_skus = [6, 7, 10]
            
            if selected_sku in growing_skus:
                st.success(f"**SKU {selected_sku}**: Growing SKU - Recommend 1-3 month lags for responsiveness")
            elif selected_sku in declining_skus:
                st.info(f"**SKU {selected_sku}**: Declining SKU - Can use 6-12 month lags for stability")
        
        with rec_col2:
            st.markdown("##### Areas to Monitor")
            
            high_bias = abs(display_kpi['Bias_%'].mean())
            if high_bias > 10:
                st.warning(f"**High Bias Detected**: {high_bias:.2f}%")
                st.markdown("This SKU shows systematic over/under-forecasting")
            
            low_service = display_kpi['Service_Level_%'].mean()
            if low_service < 85:
                st.warning(f"**Low Service Level**: {low_service:.1f}%")
                st.markdown("Consider increasing forecast values or safety stock")
            
            if worst_lag_rmse:
                st.error(f"**Avoid**: {worst_lag_rmse} month lag shows highest errors (RMSE: {lag_performance.loc[worst_lag_rmse, 'RMSE']:.2f})")
        
        st.markdown("---")
        
        # Lag performance table
        st.markdown("#### Lag Period Performance Comparison")
        st.dataframe(
            lag_performance.style.highlight_min(axis=0, color='lightgreen', subset=['MAE', 'RMSE', 'SMAPE']),
            use_container_width=True
        )
        
        st.markdown("---")
        
        # General best practices
        with st.expander("General Best Practices for Lag Selection"):
            st.markdown("""
            ### Understanding Forecast Lag Periods
            
            **Effect of Lag on Forecasts:**
            
            - **1-3 Month Lags**: Lowest errors (RMSE, MAE) - captures recent trends
            - **6 Month Lags**: Errors typically increase - less recent data
            - **12 Month Lags**: Often improves again - captures full seasonal patterns
            
            **SKU-Specific Guidelines:**
            
            **Growing SKUs (1-5, 8-9)**
            - Best with shorter lags (1-3 months)
            - Captures recent market changes
            - More responsive to demand shifts
            
            **Declining SKUs (6, 7, 10)**
            - Perform well with longer lags (6-12 months)
            - More predictable downward trajectory
            - Less sensitive to short-term changes
            
            **Key Takeaway:**
            There's no one-size-fits-all. Fast-moving products need frequent updates (1-3 months). 
            Mature or seasonal products do well with longer windows (12 months). Compare KPIs across 
            lags to find the right balance between responsiveness and stability.
            """)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()