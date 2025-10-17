import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# --- Configuration (Matching the EPM project structure) ---
# NOTE: Using the generic filename for platform compatibility
FILE_PATH = r"C:\Users\acer\Downloads\Enterprise_Sustainable Power Evaluation_Dataset.csv"
TARGET_COLUMN = 'Emissions Intensity (kg CO₂ per MWh)'
TARGET_CLEANED = 'Emissions_Intensity_kg_CO2_per_MWh'

# --- Robust Column Cleaning Function (Replicating ML analysis success) ---
@st.cache_data
def standardize_column_name_robust(col_name):
    """Aggressively cleans and standardizes a column name."""
    cleaned = col_name
    
    # 1. Handle known encoding/symbol issues (e.g., CO₂ -> CO2)
    cleaned = cleaned.replace('â\x82\x82', '2') 
    
    # 2. Replace all non-alphanumeric, non-space characters with an underscore
    cleaned = re.sub(r'[^A-Za-z0-9\s_]', '', cleaned)
    
    # 3. Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')
    
    # 4. Collapse multiple underscores and strip leading/trailing ones
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    
    # 5. FIX: Ensure the target column is mapped correctly
    if 'Emissions_Intensity_kg_CO_per_MWh' in cleaned:
        return TARGET_CLEANED
        
    return cleaned

# --- Data Loading and Cleaning ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """Loads, cleans, and prepares the dataset."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please ensure the CSV is accessible.")
        return pd.DataFrame()

    # Apply the single, robust standardization function to all columns
    df.columns = [standardize_column_name_robust(col) for col in df.columns]

    # Convert to pseudo-time-series by sorting
    df_ts = df.sort_values(by='Company_ID').reset_index(drop=True)
    
    # Create a simple time index for plotting the simulated time-series
    df_ts['Time_Step'] = df_ts.index
    
    # Ensure key columns are float for plotting
    key_cols = [TARGET_CLEANED, 'Renewable_Energy_Share', 'Sustainability_Score', 'Net_Profit_Margin']
    for col in key_cols:
        if col in df_ts.columns:
            df_ts[col] = pd.to_numeric(df_ts[col], errors='coerce')
            
    df_ts = df_ts.dropna()
    return df_ts

df = load_and_preprocess_data(FILE_PATH)

# Check if data loaded successfully
if df.empty:
    st.stop()

# --- Dashboard Layout and Styling ---
st.set_page_config(
    layout="wide", 
    page_title="Holistic Horizon EPM Prediction Dashboard", 
    initial_sidebar_state="collapsed"
)

# Apply custom CSS for dark mode look and clean typography
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117; /* Dark background */
        color: #c9d1d9; /* Light text */
    }
    .stPlotly, .stAlert {
        border-radius: 8px;
        padding: 10px;
        background-color: #161b22; /* Slightly lighter container for contrast */
    }
    h1, h2, h3 {
        color: #58a6ff; /* Blue for headings */
    }
    .st-cd, .st-ce {
        background-color: #161b22;
        border-radius: 8px;
        padding: 10px;
    }
    .st-emotion-cache-1629p8f { /* Targetting metrics container for better alignment */
        gap: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


st.title("🌌 Holistic Horizon EPM Dashboard: Multi-Modal Integrated Prediction")
st.markdown("### Fusing Quantitative Time-Series Data with Qualitative LLM Context")

st.info("""
    This dashboard provides a visualization of the Enterprise Performance Management ($\text{EPM}$) metrics used for training the Hybrid Fusion Model. The core objective is to reduce prediction error for **Emissions Intensity** by incorporating **simulated narrative context** ($\text{LLM}$ embeddings).
""")

# --- 1. KPI Overview (Gauges and Metrics) ---
st.markdown("---")
st.subheader("Key Performance Indicators (EPM Snapshot)")

if TARGET_CLEANED in df.columns:
    avg_emissions = df[TARGET_CLEANED].mean()
    avg_sustainability = df['Sustainability_Score'].mean()
    avg_renewable = df['Renewable_Energy_Share'].mean()
    avg_profit = df['Net_Profit_Margin'].mean()

    col1, col2, col3, col4 = st.columns(4)

    # Metric 1: Emissions (Goal: Lower)
    col1.metric("Avg. Emissions Intensity", f"{avg_emissions:.2f} kg/MWh", delta_color="inverse")

    # Metric 2: Net Profit Margin
    col2.metric("Avg. Net Profit Margin", f"{avg_profit:.2f} %", delta=f"{df['Net_Profit_Margin'].std():.2f} Std Dev")

    # Metric 3: Sustainability Score (Gauge)
    with col3:
        fig_sustainability = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_sustainability,
            title = {'text': "Avg. Sustainability Score", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "#58a6ff"},
                'steps': [
                    {'range': [0, 60], 'color': "red"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 85}}
        ))
        fig_sustainability.update_layout(height=200, margin=dict(t=50, b=0, l=10, r=10), template="plotly_dark")
        st.plotly_chart(fig_sustainability, use_container_width=True)

    # Metric 4: Renewable Share (Gauge)
    with col4:
        fig_renewable = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = avg_renewable,
            title = {'text': "Avg. Renewable Share", 'font': {'size': 14}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': "#2A803B"}, # Darker green for energy
                'steps': [
                    {'range': [0, 25], 'color': "red"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "green"}],
                'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 60}}
        ))
        fig_renewable.update_layout(height=200, margin=dict(t=50, b=0, l=10, r=10), template="plotly_dark")
        st.plotly_chart(fig_renewable, use_container_width=True)


# --- 2. Dual-Axis Time-Series Trend ---
st.markdown("---")
st.subheader("Quantitative Path Visualization: Simulated EPM Trend")

tab1, tab2 = st.tabs(["Dual-Axis Trend (Target vs. Driver)", "Feature Distributions"])

with tab1:
    fig_ts = go.Figure()

    # Emissions (Primary Axis) - Orange/Red for warning/emissions
    fig_ts.add_trace(go.Scatter(
        x=df['Time_Step'], 
        y=df[TARGET_CLEANED], 
        mode='lines', 
        name=TARGET_COLUMN, 
        yaxis='y1',
        line=dict(color='#ff7f0e', width=3)
    ))

    # Renewable Energy Share (Secondary Axis) - Blue/Green for progress
    fig_ts.add_trace(go.Scatter(
        x=df['Time_Step'], 
        y=df['Renewable_Energy_Share'], 
        mode='lines', 
        name='Renewable Energy Share (%)', 
        yaxis='y2',
        line=dict(color='#1f77b4', dash='dash', width=2)
    ))

    fig_ts.update_layout(
        title='Simulated EPM Trend: Emissions Intensity (Target) vs. Renewable Share (Feature)',
        xaxis_title='Simulated Time Step (Company Index)',
        yaxis=dict(
            title=TARGET_COLUMN,
            titlefont=dict(color='#ff7f0e'),
            tickfont=dict(color='#ff7f0e'),
            gridcolor='#161b22'
        ),
        yaxis2=dict(
            title='Renewable Energy Share (%)',
            titlefont=dict(color='#1f77b4'),
            tickfont=dict(color='#1f77b4'),
            overlaying='y',
            side='right',
            gridcolor='#161b22'
        ),
        height=550,
        template="plotly_dark"
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    st.markdown("""
        *Observation:* This dual-axis chart shows the sequential data fed to the $\text{LSTM}$ model. The **Hybrid Model** uses this history *plus* the **Narrative Embedding** corresponding to the prediction step to capture non-linear market/policy impacts.
    """)

with tab2:
    selected_feature = st.selectbox(
        'Select a Feature to view its Distribution:',
        options=[
            'Revenue_USD', 'Net_Profit_Margin', 'Energy_Efficiency', 
            'Sustainability_Score', TARGET_CLEANED
        ]
    )
    
    if selected_feature in df.columns:
        fig_dist = px.histogram(
            df, 
            x=selected_feature, 
            title=f'Distribution of {selected_feature.replace("_", " ")}',
            color_discrete_sequence=['#5D9C3E'],
            template="plotly_dark"
        )
        fig_dist.update_layout(height=450)
        st.plotly_chart(fig_dist, use_container_width=True)

# --- 3. Feature Relationship (Scatter Plot) ---
st.markdown("---")
st.subheader("Feature Correlation: Sustainability vs. Emissions")

if 'Sustainability_Score' in df.columns:
    fig_scatter = px.scatter(
        df, 
        x='Sustainability_Score', 
        y=TARGET_CLEANED, 
        color='Net_Profit_Margin', 
        size='Revenue_USD', # Use Revenue to denote company size/impact
        hover_data=['Company_ID'],
        title=f'Emissions Intensity vs. Sustainability Score, Colored by Profit Margin',
        labels={
            TARGET_CLEANED: TARGET_COLUMN,
            'Sustainability_Score': 'Overall Sustainability Score (0-100)',
            'Net_Profit_Margin': 'Net Profit Margin (%)'
        },
        color_continuous_scale=px.colors.sequential.Viridis,
        template='plotly_dark'
    )
    fig_scatter.update_layout(height=600, coloraxis_colorbar=dict(title="Profit Margin %"))
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("""
        *Insight:* Outliers in this plot—companies with high scores but high emissions, or low scores but high profit—are where the **Narrative Path** is most crucial. The qualitative context can explain these non-linear relationships, which a pure time-series model would struggle to capture.
    """)


# --- 4. Project Interpretation and Architecture ---
st.markdown("---")
st.subheader("Hybrid Model Architecture: Fusing Whispers to Roars")

st.markdown("""
    The "Holistic Horizon" model is built on a **Dual-Input Fusion Architecture** to achieve superior $\text{EPM}$ prediction:

    1.  **Quantitative Path (Time-Series $\text{LSTM}$):** Learns the temporal dependencies and patterns inherent in the numerical $\text{KPIs}$ (e.g., historical revenue, efficiency, and emissions).
    2.  **Qualitative Path ($\text{Dense Network}$):** Processes the **Narrative Embedding Vector** (simulated $\text{LLM}$ output) to capture the semantic context, such as policy shifts, strategic management decisions, or unplanned operational events.
    3.  **Fusion:** The feature vectors from both paths are **concatenated** at a bottleneck layer, allowing the model to learn combined weights and predict the target based on **both historical trends and qualitative context**.

    This integration is why the Hybrid Model is expected to outperform the pure $\text{LSTM}$ baseline.
""")

st.code(
    """
    # Conceptual Keras Fusion
    ts_input = Input(shape=(SEQUENCE_LENGTH, num_ts_features))
    narrative_input = Input(shape=(EMBEDDING_DIMENSION,))

    ts_output = LSTM_path(ts_input)            # Quantitative feature vector (e.g., 16 units)
    narrative_output = Dense_path(narrative_input) # Qualitative feature vector (e.g., 16 units)

    fusion = Concatenate()([ts_output, narrative_output])
    prediction = Dense(1)(fusion)
    """
)
