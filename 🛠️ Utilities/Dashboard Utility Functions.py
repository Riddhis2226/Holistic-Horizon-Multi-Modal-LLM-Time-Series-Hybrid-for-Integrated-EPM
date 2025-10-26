import pandas as pd
import streamlit as st

# --- Data Cleaning and Loading ---

def clean_col_name(col):
    """Cleans column names for code compatibility and fixes specific encoding issues."""
    col = col.replace(" (USD)", "_USD").replace(" (%)", "_Pct").replace(" (tons/year)", "_TonsPerYear")
    # Correct the specific encoding issue for the Emissions column
    col = col.replace(" (kg COâ‚‚ per MWh)", "_kgCO2PerMWh").replace(" ", "_").replace("(", "").replace(")", "").replace("COâ‚‚", "CO2")
    return col

@st.cache_data
def load_epm_data():
    """Loads the original EPM data, cleans column names, and creates a performance group."""
    file_name = "Enterprise_Sustainable Power Evaluation_Dataset.csv"
    
    try:
        df = pd.read_csv(file_name, index_col="Company_ID")
        df.columns = [clean_col_name(col) for col in df.columns]

        # Create a simplified 'Performance_Group' for visualization grouping
        df['Performance_Group'] = pd.cut(df['Sustainability_Score'],
                                         bins=[0, 30, 45, df['Sustainability_Score'].max() + 1],
                                         labels=['Needs Improvement', 'Solid Performer', 'Top Tier'],
                                         right=False)
        return df
    except FileNotFoundError:
        st.error(f"Error: Required file '{file_name}' not found.")
        st.info("Please ensure 'Enterprise_Sustainable Power Evaluation_Dataset.csv' is in the same folder as this script.")
        return None
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# --- Dynamic Analysis Functions ---

def analyze_and_present_insights(df):
    """Generates dynamic, data-driven insights based on the dataset."""
    
    # 1. R&D vs Innovation correlation
    corr_rd_innov = df['Investment_in_R&D_Pct'].corr(df['Innovation_Index'])
    
    # 2. Top and bottom performers
    top_company = df['Sustainability_Score'].idxmax()
    top_score = df['Sustainability_Score'].max()
    
    bottom_company = df['Sustainability_Score'].idxmin()
    bottom_score = df['Sustainability_Score'].min()
    
    # 3. Renewable Energy adoption trend
    high_renewable = (df['Renewable_Energy_Share_Pct'] > 75).sum()
    
    insight_text = f"""
        <h3 class='text-xl font-bold mb-2 text-white'>🔎 Data-Driven Insights</h3>
        <ul style='list-style-type: disc; margin-left: 20px; color: #ccc;'>
            <li style='margin-bottom: 5px;'>
                **R&D vs Innovation:** The correlation between Investment in R&D and Innovation Index is **{corr_rd_innov:.2f}**. 
                This indicates a {"strong positive" if corr_rd_innov > 0.5 else "moderate to weak"} relationship.
            </li>
            <li style='margin-bottom: 5px;'>
                **Sustainability Extremes:** Top performer: **{top_company}** ({top_score:.2f}). Lowest performer: **{bottom_company}** ({bottom_score:.2f}).
            </li>
            <li style='margin-bottom: 5px;'>
                **Renewable Adoption:** **{high_renewable}** enterprises have achieved over 75% Renewable Energy Share.
            </li>
        </ul>
    """
    st.markdown(f"<div class='dynamic-insight'>{insight_text}</div>", unsafe_allow_html=True)
