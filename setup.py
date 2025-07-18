import streamlit as st
import datetime

# --- Utility Functions ---
def inject_tooltip_css():
    st.markdown("""
        <style>
            .info-icon {
                position: relative;
                display: inline-block;
                cursor: pointer;
                font-weight: bold;
                color: #00BFFF;
            }

            .info-icon .tooltiptext {
                visibility: hidden;
                width: 300px;
                background-color: #333;
                color: #fff;
                text-align: left;
                border-radius: 6px;
                padding: 8px 12px;
                position: absolute;
                z-index: 1;
                bottom: 120%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 13px;
            }

            .info-icon:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
        </style>
    """, unsafe_allow_html=True)

def graph_heading_with_info(title: str, tooltip: str):
    st.markdown(f"""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <h3 style='margin: 0;'>{title}</h3>
            <div class='info-icon'>&#9432;
                <span class='tooltiptext'>{tooltip}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- Streamlit UI Setup ---
def setup():
    st.set_page_config(page_title="📊 Morpheus Activity Dashboard", layout="wide")
    st.title("📊 Morpheus Activity Dashboard")
    inject_tooltip_css()
    
# --- Sidebar Filters ---
def setup_sidebar(df):
    st.sidebar.header("Filters")
    tenant_options = df["tenant"].unique().tolist()
    selected_tenants = st.sidebar.multiselect("Select Tenant(s)", tenant_options, default=tenant_options)

    min_date = datetime.date(2025, 1, 1)
    max_date = datetime.date.today()

    date_range = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    return selected_tenants, date_range

# --- Filtered Data ---
def filter_data(df, selected_tenants, date_range):
    filtered_df = df[
        (df["tenant"].isin(selected_tenants)) &
        (df["date"] >= date_range[0]) &
        (df["date"] <= date_range[1])
    ].copy()
    return filtered_df
