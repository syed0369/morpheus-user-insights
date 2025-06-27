import streamlit as st
import pandas as pd
import plotly.express as px
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from datetime import datetime

# --- PostgreSQL Connection ---
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'morpheuslogs'
DB_USER = 'postgres'
DB_PASS = quote_plus('Umair@123')  # Use env variable in production

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
        query = """
        SELECT 
            il.tenant AS "Tenant",
            il.created_by AS "User",
            il.instance_type AS "VM_Type",
            al.status AS "Action",
            al.startDate AS "Timestamp",
            il.used_cpu AS "CPU_Usage",
            al.cost AS "Cost"
        FROM activity_logs al
        JOIN instance_logs il ON al.instance_id = il.instance_id;
        """
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# --- Main Dashboard ---
st.set_page_config(page_title="Morpheus Dashboard", layout="wide")
st.title("☁️ Morpheus User Pattern Dashboard")
st.markdown("Analyze VM provisioning behavior, usage trends, and user actions.")

# --- Load and Filter Data ---
data = load_data()
if data.empty:
    st.stop()

# Sidebar filters
st.sidebar.header("📊 Filters")
tenant_filter = st.sidebar.multiselect("Tenant", options=data["Tenant"].unique(), default=data["Tenant"].unique())
user_filter = st.sidebar.multiselect("User", options=data["User"].unique(), default=data["User"].unique())
action_filter = st.sidebar.multiselect("Action", options=data["Action"].unique(), default=data["Action"].unique())

# Date range
data["Date"] = pd.to_datetime(data["Timestamp"]).dt.date
min_date = data["Date"].min()
max_date = data["Date"].max()
date_range = st.sidebar.slider("Date Range", min_value=min_date, max_value=max_date, value=(min_date, max_date), format="YYYY-MM-DD")

# Apply filters
filtered_data = data[
    (data["Tenant"].isin(tenant_filter)) &
    (data["User"].isin(user_filter)) &
    (data["Action"].isin(action_filter)) &
    (data["Date"] >= date_range[0]) &
    (data["Date"] <= date_range[1])
]

# --- Summary ---
st.subheader("🔢 Summary Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Actions", len(filtered_data))
col2.metric("Average CPU Usage", round(filtered_data["CPU_Usage"].mean(), 2))

# --- Charts ---
st.subheader("📊 VM Actions Over Time")
fig1 = px.histogram(filtered_data, x="Timestamp", color="Action", barmode="group", title="Actions per Day")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("⚙️ CPU Usage by VM Type")
fig2 = px.box(filtered_data, x="VM_Type", y="CPU_Usage", color="VM_Type", title="CPU Usage Distribution")
st.plotly_chart(fig2, use_container_width=True)

# --- Raw Data ---
st.subheader("📄 Raw Data Table")
st.dataframe(filtered_data.reset_index(drop=True))
