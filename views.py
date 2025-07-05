import streamlit as st
import pandas as pd
import plotly.express as px
from neo4j import GraphDatabase
import plotly.graph_objects as go
import datetime
from itertools import product
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
# --- Neo4j Connection ---
@st.cache_data
def load_activity_data():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    def fetch_data(tx):
        query = """
        MATCH (t:Tenant)-[:HAS]->(u:User)-[:PERFORMED]->(a:Action)
        RETURN t.name AS tenant, u.username AS username, 
               a.type AS type, a.ts AS ts, a.name AS name, 
               a.message AS message, a.objecttype AS objecttype, a.id AS id
        """
        return pd.DataFrame(tx.run(query).data())

    with driver.session() as session:
        df = session.execute_read(fetch_data)

    driver.close()
    return df

# --- Streamlit UI Setup ---
st.set_page_config(page_title="📊 Morpheus Activity Dashboard", layout="wide")
st.title("📊 Morpheus Activity Dashboard")

# --- Load Data ---
df = load_activity_data()
df["ts"] = pd.to_datetime(df["ts"])
df["date"] = df["ts"].dt.date

# --- Sidebar Filters ---
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

# --- Filtered Data ---
filtered_df = df[
    (df["tenant"].isin(selected_tenants)) &
    (df["date"] >= date_range[0]) &
    (df["date"] <= date_range[1])
]

# --- Activity Timeline with Annotated Latest Activities ---
st.subheader("Tenant Wize Activity Timeline")

fig = px.scatter(
    filtered_df,
    x="ts",
    y="tenant",
    color="tenant",
    hover_data=["message"],
    title="Activity by Tenant Over Time",
    height=500
)
fig.update_traces(marker=dict(size=12))
latest_activity = filtered_df.sort_values("ts", ascending=False).groupby("tenant").first().reset_index()

for _, row in latest_activity.iterrows():
    fig.add_annotation(
        x=row["ts"],
        y=row["tenant"],
        text=f"{row['type']}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#444",
        borderwidth=1,
        font=dict(size=12, color="black"),
    )

st.plotly_chart(fig, use_container_width=True)

# --- Weekly Activity Overview with Tenant Comparison ---
st.subheader("Weekly Activity by Tenants")
weeks_back = st.number_input("Showing data for past N weeks:", min_value=1, max_value=52, value=6, step=1)

filtered_df["week_start"] = filtered_df["ts"] - pd.to_timedelta(filtered_df["ts"].dt.weekday, unit='D')
filtered_df["week_start"] = filtered_df["week_start"].dt.date

latest_week = filtered_df["week_start"].max()
cutoff_week = latest_week - datetime.timedelta(weeks=weeks_back)
df_recent = filtered_df[filtered_df["week_start"] >= cutoff_week]

grouped = df_recent.groupby(["week_start", "tenant"]).size().reset_index(name="activity_count")
pivot = grouped.pivot(index="week_start", columns="tenant", values="activity_count").fillna(0)
pivot = pivot.sort_index()

scatter_fig = px.scatter(
    filtered_df,
    x="ts",
    y="tenant",
    color="tenant"
)
color_map = {trace.name: trace.marker.color for trace in scatter_fig.data}

fig = go.Figure()

for tenant in pivot.columns:
    fig.add_trace(go.Bar(
        x=pivot.index,
        y=pivot[tenant],
        name=tenant,
        marker_color=color_map.get(tenant, None),
        hovertemplate="Week: %{x}<br>Tenant: <b>" + tenant + "</b><br>Count: %{y}<extra></extra>",
    ))

fig.update_layout(
    barmode='group',
    title=f"Weekly Activity Count (Last {weeks_back} Weeks)",
    xaxis_title="Week Starting",
    yaxis_title="Activity Count",
    xaxis=dict(type='category', tickangle=-45),
    height=600,
    legend_title_text="Tenant",
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

# --- Daily activity chart ---
available_weeks = pivot.index.astype(str).tolist()
st.subheader("🔍 View Daily Activity Across Tenants")
select_all = st.checkbox("Select All Weeks", value=False)

if select_all:
    selected_weeks = available_weeks
else:
    selected_weeks = st.multiselect("Select Week(s)", available_weeks, default=available_weeks[-1:])

combined_daywise = pd.DataFrame()

for selected_week in selected_weeks:
    start_week = datetime.datetime.strptime(selected_week, "%Y-%m-%d").date()
    end_week = start_week + datetime.timedelta(days=7)

    df_week = filtered_df[
        (filtered_df["tenant"].isin(selected_tenants)) &
        (filtered_df["date"] >= start_week) &
        (filtered_df["date"] < end_week)
    ].copy()

    df_week["weekday"] = pd.to_datetime(df_week["date"]).dt.day_name()
    df_week["week"] = selected_week

    combined_daywise = pd.concat([combined_daywise, df_week])

if not combined_daywise.empty:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    all_pairs = pd.DataFrame(
        list(product(weekday_order, selected_tenants)),
        columns=["weekday", "tenant"]
    )

    grouped = (
        combined_daywise.groupby(["weekday", "tenant"])
        .size()
        .reset_index(name="count")
    )

    # --- Merge actual counts with all weekday–tenant pairs ---
    merged = pd.merge(all_pairs, grouped, on=["weekday", "tenant"], how="left").fillna(0)
    merged["count"] = merged["count"].astype(int)

    fig_combined = px.bar(
        merged,
        x="weekday",
        y="count",
        color="tenant",
        barmode="group",
        category_orders={"weekday": weekday_order},
        title=f"Weekday Activity for Selected Week(s): {', '.join(selected_weeks)}",
        labels={"count": "Activity Count", "weekday": "Weekday", "tenant": "Tenant"},
        height=500
    )
    fig_combined.update_layout(
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font_color="white"
    )
    st.plotly_chart(fig_combined, use_container_width=True)

else:
    st.info("No daily activity data for selected tenants and weeks.")

# --- Top N Active Users per Tenant ---
st.subheader("Top N Active Users per Tenant")
top_n = st.number_input("Select how many top users to show per tenant:", min_value=1, max_value=10, value=2, step=1)

user_activity = (
    combined_daywise.groupby(["tenant", "username"])
    .size()
    .reset_index(name="action_count")
)

top_users = (
    user_activity.sort_values(["tenant", "action_count"], ascending=[True, False])
    .groupby("tenant")
    .head(top_n)
)

for tenant in top_users["tenant"].unique():
    tenant_users = top_users[top_users["tenant"] == tenant]
    with st.expander(f"Tenant: {tenant}"):
        for _, row in tenant_users.iterrows():
            user_label = f"👤 {row['username']} ({row['action_count']} actions)"
            key = f"expand_{tenant}_{row['username']}"
            if st.button(f"👤 {row['username']} ({row['action_count']} actions)", key=key + "_btn"):
                st.session_state[key] = not st.session_state.get(key, False)
            if st.session_state.get(key, False):
                user_df = combined_daywise[
                    (combined_daywise["tenant"] == tenant) & 
                    (combined_daywise["username"] == row["username"])
                ]

                user_df["weekday"] = user_df["ts"].dt.day_name()

                weekday_counts = (
                    user_df.groupby("weekday")
                    .size()
                    .reset_index(name="count")
                )

                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                full_weekdays = pd.DataFrame({"weekday": weekday_order})
                full_counts = pd.merge(full_weekdays, weekday_counts, on="weekday", how="left").fillna(0)
                full_counts["count"] = full_counts["count"].astype(int)

                fig_user = px.bar(
                    full_counts,
                    x="weekday",
                    y="count",
                    title=f"📅 Weekday Activity for {row['username']} in {tenant}",
                    category_orders={"weekday": weekday_order},
                    labels={"count": "Action Count", "weekday": "Weekday"},
                    height=400
                )
                fig_user.update_layout(
                    plot_bgcolor="#111111",
                    paper_bgcolor="#111111",
                    font_color="white"
                )
                st.plotly_chart(fig_user, use_container_width=True)
