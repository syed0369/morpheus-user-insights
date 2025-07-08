import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from itertools import product
import json
from load_data import load_activity_data, fetch_instance_counts, fetch_run_data, fetch_execution_data, fetch_temporal_activity_data, get_temporal_insights_from_ai, prepare_llm_friendly_json

# --- Streamlit UI Setup ---
def setup():
    st.set_page_config(page_title="📊 Morpheus Activity Dashboard", layout="wide")
    st.title("📊 Morpheus Activity Dashboard")

# --- Load Data ---
def load_data():
    df = load_activity_data()
    exec_df = fetch_execution_data()

    # Normalize both to have similar fields for merging
    df["source"] = "Action"
    exec_df["source"] = "Execution"
    exec_df["type"] = exec_df["process_type"]
    exec_df["message"] = "Job: " + exec_df["job_name"] + " | Status: " + exec_df["status"]
    exec_df = exec_df[["tenant", "username", "type", "ts", "message", "source"]]

    df = df[["tenant", "username", "type", "ts", "message", "source"]]

    combined_df = pd.concat([df, exec_df], ignore_index=True)
    combined_df["date"] = combined_df["ts"].dt.date
    return combined_df

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

# --- Activity Timeline with Annotated Latest Activities ---
def display_activity_chart(filtered_df):
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
def display_weekly_activity(filtered_df):
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
    return pivot

# --- Daily activity chart ---
def display_daily_activity(pivot, filtered_df, selected_tenants):
    available_weeks = pivot.index.astype(str).tolist()
    st.subheader("View Daily Activity Across Tenants")
    select_all = st.checkbox("Select All Weeks", value=True)

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
    return combined_daywise, selected_weeks, available_weeks, select_all

# --- Top N Active Users per Tenant ---
def display_top_active_users(combined_daywise):
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
                        title=f"Weekday Activity for {row['username']} in {tenant}",
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


def instance_type_distribution(selected_tenants):
    
    instance_df = fetch_instance_counts()

    # Show filtered instance count summary
    filtered_instance_df = instance_df[instance_df["tenant"].isin(selected_tenants)]
    if not filtered_instance_df.empty:
        instance_count_summary = filtered_instance_df.groupby("tenant")["instance_id"].nunique().reset_index()
        instance_count_summary.columns = ["Tenant", "Total Instances"]
        st.markdown("### Total Instances per Tenant")
        st.dataframe(instance_count_summary, use_container_width=True)

        # --- Instance Type Distribution ---
        st.subheader("Total Instance Type Distribution")
        
        instance_type_counts = (
            filtered_instance_df.groupby(["tenant", "instance_type"])["instance_id"]
            .nunique()
            .reset_index(name="count")
        )

        total_by_tenant = instance_type_counts.groupby("tenant")["count"].transform("sum")
        instance_type_counts["percent"] = (instance_type_counts["count"] / total_by_tenant * 100).round(2)

        view_mode = st.radio("Select View Mode:", ["Absolute Count", "Percentage"])
        y_col = "count" if view_mode == "Absolute Count" else "percent"

        fig_type = px.bar(
            instance_type_counts,
            x="instance_type",
            y=y_col,
            color="tenant",
            barmode="group",
            labels={
                "instance_type": "Instance Type",
                "count": "Instance Count",
                "percent": "Percentage",
                "tenant": "Tenant"
            },
            title=f"Instance Types per Tenant ({view_mode})",
            height=500
        )

        fig_type.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font_color="white"
        )

        st.plotly_chart(fig_type, use_container_width=True)

# --- Gantt Chart ---
def display_tenant_gantt_chart(selected_tenants, date_range, selected_weeks, available_weeks, select_all):

    st.subheader("Tenant-Level Gantt Chart of Runs (Avg CPU %) Per Week")

    runs_df = fetch_run_data()
    instance_df = fetch_instance_counts()

    if not runs_df.empty:
        runs_df = runs_df[runs_df["tenant"].isin(selected_tenants)].copy()
        runs_df["week_start"] = runs_df["start"] - pd.to_timedelta(runs_df["start"].dt.weekday, unit="D")
        runs_df["week_start"] = runs_df["week_start"].dt.date.astype(str)

        week_filter = selected_weeks if not select_all else available_weeks
        filtered_runs = runs_df[
            (runs_df["week_start"].isin(week_filter)) &
            (
                ((runs_df["start"].dt.date >= date_range[0]) & (runs_df["start"].dt.date <= date_range[1])) |
                ((runs_df["end"].dt.date >= date_range[0]) & (runs_df["start"].dt.date <= date_range[1]))
            )
        ].copy()

        if not filtered_runs.empty:
            tenant_instance_counts = (
                instance_df.groupby("tenant")["instance_id"]
                .nunique()
                .reset_index()
                .rename(columns={"instance_id": "total_instances"})
            )

            cpu_summary = (
                filtered_runs.groupby(["tenant", "week_start"])
                .agg(
                    total_cpu=("avg_cpu", "sum"),
                    start=("start", "min"),
                    end=("end", "max"),
                    running_instances=("instance_id", pd.Series.nunique)
                )
                .reset_index()
            )

            tenant_summary = pd.merge(cpu_summary, tenant_instance_counts, on="tenant", how="left")
            tenant_summary["avg_cpu"] = (
                tenant_summary["total_cpu"] / tenant_summary["total_instances"]
            ).fillna(0)

            fig = px.timeline(
                tenant_summary,
                x_start="start",
                x_end="end",
                y="tenant",
                color="avg_cpu",
                color_continuous_scale="Viridis",
                labels={"avg_cpu": "Avg CPU (%)"},
                title="Tenant Run Activity Timeline (Avg CPU % per Week)",
                hover_data={
                    "start": True,
                    "end": True,
                    "avg_cpu": True,
                    "total_instances": True,
                    "running_instances": True
                }
            )

            fig.update_yaxes(autorange="reversed")
            fig.update_layout(
                height=600,
                plot_bgcolor="#111111",
                paper_bgcolor="#111111",
                font_color="white",
                coloraxis_colorbar=dict(title="Avg CPU %")
            )        

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No run data available for selected filters.")
    else:
        st.warning("No run data found in the database.")

def insights(selected_tenants):
    with st.expander("🧠 Tenant-wise Temporal Behavior Insights (AI Generated)"):
        tenants = selected_tenants if selected_tenants else None
        temp_df = fetch_temporal_activity_data(tenants)

        if temp_df.empty:
            st.warning("No temporal activity data found for the selected tenants.")
            return

        tenant_payloads = prepare_llm_friendly_json(temp_df)

        for tenant_name, payload in tenant_payloads.items():
            payload_json = json.dumps(payload, indent=2)

            with st.spinner(f"Analyzing behavior for {tenant_name}..."):
                insight = get_temporal_insights_from_ai(payload_json)

            with st.expander(f"🔹 {tenant_name} Insights"):
                st.markdown(insight)
