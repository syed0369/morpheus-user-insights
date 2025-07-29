import views
import setup
from load_data import load_combined_data

if __name__ == "__main__":

    setup.setup()
    df = load_combined_data()
    selected_tenants, date_range = setup.setup_sidebar(df)
    # views.insights(selected_tenants)
    filtered_df = setup.filter_data(df, selected_tenants, date_range)
    views.display_activity_chart(filtered_df)
    pivot = views.display_weekly_activity(filtered_df)
    combined_daywise, selected_weeks, available_weeks, select_all = views.display_daily_activity(pivot, filtered_df, selected_tenants)
    views.display_bcg_matrix()
    views.display_top_active_users(combined_daywise)
    views.instance_type_distribution(selected_tenants, date_range, selected_weeks, available_weeks, select_all)
    views.display_tenant_gantt_chart(selected_tenants, date_range, selected_weeks, available_weeks, select_all)
    views.chatbot_ui()
