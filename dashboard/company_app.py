"""Streamlit dashboard for Company-level Earth Task Telemetry."""
import os

import streamlit as st
import pandas as pd
import requests
import plotly.express as px


# Configuration: prioritize Streamlit secrets, then env var, then localhost fallback
def get_backend_url():
    """Get backend URL from secrets, env var, or fallback to localhost."""
    try:
        if hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"].rstrip("/")
    except Exception:
        pass
    env_url = os.getenv("BACKEND_URL") or os.getenv("API_BASE")
    if env_url:
        return env_url.rstrip("/")
    return "http://127.0.0.1:8000"


API_BASE_URL = get_backend_url()

st.set_page_config(
    page_title="Company Dashboard - Earth Task Telemetry",
    page_icon="🏢",
    layout="wide",
)

st.title("🏢 Company Dashboard")


def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        data = response.json()
        return response.status_code == 200 and (data.get("ok") or data.get("status") == "ok")
    except requests.exceptions.RequestException:
        return False


def get_company_info(token: str):
    """Fetch company info using dashboard token."""
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard/company", params={"t": token}, timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            return {"error": "Invalid dashboard token", "status_code": 403}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "status_code": response.status_code}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_company_summary(token: str):
    """Fetch company aggregate stats."""
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard/summary", params={"t": token}, timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            return {"error": "Insufficient players for anonymity", "status_code": 403}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "status_code": response.status_code}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_company_timeseries(token: str, bucket: str = "day"):
    """Fetch company time series data."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/dashboard/timeseries",
            params={"t": token, "bucket": bucket},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403:
            return {"error": "Insufficient players for anonymity", "status_code": 403}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "status_code": response.status_code}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# Get token from URL query param
query_params = st.query_params
token = query_params.get("t", None)

# Backend status
backend_ok = check_backend_health()

# Sidebar
st.sidebar.header("Dashboard Status")
if backend_ok:
    st.sidebar.markdown("**Backend:** :green_circle: Connected")
    st.sidebar.caption(f"`{API_BASE_URL}`")
else:
    st.sidebar.markdown("**Backend:** :red_circle: Unreachable")
    st.sidebar.caption(f"`{API_BASE_URL}`")

# Main content
if not token:
    st.error("No dashboard token provided.")
    st.markdown("""
    **Access Required**

    This dashboard requires a valid company token. Please use the URL provided by your administrator:

    ```
    https://your-dashboard-url/?t=YOUR_TOKEN
    ```

    If you believe you should have access, contact your system administrator.
    """)
elif not backend_ok:
    st.warning("Backend is currently unreachable. Please try again later.")
else:
    # Validate token and get company info
    company_info = get_company_info(token)

    if "error" in company_info:
        if company_info.get("status_code") == 403:
            st.error("Invalid dashboard token. Access denied.")
            st.markdown("Please verify your token URL is correct.")
        else:
            st.error(f"Error: {company_info['error']}")
    else:
        # Token is valid
        company_id = company_info["company_id"]
        n_players = company_info["n_players"]
        has_sufficient = company_info["has_sufficient_data"]

        st.sidebar.divider()
        st.sidebar.markdown(f"**Company ID:** `{company_id[:8]}...`")
        st.sidebar.markdown(f"**Players:** {n_players}")

        if not has_sufficient:
            st.warning(company_info["message"])
            st.markdown("""
            **Why can't I see the data?**

            To protect individual privacy, company aggregates are only shown when there are
            at least 5 unique players. This ensures that individual performance cannot be
            identified from the aggregate data.

            Once more players complete sessions, the dashboard will automatically show aggregates.
            """)
        else:
            # Fetch and display summary
            summary = get_company_summary(token)

            if "error" in summary:
                st.error(f"Error loading summary: {summary['error']}")
            else:
                st.subheader("Company Overview")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Players", summary["n_players"])
                with col2:
                    st.metric("Total Sessions", summary["n_sessions"])
                with col3:
                    avg_bps = summary["avg_brain_performance_score"]
                    st.metric("Avg Brain Performance Score", f"{avg_bps:.2%}")
                with col4:
                    avg_burden = summary["avg_repetition_burden"]
                    st.metric("Avg Repetition Burden", f"{avg_burden:.2f}")

                st.divider()

                # Time series chart
                st.subheader("Brain Performance Score Over Time")

                # Bucket selector
                bucket_type = st.radio(
                    "Time Bucket",
                    options=["day", "week"],
                    horizontal=True,
                    index=0,
                )

                timeseries = get_company_timeseries(token, bucket_type)

                if "error" in timeseries:
                    st.error(f"Error loading timeseries: {timeseries['error']}")
                elif not timeseries.get("buckets"):
                    st.info("No time series data available yet.")
                else:
                    # Convert to DataFrame
                    df = pd.DataFrame(timeseries["buckets"])
                    df["bucket_start"] = pd.to_datetime(df["bucket_start"])

                    # Brain Performance Score chart
                    fig_bps = px.line(
                        df,
                        x="bucket_start",
                        y="avg_brain_performance_score",
                        markers=True,
                        title="Average Brain Performance Score",
                        color_discrete_sequence=["#7ed957"],
                    )
                    fig_bps.update_yaxes(range=[0, 1.1], tickformat=".0%")
                    fig_bps.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Brain Performance Score"
                    )
                    st.plotly_chart(fig_bps, use_container_width=True)

                    # Repetition burden chart
                    col_left, col_right = st.columns(2)

                    with col_left:
                        fig_burden = px.line(
                            df,
                            x="bucket_start",
                            y="avg_repetition_burden",
                            markers=True,
                            title="Average Repetition Burden",
                            color_discrete_sequence=["#ff7f0e"],
                        )
                        fig_burden.update_yaxes(range=[0.5, 3.5])
                        fig_burden.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Avg Attempts Used"
                        )
                        st.plotly_chart(fig_burden, use_container_width=True)

                    with col_right:
                        fig_sessions = px.bar(
                            df,
                            x="bucket_start",
                            y="n_sessions",
                            title="Sessions per Period",
                            color_discrete_sequence=["#1f77b4"],
                        )
                        fig_sessions.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Number of Sessions"
                        )
                        st.plotly_chart(fig_sessions, use_container_width=True)

                    # Data table
                    st.subheader("Period Details")
                    display_df = df[["bucket_start", "n_sessions", "avg_brain_performance_score", "avg_repetition_burden"]].copy()
                    display_df.columns = ["Period Start", "Sessions", "Brain Performance Score", "Repetition Burden"]
                    display_df["Brain Performance Score"] = display_df["Brain Performance Score"].apply(lambda x: f"{x:.2%}")
                    display_df["Repetition Burden"] = display_df["Repetition Burden"].apply(lambda x: f"{x:.2f}")
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Footer
st.sidebar.divider()
st.sidebar.caption("Earth Task Telemetry - Company Dashboard v0.1.0")
