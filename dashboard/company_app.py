"""Streamlit dashboard for Company-level Earth Task Telemetry."""
import os

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go


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
    page_icon="\U0001f3e2",
    layout="wide",
)

st.title("\U0001f3e2 Company Dashboard")


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


@st.cache_data(ttl=60)
def _cached_player_sessions(player_id, api_base):
    """Fetch and cache player sessions (TTL 60s)."""
    try:
        r = requests.get(f"{api_base}/players/{player_id}/sessions", timeout=3)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return None


@st.cache_data(ttl=60)
def _cached_model_state(player_id, model_name, api_base):
    """Fetch and cache model state (TTL 60s)."""
    try:
        r = requests.get(
            f"{api_base}/model/state/{player_id}",
            params={"model_name": model_name},
            timeout=3,
        )
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    return None


def fetch_company_predictions(player_ids, api_base):
    """Aggregate per-player ML predictions for the company.

    Returns dict with keys:
        earth_pred: float | None  (mean predicted Brain Performance 0-100%)
        water_pred: float | None  (mean predicted Calmness 0-10)
        earth_n: int  (number of players with valid earth predictions)
        water_n: int  (number of players with valid water predictions)
    """
    earth_preds = []
    water_preds = []

    for pid in player_ids:
        # Fetch sessions once per player
        sessions = _cached_player_sessions(pid, api_base)
        if sessions is None:
            continue

        earth_sessions = [s for s in sessions if s.get("task_version") == "earth_v1"]
        water_sessions = [s for s in sessions if s.get("task_version") == "water_v1"]

        # --- Earth prediction ---
        if earth_sessions:
            ms = _cached_model_state(pid, "earth", api_base)
            if ms and ms.get("status") == "trained":
                coefficients = ms.get("coefficients")
                intercept = ms.get("intercept")
                if coefficients and len(coefficients) >= 2 and intercept is not None:
                    latest = earth_sessions[-1]
                    fta = latest.get("fta_level")
                    burden = latest.get("repetition_burden")
                    if fta is not None and burden is not None:
                        pred = intercept + coefficients[0] * fta + coefficients[1] * burden
                        pred = max(0.0, min(1.0, pred))
                        earth_preds.append(pred * 100.0)

        # --- Water prediction ---
        if water_sessions:
            ms = _cached_model_state(pid, "water", api_base)
            if ms and ms.get("status") == "trained":
                coefficients = ms.get("coefficients")
                intercept = ms.get("intercept")
                if coefficients and len(coefficients) >= 1 and intercept is not None:
                    latest = water_sessions[-1]
                    calmness = latest.get("calmness_score")
                    if calmness is not None:
                        pred = intercept + coefficients[0] * calmness
                        pred = max(0.0, min(10.0, pred))
                        water_preds.append(pred)

    return {
        "earth_pred": sum(earth_preds) / len(earth_preds) if earth_preds else None,
        "water_pred": sum(water_preds) / len(water_preds) if water_preds else None,
        "earth_n": len(earth_preds),
        "water_n": len(water_preds),
    }


# Privacy gate placeholder text
PRIVACY_MSG = "Hidden until 5 players for anonymity."

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
        player_ids = company_info.get("player_ids", [])

        st.sidebar.divider()
        st.sidebar.markdown(f"**Company ID:** `{company_id[:8]}...`")
        st.sidebar.markdown(f"**Players:** {n_players}")

        # Privacy gate: determine if data is visible
        privacy_gated = not has_sufficient

        # Fetch summary + timeseries (gracefully handle 403)
        summary = None
        timeseries = None
        if not privacy_gated:
            summary = get_company_summary(token)
            if "error" in summary:
                # Treat 403 as privacy-gated
                if summary.get("status_code") == 403:
                    privacy_gated = True
                    summary = None
                else:
                    st.error(f"Error loading summary: {summary['error']}")
                    summary = None

        if privacy_gated:
            st.warning(
                f"Insufficient players ({n_players}/5). "
                "Aggregates are hidden to protect individual privacy. "
                "Once 5 or more unique players complete sessions, data will appear."
            )

        # --- KPI Row ---
        st.subheader("Company Overview")
        kpi_cols = st.columns(4)

        if privacy_gated or summary is None:
            with kpi_cols[0]:
                st.metric("Total Players", "\u2014")
            with kpi_cols[1]:
                st.metric("Total Sessions", "\u2014")
            with kpi_cols[2]:
                st.metric("Avg Brain Performance Score", "\u2014")
            with kpi_cols[3]:
                st.metric("Avg Calmness Score", "\u2014")
        else:
            has_earth = summary.get("avg_brain_performance_score") is not None
            has_water = summary.get("avg_calmness_score") is not None

            with kpi_cols[0]:
                st.metric("Total Players", summary["n_players"])
            with kpi_cols[1]:
                st.metric("Total Sessions", summary["n_sessions"])
            with kpi_cols[2]:
                if has_earth:
                    avg_bps = summary["avg_brain_performance_score"]
                    st.metric("Avg Brain Performance Score", f"{avg_bps:.2%}")
                else:
                    st.metric("Avg Brain Performance Score", "\u2014")
            with kpi_cols[3]:
                if has_water:
                    avg_calm = summary["avg_calmness_score"]
                    st.metric("Avg Calmness Score", f"{avg_calm:.2f}")
                else:
                    st.metric("Avg Calmness Score", "\u2014")

            # Advanced metrics expander (repetition burden)
            if has_earth:
                with st.expander("Advanced metrics"):
                    avg_burden = summary["avg_repetition_burden"]
                    st.metric("Avg Repetition Burden", f"{avg_burden:.2f}")

        st.divider()

        # --- Timeseries ---
        # Bucket selector (always visible)
        bucket_type = st.radio(
            "Time Bucket",
            options=["day", "week"],
            horizontal=True,
            index=0,
        )

        if not privacy_gated and timeseries is None:
            timeseries = get_company_timeseries(token, bucket_type)
            if "error" in timeseries:
                if timeseries.get("status_code") == 403:
                    privacy_gated = True
                    timeseries = None
                else:
                    st.error(f"Error loading timeseries: {timeseries['error']}")
                    timeseries = None

        # Prepare timeseries DataFrame
        df = None
        has_earth_ts = False
        has_water_ts = False
        if timeseries and timeseries.get("buckets"):
            df = pd.DataFrame(timeseries["buckets"])
            df["bucket_start"] = pd.to_datetime(df["bucket_start"])
            has_earth_ts = df["avg_brain_performance_score"].notna().any()
            has_water_ts = df["avg_calmness_score"].notna().any()

        # --- Predictions ---
        predictions = None
        if not privacy_gated and player_ids:
            predictions = fetch_company_predictions(player_ids, API_BASE_URL)

        # --- Brain Performance Section ---
        st.subheader("Brain Performance Score Over Time")
        if privacy_gated:
            st.info(PRIVACY_MSG)
        elif df is not None and has_earth_ts:
            df_earth_ts = df.dropna(subset=["avg_brain_performance_score"])
            fig_bps = px.line(
                df_earth_ts,
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
            # Add prediction line if available (earth_pred is 0-100, chart y is 0-1)
            if predictions and predictions["earth_pred"] is not None:
                pred_frac = predictions["earth_pred"] / 100.0
                last_x = df_earth_ts["bucket_start"].iloc[-1]
                fig_bps.add_trace(go.Scatter(
                    x=[last_x],
                    y=[pred_frac],
                    mode="markers",
                    marker=dict(symbol="diamond", size=12, color="#ff6347"),
                    name=f"Predicted ({predictions['earth_pred']:.1f}%)",
                    showlegend=True,
                ))
                fig_bps.add_hline(
                    y=pred_frac,
                    line_dash="dash",
                    line_color="#ff6347",
                    opacity=0.5,
                )
            st.plotly_chart(fig_bps, use_container_width=True)
        else:
            st.info("No Brain Performance data yet.")

        st.divider()

        # --- Calmness Section ---
        st.subheader("Calmness Score Over Time")
        if privacy_gated:
            st.info(PRIVACY_MSG)
        elif df is not None and has_water_ts:
            df_water_ts = df.dropna(subset=["avg_calmness_score"])
            fig_calm = px.line(
                df_water_ts,
                x="bucket_start",
                y="avg_calmness_score",
                markers=True,
                title="Average Calmness Score Over Time",
                color_discrete_sequence=["#1f77b4"],
            )
            fig_calm.update_yaxes(range=[0, 10.5])
            fig_calm.update_layout(
                xaxis_title="Date",
                yaxis_title="Calmness Score"
            )
            # Add prediction line if available
            if predictions and predictions["water_pred"] is not None:
                last_x = df_water_ts["bucket_start"].iloc[-1]
                fig_calm.add_trace(go.Scatter(
                    x=[last_x],
                    y=[predictions["water_pred"]],
                    mode="markers",
                    marker=dict(symbol="diamond", size=12, color="#ff6347"),
                    name=f"Predicted ({predictions['water_pred']:.2f})",
                    showlegend=True,
                ))
                fig_calm.add_hline(
                    y=predictions["water_pred"],
                    line_dash="dash",
                    line_color="#ff6347",
                    opacity=0.5,
                )
            st.plotly_chart(fig_calm, use_container_width=True)
        else:
            st.info("No Calmness data yet.")

        st.divider()

        # --- ML Prediction Tiles ---
        st.subheader("ML Predictions")
        pred_cols = st.columns(2)

        if privacy_gated:
            with pred_cols[0]:
                st.metric("Predicted Next Brain Performance", "\u2014")
                st.caption(PRIVACY_MSG)
            with pred_cols[1]:
                st.metric("Predicted Next Calmness (0\u201310)", "\u2014")
                st.caption(PRIVACY_MSG)
        elif predictions is None:
            with pred_cols[0]:
                st.metric("Predicted Next Brain Performance", "N/A")
                st.caption("No trained models yet")
            with pred_cols[1]:
                st.metric("Predicted Next Calmness (0\u201310)", "N/A")
                st.caption("No trained models yet")
        else:
            with pred_cols[0]:
                if predictions["earth_pred"] is not None:
                    st.metric(
                        "Predicted Next Brain Performance",
                        f"{predictions['earth_pred']:.1f}%",
                    )
                    st.caption(f"Based on {predictions['earth_n']} player model(s)")
                else:
                    st.metric("Predicted Next Brain Performance", "N/A")
                    st.caption("No trained earth models yet")
            with pred_cols[1]:
                if predictions["water_pred"] is not None:
                    st.metric(
                        "Predicted Next Calmness (0\u201310)",
                        f"{predictions['water_pred']:.2f}",
                    )
                    st.caption(f"Based on {predictions['water_n']} player model(s)")
                else:
                    st.metric("Predicted Next Calmness (0\u201310)", "N/A")
                    st.caption("No trained water models yet")

        st.divider()

        # --- Additional charts (only when data available and not gated) ---
        if not privacy_gated and df is not None:
            col_left, col_right = st.columns(2)

            with col_left:
                if has_earth_ts:
                    df_burden = df.dropna(subset=["avg_repetition_burden"])
                    fig_burden = px.line(
                        df_burden,
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
            table_cols = ["bucket_start", "n_sessions"]
            col_names = ["Period Start", "Sessions"]
            if has_earth_ts:
                table_cols += ["avg_brain_performance_score", "avg_repetition_burden"]
                col_names += ["Brain Performance Score", "Repetition Burden"]
            if has_water_ts:
                table_cols += ["avg_calmness_score"]
                col_names += ["Calmness Score"]

            display_df = df[table_cols].copy()
            display_df.columns = col_names
            if "Brain Performance Score" in display_df.columns:
                display_df["Brain Performance Score"] = display_df["Brain Performance Score"].apply(
                    lambda x: f"{x:.2%}" if x is not None and pd.notna(x) else "-"
                )
            if "Repetition Burden" in display_df.columns:
                display_df["Repetition Burden"] = display_df["Repetition Burden"].apply(
                    lambda x: f"{x:.2f}" if x is not None and pd.notna(x) else "-"
                )
            if "Calmness Score" in display_df.columns:
                display_df["Calmness Score"] = display_df["Calmness Score"].apply(
                    lambda x: f"{x:.2f}" if x is not None and pd.notna(x) else "-"
                )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# Footer
st.sidebar.divider()
st.sidebar.caption("Earth Task Telemetry - Company Dashboard v0.2.0")
