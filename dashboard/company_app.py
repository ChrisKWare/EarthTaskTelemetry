"""Streamlit dashboard for Company-level Earth Task Telemetry."""
import base64
import os
from pathlib import Path

import streamlit as st
import pandas as pd
import requests
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
    page_title="Company Dashboard - Neuraserv",
    page_icon="\U0001f3e2",
    layout="wide",
)

# --- Header: Neuraserv logo + title ---
_LOGO_PATH = Path(__file__).parent / "assets" / "neuraserv_logo.png"


def _render_header():
    """Render the dashboard header with logo and title."""
    logo_b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:0.5rem;">
            <img src="data:image/png;base64,{logo_b64}"
                 alt="Neuraserv"
                 style="height:48px; width:auto; object-fit:contain;">
            <span style="font-size:2rem; font-weight:700; line-height:1.2;">
                Company Dashboard
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


_render_header()


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
        water_pred: float | None  (mean predicted Stress 0-100%)
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
                    stress = latest.get("stress_score")
                    if stress is not None:
                        pred = intercept + coefficients[0] * stress
                        pred = max(0.0, min(100.0, pred))
                        water_preds.append(pred)

    return {
        "earth_pred": sum(earth_preds) / len(earth_preds) if earth_preds else None,
        "water_pred": sum(water_preds) / len(water_preds) if water_preds else None,
        "earth_n": len(earth_preds),
        "water_n": len(water_preds),
    }


def _build_performance_chart(df, has_earth, has_water, predictions, title):
    """Build a combined Brain Performance + Stress chart from a timeseries DataFrame."""
    fig = go.Figure()

    df_earth_ts = None
    df_water_ts = None

    if has_earth:
        df_earth_ts = df.dropna(subset=["avg_brain_performance_score"])
        fig.add_trace(go.Scatter(
            x=df_earth_ts["bucket_start"],
            y=df_earth_ts["avg_brain_performance_score"] * 100,
            mode="lines+markers",
            name="Brain Performance Score",
            line=dict(color="#7ed957"),
            marker=dict(color="#7ed957"),
            hovertemplate="%{x|%Y-%m-%d}<br>Brain Performance: %{y:.1f}%<extra></extra>",
        ))

    if has_water:
        df_water_ts = df.dropna(subset=["avg_stress_score"])
        fig.add_trace(go.Scatter(
            x=df_water_ts["bucket_start"],
            y=df_water_ts["avg_stress_score"],
            mode="lines+markers",
            name="Stress Score",
            line=dict(color="#8c52ff"),
            marker=dict(color="#8c52ff"),
            hovertemplate="%{x|%Y-%m-%d}<br>Stress: %{y:.1f}%<extra></extra>",
        ))

    if predictions:
        if predictions["earth_pred"] is not None and df_earth_ts is not None and len(df_earth_ts):
            last_x = df_earth_ts["bucket_start"].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_x],
                y=[predictions["earth_pred"]],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#ff6347"),
                name=f"Predicted Brain ({predictions['earth_pred']:.1f}%)",
                showlegend=True,
            ))
        if predictions["water_pred"] is not None and df_water_ts is not None and len(df_water_ts):
            last_x = df_water_ts["bucket_start"].iloc[-1]
            fig.add_trace(go.Scatter(
                x=[last_x],
                y=[predictions["water_pred"]],
                mode="markers",
                marker=dict(symbol="diamond", size=12, color="#ff6347"),
                name=f"Predicted Stress ({predictions['water_pred']:.1f}%)",
                showlegend=True,
            ))

    fig.update_yaxes(range=[0, 110], ticksuffix="%")
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Score (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    return fig


# Privacy gate placeholder text
PRIVACY_MSG = "Hidden until 1 player for anonymity."

# Get token from URL query param
query_params = st.query_params
token = query_params.get("t", None)

# Backend status
backend_ok = check_backend_health()

# Sidebar
st.sidebar.header("Dashboard")

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
                f"Insufficient players ({n_players}/1). "
                "Aggregates are hidden to protect individual privacy. "
                "Once 1 or more unique players complete sessions, data will appear."
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
                st.metric("Avg Stress Score", "\u2014")
        else:
            has_earth = summary.get("avg_brain_performance_score") is not None
            has_water = summary.get("avg_stress_score") is not None

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
                    avg_stress = summary["avg_stress_score"]
                    st.metric("Avg Stress Score", f"{avg_stress:.1f}%")
                else:
                    st.metric("Avg Stress Score", "\u2014")

            # Advanced metrics expander (repetition burden)
            if has_earth:
                with st.expander("Advanced metrics"):
                    avg_burden = summary["avg_repetition_burden"]
                    st.metric("Avg Repetition Burden", f"{avg_burden:.2f}")

        st.divider()

        # --- Timeseries ---
        # Fetch both day (weekly view) and week (monthly view) buckets
        ts_day = None
        ts_week = None
        if not privacy_gated:
            ts_day = get_company_timeseries(token, "day")
            if "error" in ts_day:
                if ts_day.get("status_code") == 403:
                    privacy_gated = True
                    ts_day = None
                else:
                    st.error(f"Error loading daily timeseries: {ts_day['error']}")
                    ts_day = None

        if not privacy_gated:
            ts_week = get_company_timeseries(token, "week")
            if "error" in ts_week:
                if ts_week.get("status_code") == 403:
                    privacy_gated = True
                    ts_week = None
                else:
                    st.error(f"Error loading weekly timeseries: {ts_week['error']}")
                    ts_week = None

        # Build DataFrames for each bucket type
        df_day = None
        df_week = None
        has_earth_day = False
        has_water_day = False
        has_earth_week = False
        has_water_week = False

        if ts_day and ts_day.get("buckets"):
            df_day = pd.DataFrame(ts_day["buckets"])
            df_day["bucket_start"] = pd.to_datetime(df_day["bucket_start"])
            has_earth_day = df_day["avg_brain_performance_score"].notna().any()
            has_water_day = df_day["avg_stress_score"].notna().any()

        if ts_week and ts_week.get("buckets"):
            df_week = pd.DataFrame(ts_week["buckets"])
            df_week["bucket_start"] = pd.to_datetime(df_week["bucket_start"])
            has_earth_week = df_week["avg_brain_performance_score"].notna().any()
            has_water_week = df_week["avg_stress_score"].notna().any()

        # Use df_day as the default df for downstream sections (data table, etc.)
        df = df_day
        has_earth_ts = has_earth_day
        has_water_ts = has_water_day

        # --- Predictions ---
        predictions = None
        if not privacy_gated and player_ids:
            predictions = fetch_company_predictions(player_ids, API_BASE_URL)

        # --- Weekly / Monthly Side-by-Side Charts ---
        if privacy_gated:
            st.subheader("Performance Over Time")
            st.info(PRIVACY_MSG)
        elif (df_day is not None and (has_earth_day or has_water_day)) or \
             (df_week is not None and (has_earth_week or has_water_week)):

            # CSS gradient vertical separator
            st.markdown(
                """
                <style>
                .gradient-separator {
                    width: 4px;
                    min-height: 100%;
                    height: 100%;
                    background: linear-gradient(180deg, #0097b2, #7ed957);
                    border-radius: 2px;
                    margin: 0 auto;
                }
                /* Stretch the separator column to full height */
                div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
                    display: flex;
                    align-items: stretch;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            col_weekly, col_sep, col_monthly = st.columns([10, 1, 10])

            with col_weekly:
                st.subheader("Weekly Progress")
                if df_day is not None and (has_earth_day or has_water_day):
                    fig_weekly = _build_performance_chart(
                        df_day, has_earth_day, has_water_day,
                        predictions, "Brain Performance & Stress (Daily)",
                    )
                    st.plotly_chart(fig_weekly, use_container_width=True)
                else:
                    st.info("No daily data yet.")

            with col_sep:
                st.markdown(
                    '<div class="gradient-separator"></div>',
                    unsafe_allow_html=True,
                )

            with col_monthly:
                st.subheader("Monthly Progress")
                if df_week is not None and (has_earth_week or has_water_week):
                    fig_monthly = _build_performance_chart(
                        df_week, has_earth_week, has_water_week,
                        predictions, "Brain Performance & Stress (Weekly)",
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)
                else:
                    st.info("No weekly data yet.")
        else:
            st.subheader("Performance Over Time")
            st.info("No performance data yet.")

        st.divider()

        # --- ML Prediction Tiles ---
        st.subheader("ML Predictions")
        pred_cols = st.columns(2)

        if privacy_gated:
            with pred_cols[0]:
                st.metric("Predicted Next Brain Performance", "\u2014")
                st.caption(PRIVACY_MSG)
            with pred_cols[1]:
                st.metric("Predicted Next Stress Score", "\u2014")
                st.caption(PRIVACY_MSG)
        elif predictions is None:
            with pred_cols[0]:
                st.metric("Predicted Next Brain Performance", "N/A")
                st.caption("No trained models yet")
            with pred_cols[1]:
                st.metric("Predicted Next Stress Score", "N/A")
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
                        "Predicted Next Stress Score",
                        f"{predictions['water_pred']:.1f}%",
                    )
                    st.caption(f"Based on {predictions['water_n']} player model(s)")
                else:
                    st.metric("Predicted Next Stress Score", "N/A")
                    st.caption("No trained water models yet")

        st.divider()

        # --- Period Details (collapsed by default) ---
        if not privacy_gated and df is not None:
            with st.expander("Period details"):
                table_cols = ["bucket_start", "n_sessions"]
                col_names = ["Period Start", "Sessions"]
                if has_earth_ts:
                    table_cols += ["avg_brain_performance_score", "avg_repetition_burden"]
                    col_names += ["Brain Performance Score", "Repetition Burden"]
                if has_water_ts:
                    table_cols += ["avg_stress_score"]
                    col_names += ["Stress Score"]

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
                if "Stress Score" in display_df.columns:
                    display_df["Stress Score"] = display_df["Stress Score"].apply(
                        lambda x: f"{x:.1f}%" if x is not None and pd.notna(x) else "-"
                    )
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        elif not privacy_gated:
            with st.expander("Period details"):
                st.info("No period data available yet.")

# Footer
st.sidebar.divider()
st.sidebar.caption("Powered by Neuraserv")
