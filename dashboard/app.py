"""Streamlit dashboard for Earth Task Telemetry."""
import os
import time
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# Configuration: prioritize Streamlit secrets, then env var, then localhost fallback
def get_backend_url():
    """Get backend URL from secrets, env var, or fallback to localhost."""
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        if hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"].rstrip("/")
    except Exception:
        pass
    # Try environment variable (legacy API_BASE or new BACKEND_URL)
    env_url = os.getenv("BACKEND_URL") or os.getenv("API_BASE")
    if env_url:
        return env_url.rstrip("/")
    # Fallback to localhost for local development
    return "http://127.0.0.1:8000"

API_BASE_URL = get_backend_url()

st.set_page_config(
    page_title="Earth Task Telemetry",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Earth Task Telemetry Dashboard")


def check_backend_health():
    """Check if backend is running (longer timeout for cloud cold starts)."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        data = response.json()
        # Support both old {"ok": True} and new {"status": "ok"} formats
        return response.status_code == 200 and (data.get("ok") or data.get("status") == "ok")
    except requests.exceptions.RequestException:
        return False


def get_player_sessions(player_id: str):
    """Fetch session summaries for a player."""
    try:
        response = requests.get(f"{API_BASE_URL}/players/{player_id}/sessions", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException:
        return []


def get_model_state(player_id: str):
    """Fetch model state for a player."""
    try:
        response = requests.get(f"{API_BASE_URL}/model/state/{player_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def create_event(
    player_id: str,
    session_id: str,
    sequence_index: int,
    attempt_index: int,
    is_correct: bool,
    sequence_length: int = 4,
    ts_utc: str = None,
):
    """Create event payload matching backend schema (from backend/tests/test_ingest_and_finalize.py)."""
    if ts_utc is None:
        ts_utc = datetime.now(timezone.utc).isoformat()
    return {
        "player_id": player_id,
        "session_id": session_id,
        "task_version": "earth_v1",
        "sequence_index": sequence_index,
        "sequence_length": sequence_length,
        "attempt_index": attempt_index,
        "presented": [1, 2, 3, 4],
        "input": [1, 2, 3, 4] if is_correct else [4, 3, 2, 1],
        "is_correct": is_correct,
        "duration_ms": 5000,
        "ts_utc": ts_utc,
        "remediation_stage": "none",
        "volts_count": None,
    }


def generate_demo_sessions(player_id: str):
    """
    Generate 5 demo sessions with varied outcomes (bad -> good progression).
    Returns (success: bool, message: str, count: int).
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    # Define session patterns: list of (session_suffix, events_pattern)
    # events_pattern: list of (sequence_index, attempts_before_correct)
    # attempts_before_correct: 0 = correct on 1st, 1 = correct on 2nd, 2 = correct on 3rd, 3 = never solved
    session_patterns = [
        # Session 1: Bad - all sequences need 3 attempts (fta=0.0, burden=3.0)
        ("demo-001", [(1, 2), (2, 2), (3, 2)]),
        # Session 2: Poor - mixed, mostly struggling (fta=0.33, burden=2.0)
        ("demo-002", [(1, 0), (2, 1), (3, 2)]),
        # Session 3: Medium - some struggle (fta=0.33, burden=2.0)
        ("demo-003", [(1, 1), (2, 1), (3, 1)]),
        # Session 4: Good - mostly first attempt (fta=0.66, burden=1.33)
        ("demo-004", [(1, 0), (2, 0), (3, 1)]),
        # Session 5: Excellent - all first attempt (fta=1.0, burden=1.0)
        ("demo-005", [(1, 0), (2, 0), (3, 0)]),
    ]

    created_count = 0

    for session_suffix, events_pattern in session_patterns:
        session_id = f"{session_suffix}-{timestamp}"
        base_time = datetime.now(timezone.utc)

        # Generate events for each sequence
        for seq_idx, attempts_before_correct in events_pattern:
            # Post wrong attempts first
            for attempt in range(1, attempts_before_correct + 1):
                event_time = base_time.isoformat()
                event = create_event(
                    player_id=player_id,
                    session_id=session_id,
                    sequence_index=seq_idx,
                    attempt_index=attempt,
                    is_correct=False,
                    ts_utc=event_time,
                )
                response = requests.post(f"{API_BASE_URL}/events", json=event, timeout=5)
                if response.status_code != 200:
                    return False, f"Failed to post event: {response.text}", created_count
                time.sleep(0.05)

            # Post correct attempt (attempt_index = attempts_before_correct + 1)
            correct_attempt = attempts_before_correct + 1
            if correct_attempt <= 3:  # Max 3 attempts allowed
                event_time = base_time.isoformat()
                event = create_event(
                    player_id=player_id,
                    session_id=session_id,
                    sequence_index=seq_idx,
                    attempt_index=correct_attempt,
                    is_correct=True,
                    ts_utc=event_time,
                )
                response = requests.post(f"{API_BASE_URL}/events", json=event, timeout=5)
                if response.status_code != 200:
                    return False, f"Failed to post event: {response.text}", created_count
                time.sleep(0.05)

        # Finalize session
        response = requests.post(f"{API_BASE_URL}/sessions/{session_id}/finalize", timeout=5)
        if response.status_code != 200:
            return False, f"Failed to finalize session {session_id}: {response.text}", created_count

        created_count += 1
        time.sleep(0.1)

    return True, f"Generated {created_count} demo sessions and retrained model.", created_count


# Sidebar
st.sidebar.header("Controls")

# Backend status indicator
backend_ok = check_backend_health()
if backend_ok:
    st.sidebar.markdown(f"**Backend:** :green_circle: Connected")
    st.sidebar.caption(f"`{API_BASE_URL}`")
else:
    st.sidebar.markdown(f"**Backend:** :red_circle: Unreachable")
    st.sidebar.caption(f"`{API_BASE_URL}`")
    st.warning("Backend unreachable (may be waking up or misconfigured). Refresh the page to retry.")

# Get player_id from URL query param if present
query_params = st.query_params
url_player_id = query_params.get("player_id", None)

# Player lookup
st.sidebar.subheader("Player Lookup")
player_id = st.sidebar.text_input(
    "Player ID",
    value=url_player_id if url_player_id else "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    help="Enter a player UUID to view their sessions",
)

# Auto-load sessions if player_id came from URL and not yet loaded
if url_player_id and "sessions" not in st.session_state:
    sessions = get_player_sessions(url_player_id)
    if sessions:
        st.session_state["sessions"] = sessions
        st.session_state["player_id"] = url_player_id

if st.sidebar.button("Load Sessions"):
    if player_id:
        sessions = get_player_sessions(player_id)
        if sessions:
            st.session_state["sessions"] = sessions
            st.session_state["player_id"] = player_id
        else:
            st.sidebar.warning("No sessions found for this player")
    else:
        st.sidebar.warning("Enter a player ID")

if st.sidebar.button("Generate Demo Sessions"):
    if not player_id:
        st.sidebar.warning("Enter a player ID first")
    elif not backend_ok:
        st.sidebar.error("Backend is offline - cannot generate sessions")
    else:
        with st.spinner("Generating demo sessions..."):
            success, message, count = generate_demo_sessions(player_id)
        if success:
            # Refresh sessions and player_id in session state
            sessions = get_player_sessions(player_id)
            if sessions:
                st.session_state["sessions"] = sessions
                st.session_state["player_id"] = player_id
            st.sidebar.success(message)
            st.rerun()
        else:
            st.error(message)

# Main content
if "sessions" in st.session_state and st.session_state["sessions"]:
    sessions = st.session_state["sessions"]
    player_id = st.session_state["player_id"]

    st.subheader(f"Sessions for Player: `{player_id}`")

    # Convert to DataFrame
    df = pd.DataFrame(sessions)
    df["created_ts_utc"] = pd.to_datetime(df["created_ts_utc"])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", len(df))
    with col2:
        avg_fta = df["fta_level"].mean()
        st.metric("Avg FTA Level", f"{avg_fta:.2%}")
    with col3:
        avg_burden = df["repetition_burden"].mean()
        st.metric("Avg Repetition Burden", f"{avg_burden:.2f}")
    with col4:
        avg_score = df["earth_score_bucket"].mean()
        st.metric("Avg Earth Score", f"{avg_score:.1f}")

    st.divider()

    # Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("FTA Level Over Sessions")
        fig_fta = px.line(
            df,
            x="created_ts_utc",
            y="fta_level",
            markers=True,
            title="First-Time Accuracy Trend",
        )
        fig_fta.update_yaxes(range=[0, 1.1], tickformat=".0%")
        fig_fta.update_layout(xaxis_title="Session Time", yaxis_title="FTA Level")
        st.plotly_chart(fig_fta, use_container_width=True)

    with col_right:
        st.subheader("Repetition Burden Over Sessions")
        fig_burden = px.line(
            df,
            x="created_ts_utc",
            y="repetition_burden",
            markers=True,
            title="Repetition Burden Trend",
            color_discrete_sequence=["#ff7f0e"],
        )
        fig_burden.update_yaxes(range=[0.5, 3.5])
        fig_burden.update_layout(xaxis_title="Session Time", yaxis_title="Avg Attempts Used")
        st.plotly_chart(fig_burden, use_container_width=True)

    # Earth Score Distribution
    st.subheader("Earth Score Distribution")
    score_counts = df["earth_score_bucket"].value_counts().sort_index()
    fig_score = px.bar(
        x=score_counts.index,
        y=score_counts.values,
        labels={"x": "Earth Score Bucket", "y": "Count"},
        title="Session Outcomes",
        color=score_counts.index,
        color_continuous_scale=["#d62728", "#ffbb78", "#2ca02c"],
    )
    fig_score.update_layout(showlegend=False)
    st.plotly_chart(fig_score, use_container_width=True)

    # Raw data table
    st.subheader("Session Details")
    st.dataframe(
        df[["session_id", "task_version", "fta_level", "fta_strict", "repetition_burden", "earth_score_bucket", "created_ts_utc"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Model State panel
    st.subheader("Model State (Learning Brain)")
    model_state = get_model_state(player_id)

    if model_state is None:
        st.info("No model state available for this player yet.")
    elif "error" in model_state:
        st.error(f"Failed to fetch model state: {model_state['error']}")
    else:
        status = model_state.get("status", "unknown")
        n_samples = model_state.get("n_samples", 0)
        is_trained = status == "trained"

        # Format trained_ts_utc safely
        trained_ts_raw = model_state.get("trained_ts_utc")
        if trained_ts_raw:
            trained_ts_display = trained_ts_raw
        else:
            trained_ts_display = "-"

        # Always show status and samples
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Status", status)
        with col_m2:
            st.metric("Samples", n_samples)

        if not is_trained:
            st.info("Model not trained yet - finalize at least 3 sessions to enable learning.")
        else:
            # Show trained timestamp, MAE, and coefficients only when trained
            st.markdown(f"**Last Trained:** `{trained_ts_display}`")

            mae = model_state.get("mae")
            mae_display = f"{mae:.4f}" if mae is not None else "-"
            st.markdown(f"**MAE:** `{mae_display}`")

            # Extract coefficients
            coefficients = model_state.get("coefficients")
            intercept = model_state.get("intercept")

            weight_prev_fta = "-"
            weight_prev_repetition_burden = "-"
            if coefficients and isinstance(coefficients, list):
                if len(coefficients) > 0:
                    weight_prev_fta = f"{coefficients[0]:.6f}"
                if len(coefficients) > 1:
                    weight_prev_repetition_burden = f"{coefficients[1]:.6f}"

            intercept_display = f"{intercept:.6f}" if intercept is not None else "-"

            st.markdown(f"**weight_prev_fta:** `{weight_prev_fta}`")
            st.markdown(f"**weight_prev_repetition_burden:** `{weight_prev_repetition_burden}`")
            st.markdown(f"**intercept:** `{intercept_display}`")

    # Next-Session Prediction panel
    st.divider()
    st.subheader("Next-Session Prediction")

    # Determine if we can compute a prediction
    can_predict = False
    prediction_reason = None

    if model_state is None:
        prediction_reason = "No model state available for this player yet."
    elif "error" in model_state:
        prediction_reason = f"Failed to fetch model state: {model_state['error']}"
    elif model_state.get("status") != "trained":
        prediction_reason = "Model not trained yet - finalize at least 3 sessions to enable predictions."
    elif len(sessions) == 0:
        prediction_reason = "No sessions available to base prediction on."
    else:
        coefficients = model_state.get("coefficients")
        intercept = model_state.get("intercept")

        if not coefficients or not isinstance(coefficients, list) or len(coefficients) < 2:
            prediction_reason = "Model coefficients not available."
        elif intercept is None:
            prediction_reason = "Model intercept not available."
        else:
            latest_session = sessions[-1]
            last_fta = latest_session.get("fta_level")
            last_burden = latest_session.get("repetition_burden")

            if last_fta is None or last_burden is None:
                prediction_reason = "Latest session missing required fields (fta_level or repetition_burden)."
            else:
                can_predict = True

    if can_predict:
        # Compute prediction: y_hat = intercept + coef[0] * last_fta + coef[1] * last_burden
        y_hat = intercept + coefficients[0] * last_fta + coefficients[1] * last_burden
        # Clamp to valid FTA range [0.0, 1.0]
        y_hat = max(0.0, min(1.0, y_hat))

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Predicted Next-Session FTA", f"{y_hat:.2f}")
        with col_p2:
            st.metric("Last Actual FTA", f"{last_fta:.2f}")
        with col_p3:
            st.metric("Last Repetition Burden", f"{last_burden:.2f}")
    else:
        st.info(prediction_reason)

else:
    st.info("Enter a player ID in the sidebar and click 'Load Sessions' to view telemetry data.")

    # Show instructions when no data
    st.subheader("Getting Started")
    if backend_ok:
        st.markdown("""
        **Quick Start:**
        1. Click **"Generate Demo Sessions"** in the sidebar to create sample data
        2. Or enter a known **Player ID** and click **"Load Sessions"**

        The demo generator creates 5 sessions showing a learning progression from poor to excellent performance.
        """)
    else:
        st.markdown("""
        **Backend Unavailable**

        The backend API is currently unreachable. This could mean:
        - The backend is starting up (cloud services may take 30-60 seconds on first request)
        - There's a configuration issue with the backend URL

        **Try:** Refresh this page in a few seconds.

        *For developers:* See the project README for local setup instructions.
        """)

# Footer
st.sidebar.divider()
st.sidebar.caption("Earth Task Telemetry v0.1.0")
