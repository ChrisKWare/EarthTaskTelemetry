"""Streamlit dashboard for Earth Task Telemetry."""
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Earth Task Telemetry",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Earth Task Telemetry Dashboard")


def check_backend_health():
    """Check if backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200 and response.json().get("ok")
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


# Sidebar
st.sidebar.header("Controls")

# Backend status
backend_ok = check_backend_health()
if backend_ok:
    st.sidebar.success("✅ Backend connected")
else:
    st.sidebar.error("❌ Backend offline")
    st.warning("Backend is not running. Start it with: `uvicorn backend.app.main:app --reload --port 8000`")

# Player lookup
st.sidebar.subheader("Player Lookup")
player_id = st.sidebar.text_input(
    "Player ID",
    value="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    help="Enter a player UUID to view their sessions",
)

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

else:
    st.info("Enter a player ID in the sidebar and click 'Load Sessions' to view telemetry data.")

    # Show instructions when no data
    st.subheader("Getting Started")
    st.markdown("""
    1. **Start the backend** (in a terminal):
       ```
       uvicorn backend.app.main:app --reload --port 8000
       ```

    2. **Load sample data** (in another terminal):
       ```python
       import requests
       import json

       with open("shared/sample_events.jsonl") as f:
           for line in f:
               event = json.loads(line)
               requests.post("http://localhost:8000/events", json=event)

       # Finalize the session
       requests.post("http://localhost:8000/sessions/sess-001-2024-01-15/finalize")
       ```

    3. **Enter the player ID** from sample data:
       ```
       a1b2c3d4-e5f6-7890-abcd-ef1234567890
       ```
    """)

# Footer
st.sidebar.divider()
st.sidebar.caption("Earth Task Telemetry v0.1.0")
