import streamlit as st
import cv2
import time
from focus_detector import detect_focus, manage_timer, check_mobile_usage

st.set_page_config(page_title="Focus Tracker", layout="wide")
st.title("🧠 AI-Based Focus Tracker")

# Sidebar for input settings
st.sidebar.header("Session Settings")
focus_duration = st.sidebar.number_input("Focus Duration (min)", 1, 60, 25)
break_duration = st.sidebar.number_input("Break Duration (min)", 1, 60, 5)

# Initialize session state variables
if "run" not in st.session_state:
    st.session_state.run = False
if "focus_score" not in st.session_state:
    st.session_state.focus_score = 100
if "status" not in st.session_state:
    st.session_state.status = "Focus"
if "focus_start" not in st.session_state:  # Make sure it's initialized
    st.session_state.focus_start = None
if "break_start" not in st.session_state:  # Make sure it's initialized
    st.session_state.break_start = None
if "focus_timer_display" not in st.session_state:
    st.session_state.focus_timer_display = "00:00"
if "break_timer_display" not in st.session_state:
    st.session_state.break_timer_display = "00:00"
if "mobile_warning" not in st.session_state:
    st.session_state.mobile_warning = False

# Start/Stop toggle
start_stop = st.sidebar.button("Start" if not st.session_state.run else "Stop")
if start_stop:
    st.session_state.run = not st.session_state.run

# Layout for live feed and metrics
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Live Webcam Feed")
    FRAME_WINDOW = st.empty()

with col2:
    st.subheader("Focus Metrics")
    focus_score_display = st.empty()
    focus_timer_display = st.empty()
    break_timer_display = st.empty()
    status_display = st.empty()
    warning_display = st.empty()

# Webcam and processing logic
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)

        # Detect focus and update score
        frame, st.session_state.focus_score = detect_focus(frame, st.session_state.focus_score)

        # Timer management
        status, focus_timer_str, break_timer_str = manage_timer(focus_duration, break_duration)
        st.session_state.status = status
        st.session_state.focus_timer_display = focus_timer_str
        st.session_state.break_timer_display = break_timer_str

        # Mobile usage detection
        st.session_state.mobile_warning = check_mobile_usage(frame)

        # Update UI
        FRAME_WINDOW.image(frame, channels="BGR")
        focus_score_display.metric("Focus Score", st.session_state.focus_score)
        focus_timer_display.metric("Focus Timer", st.session_state.focus_timer_display)
        break_timer_display.metric("Break Timer", st.session_state.break_timer_display)
        status_display.markdown(f"**Status:** {st.session_state.status}")
        # if st.session_state.mobile_warning:
        #     warning_display.error("Warning: You have been using your phone too long!")

        # Allow Streamlit to update screen
        time.sleep(1 / 12)

    cap.release()
else:
    st.warning("Click 'Start' in the sidebar to begin tracking.")
