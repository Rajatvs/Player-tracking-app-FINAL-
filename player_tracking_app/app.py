"""
AI Player Tracking & Analysis
==============================
Real-time player detection & tracking (YOLOv8 + DeepSORT), movement
heatmap generation, and interactive analysis via Streamlit.

Run with:
    streamlit run app.py
"""
import os
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.video_processor import VideoProcessor

st.set_page_config(page_title="AI Player Tracking & Analysis", page_icon="🏟️", layout="wide")

MODEL_OPTIONS = {
    "YOLOv8n (fastest)": "yolov8n.pt",
    "YOLOv8s (balanced)": "yolov8s.pt",
    "YOLOv8m (most accurate)": "yolov8m.pt",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_detector(model_path: str, conf: float):
    return PlayerDetector(model_path=model_path, conf_threshold=conf)


def new_tracker():
    return PlayerTracker(max_age=30, n_init=3)


def init_session_state():
    defaults = {
        "running": False,
        "processor": None,
        "cap": None,
        "history": [],  # list of (frame_idx, active_players, total_unique)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sidebar_controls():
    st.sidebar.title("⚙️ Settings")

    source_type = st.sidebar.radio("Video source", ["Upload a video file", "Webcam (local only)"])

    uploaded_file = None
    if source_type == "Upload a video file":
        uploaded_file = st.sidebar.file_uploader("Upload match footage", type=["mp4", "avi", "mov", "mkv"])

    model_label = st.sidebar.selectbox("YOLOv8 model", list(MODEL_OPTIONS.keys()), index=0)
    conf_threshold = st.sidebar.slider("Detection confidence threshold", 0.10, 0.90, 0.35, 0.05)
    heatmap_alpha = st.sidebar.slider("Heatmap overlay opacity", 0.0, 1.0, 0.55, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "First run will download the selected YOLOv8 weights automatically "
        "(requires an internet connection on this machine)."
    )

    return source_type, uploaded_file, MODEL_OPTIONS[model_label], conf_threshold, heatmap_alpha


def main():
    init_session_state()

    st.title("🏟️ AI Player Tracking & Analysis")
    st.markdown(
        "Real-time player **detection** (YOLOv8) and **tracking** (DeepSORT) "
        "with live **movement heatmaps**, built with OpenCV and Streamlit."
    )

    source_type, uploaded_file, model_path, conf_threshold, heatmap_alpha = sidebar_controls()

    col_start, col_stop, _ = st.columns([1, 1, 4])
    start_clicked = col_start.button("▶️ Start / Process", type="primary")
    stop_clicked = col_stop.button("⏹️ Stop")

    if stop_clicked:
        st.session_state.running = False

    video_col, heatmap_col = st.columns(2)
    video_placeholder = video_col.empty()
    heatmap_placeholder = heatmap_col.empty()

    metric_cols = st.columns(4)
    active_metric = metric_cols[0].empty()
    total_metric = metric_cols[1].empty()
    fps_metric = metric_cols[2].empty()
    frame_metric = metric_cols[3].empty()

    chart_placeholder = st.empty()
    download_placeholder = st.container()

    if start_clicked:
        if source_type == "Upload a video file" and uploaded_file is None:
            st.warning("Please upload a video file first.")
            return
        if source_type == "Webcam (local only)":
            st.info(
                "Webcam mode only works when this app runs on your own machine "
                "(not in a hosted/cloud environment)."
            )

        # Persist uploaded file to disk so cv2.VideoCapture can read it.
        video_path = 0
        if source_type == "Upload a video file":
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded_file.read())
            tmp.close()
            video_path = tmp.name

        detector = load_detector(model_path, conf_threshold)
        tracker = new_tracker()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open the video source.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        processor = VideoProcessor(detector, tracker, width, height)

        # Prepare an output writer so the processed video can be downloaded.
        out_path = os.path.join(OUTPUT_DIR, "processed_output.mp4")
        heatmap_path = os.path.join(OUTPUT_DIR, "final_heatmap.png")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (width, height))

        st.session_state.running = True
        history = []
        progress_bar = st.progress(0.0) if total_frames > 0 else None

        frame_idx = 0
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            annotated, heatmap_overlay, stats = processor.process_frame(frame, heatmap_alpha=heatmap_alpha)
            writer.write(annotated)

            video_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Live Detection & Tracking", use_container_width=True
            )
            heatmap_placeholder.image(
                cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB), caption="Live Movement Heatmap", use_container_width=True
            )

            active_metric.metric("Active Players", stats["active_players"])
            total_metric.metric("Unique Players Tracked", stats["total_unique_players"])
            fps_metric.metric("Processing FPS", stats["fps"])
            frame_metric.metric("Frame", f"{frame_idx}/{total_frames or '?'}")

            history.append(stats)
            if len(history) % 5 == 0:
                df = pd.DataFrame(history)
                chart_placeholder.line_chart(df.set_index("frame")[["active_players", "total_unique_players"]])

            if progress_bar is not None and total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

        cap.release()
        writer.release()
        cv2.imwrite(heatmap_path, processor.heatmap.as_standalone_image())
        st.session_state.running = False

        if video_path != 0 and os.path.exists(video_path):
            os.remove(video_path)

        st.success("Processing complete.")

        with download_placeholder:
            dl_col1, dl_col2 = st.columns(2)
            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    dl_col1.download_button("⬇️ Download annotated video", f, file_name="player_tracking_output.mp4")
            if os.path.exists(heatmap_path):
                with open(heatmap_path, "rb") as f:
                    dl_col2.download_button("⬇️ Download heatmap image", f, file_name="movement_heatmap.png")


if __name__ == "__main__":
    main()
