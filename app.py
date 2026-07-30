"""
app.py
======
Silent Voice — Streamlit UI

Wraps predict_live.py's inference engine in a Streamlit web interface that
provides:
  - A live webcam video feed (updated at ~15 fps via st.rerun())
  - The currently predicted ASL sign with confidence
  - A scrolling sentence box built from stable predictions
  - A "Speak" button for offline TTS playback
  - A "Clear" button to reset the sentence
  - A sidebar with configuration controls

Architecture
------------
A daemon background thread (camera_worker) runs the webcam capture and
inference loop continuously, writing the latest annotated frame and
prediction into a thread-safe shared-state dict (_shared).  The main
Streamlit script reads from _shared on every re-run, displays the frame
via st.image(), then sleeps briefly and calls st.rerun() to trigger the
next display update.  This pattern requires no extra dependencies beyond
the fixed tech stack.

Usage
-----
    streamlit run app.py

Reference
---------
UI layout inspired by references/StreamSign/app.py (hamdi3/StreamSign):
custom CSS header, centred title, clean column layout.

Author : Silent Voice Team
Date   : 2026
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Internal modules
from predict_live import (
    SentenceBuffer,
    build_hands,
    draw_overlay,
    get_prediction,
    load_model,
    speak_text_async,
)

# ── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="Silent Voice",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (StreamSign-inspired clean styling) ─────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ───────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide default Streamlit chrome ────────────────────────────── */
    #MainMenu  { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* ── App background ────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }

    /* ── Hero title ───────────────────────────────────────────────── */
    .sv-hero {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .sv-hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #64d2ff, #bf5af2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sv-hero p {
        color: #9090b0;
        font-size: 0.95rem;
        margin-top: 0;
    }

    /* ── Cards ────────────────────────────────────────────────────── */
    .sv-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .sv-card-label {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7070a0;
        margin-bottom: 0.3rem;
    }

    /* ── Sign display ─────────────────────────────────────────────── */
    .sv-sign {
        font-size: 4rem;
        font-weight: 700;
        color: #64d2ff;
        line-height: 1;
    }
    .sv-conf {
        font-size: 0.85rem;
        color: #50d0a0;
    }

    /* ── Sentence box ─────────────────────────────────────────────── */
    .sv-sentence {
        font-size: 1.1rem;
        color: #e8e8ff;
        min-height: 4em;
        word-break: break-word;
        line-height: 1.6;
    }

    /* ── Status badge ─────────────────────────────────────────────── */
    .sv-badge-on  { color: #30d0a0; font-weight: 600; }
    .sv-badge-off { color: #c04040; font-weight: 600; }

    /* ── Buttons ──────────────────────────────────────────────────── */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.08);
        color: #f0f0f0;
        font-size: 0.95rem;
        padding: 0.55rem 0;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.15);
        border-color: #64d2ff;
    }

    /* ── Sidebar ──────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(15,12,41,0.8);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Shared state between the background thread and Streamlit re-runs
# ═══════════════════════════════════════════════════════════════════════════════

# This dict is module-level so it persists across st.rerun() calls in the
# same process.  Access is protected by a threading.Lock.
_shared: dict = {
    "frame_rgb": None,      # latest annotated frame in RGB (for st.image)
    "label":     None,      # latest predicted class label
    "confidence": 0.0,      # latest classifier confidence
    "lock":      threading.Lock(),
    "running":   False,     # thread lifecycle flag
    "thread":    None,      # reference to the camera_worker Thread
}


def camera_worker(
    camera_index: int,
    model,
    buffer: SentenceBuffer,
    min_det: float,
    min_track: float,
) -> None:
    """
    Background daemon thread: captures webcam frames, runs inference, and
    writes results into _shared for the Streamlit main thread to display.

    Parameters
    ----------
    camera_index : OpenCV device index.
    model        : Loaded Random Forest classifier.
    buffer       : SentenceBuffer shared with the Streamlit session.
    min_det      : MediaPipe minimum detection confidence.
    min_track    : MediaPipe minimum tracking confidence.
    """
    hands = build_hands(
        static_mode=False,
        min_detection_conf=min_det,
        min_tracking_conf=min_track,
    )
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        _shared["running"] = False
        return

    while _shared["running"]:
        ret, frame_bgr = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_bgr = cv2.flip(frame_bgr, 1)     # mirror for natural feel
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run inference (on RGB)
        label, confidence, mp_results = get_prediction(frame_rgb, hands, model)

        # Update sentence buffer (thread-safe: buffer has its own simple state)
        buffer.push(label, confidence)

        # Draw overlays onto the BGR frame, then convert to RGB for st.image
        annotated_bgr = draw_overlay(
            frame_bgr.copy(), mp_results, label, confidence, buffer.sentence
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with _shared["lock"]:
            _shared["frame_rgb"]  = annotated_rgb
            _shared["label"]      = label
            _shared["confidence"] = confidence

        time.sleep(0.033)   # ~30 fps capture rate

    cap.release()
    if hasattr(hands, "close"):
        hands.close()


def start_camera(
    camera_index: int,
    model,
    buffer: SentenceBuffer,
    min_det: float,
    min_track: float,
) -> None:
    """Start the camera_worker thread if it is not already running."""
    if _shared["running"] and _shared["thread"] and _shared["thread"].is_alive():
        return   # already running

    _shared["running"] = True
    t = threading.Thread(
        target=camera_worker,
        args=(camera_index, model, buffer, min_det, min_track),
        daemon=True,     # exits when the main Streamlit process exits
    )
    t.start()
    _shared["thread"] = t


def stop_camera() -> None:
    """Signal the background thread to stop."""
    _shared["running"] = False


# ═══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation  (runs once per browser session)
# ═══════════════════════════════════════════════════════════════════════════════

def init_session() -> None:
    """Initialise Streamlit session state keys on first run."""
    if "model" not in st.session_state:
        with st.spinner("Loading classifier …"):
            try:
                st.session_state["model"] = load_model()
                st.session_state["model_error"] = None
            except FileNotFoundError as exc:
                st.session_state["model"] = None
                st.session_state["model_error"] = str(exc)

    if "buffer" not in st.session_state:
        st.session_state["buffer"] = SentenceBuffer()

    if "camera_active" not in st.session_state:
        st.session_state["camera_active"] = False


# ═══════════════════════════════════════════════════════════════════════════════
# Main UI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    init_session()

    # ── Hero header ────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="sv-hero">
            <h1>🤟 Silent Voice</h1>
            <p>Real-time ASL sign language → text → speech &nbsp;|&nbsp; Fully offline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")

        camera_index = st.number_input(
            "Camera Index", min_value=0, max_value=9, value=0, step=1,
            help="OpenCV device index (0 = built-in webcam).",
        )
        min_det = st.slider(
            "Detection Confidence", 0.40, 1.00, 0.70, 0.05,
            help="MediaPipe minimum detection confidence threshold.",
        )
        min_track = st.slider(
            "Tracking Confidence", 0.30, 1.00, 0.50, 0.05,
            help="MediaPipe minimum tracking confidence threshold.",
        )
        buffer_size = st.slider(
            "Buffer Window (frames)", 5, 30, 15, 1,
            help="Rolling window size for prediction stability check.",
        )
        stability = st.slider(
            "Stability Threshold", 0.50, 1.00, 0.80, 0.05,
            help="Fraction of window that must agree before committing a sign.",
        )
        cooldown = st.slider(
            "Cooldown (seconds)", 0.5, 5.0, 2.0, 0.5,
            help="Minimum gap before the same sign is committed again.",
        )

        # Apply buffer parameters (re-create buffer on change)
        buf: SentenceBuffer = st.session_state["buffer"]
        buf.buffer_size = buffer_size
        buf.threshold   = stability
        buf.cooldown    = cooldown

        st.markdown("---")
        st.markdown("### 📋 Vocabulary")
        st.markdown(
            "**Alphabet:** A – Z (26 signs)\n\n"
            "**Words:** hello, thanks, yes, no, help, please, sorry, "
            "water, food, stop, more, eat, drink, love, bathroom"
        )
        st.markdown("---")
        st.caption("Silent Voice v1.0 — Academic Project")

    # ── Model error guard ──────────────────────────────────────────────────────
    if st.session_state.get("model_error"):
        st.error(
            f"⚠️ Model not loaded: {st.session_state['model_error']}\n\n"
            "Run `python train_model.py` to train and save the classifier, "
            "then refresh this page."
        )
        return

    # ── Camera start / stop controls ───────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

    with ctrl_col1:
        if not st.session_state["camera_active"]:
            if st.button("▶  Start Camera", key="btn_start"):
                st.session_state["camera_active"] = True
                start_camera(
                    int(camera_index),
                    st.session_state["model"],
                    st.session_state["buffer"],
                    float(min_det),
                    float(min_track),
                )
        else:
            if st.button("⏹  Stop Camera", key="btn_stop"):
                stop_camera()
                st.session_state["camera_active"] = False

    with ctrl_col2:
        camera_status = (
            '<span class="sv-badge-on">● LIVE</span>'
            if st.session_state["camera_active"]
            else '<span class="sv-badge-off">● OFF</span>'
        )
        st.markdown(f"**Status:** {camera_status}", unsafe_allow_html=True)

    # ── Main layout: video | prediction + sentence ─────────────────────────────
    video_col, info_col = st.columns([3, 2])

    with video_col:
        st.markdown('<div class="sv-card-label">Live Feed</div>',
                    unsafe_allow_html=True)
        frame_slot = st.empty()   # placeholder updated each re-run

    with info_col:
        # ── Current sign ───────────────────────────────────────────────────────
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        st.markdown('<div class="sv-card-label">Current Sign</div>',
                    unsafe_allow_html=True)
        sign_slot = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Sentence ───────────────────────────────────────────────────────────
        st.markdown('<div class="sv-card">', unsafe_allow_html=True)
        st.markdown('<div class="sv-card-label">Sentence</div>',
                    unsafe_allow_html=True)
        sentence_slot = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Action buttons ─────────────────────────────────────────────────────
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            speak_clicked = st.button("🔊  Speak", key="btn_speak",
                                      help="Read the sentence aloud via TTS.")
        with btn_col2:
            clear_clicked = st.button("🗑️  Clear", key="btn_clear",
                                      help="Clear the sentence buffer.")

        # ── Delete last char ───────────────────────────────────────────────────
        delete_clicked = st.button("⌫  Delete Last", key="btn_del",
                                   help="Remove the last character.")

    # ── Handle button actions ──────────────────────────────────────────────────
    buf = st.session_state["buffer"]

    if speak_clicked:
        speak_text_async(buf.sentence)

    if clear_clicked:
        buf.clear()

    if delete_clicked:
        buf.delete_last()

    # ── Read latest frame and prediction from shared state ─────────────────────
    with _shared["lock"]:
        frame_rgb  = _shared["frame_rgb"]
        label      = _shared["label"]
        confidence = _shared["confidence"]

    # ── Update placeholders ────────────────────────────────────────────────────
    if frame_rgb is not None:
        frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)
    elif st.session_state["camera_active"]:
        frame_slot.info("⏳ Waiting for first frame …")
    else:
        frame_slot.markdown(
            '<div style="height:300px; display:flex; align-items:center; '
            'justify-content:center; background:rgba(0,0,0,0.3); '
            'border-radius:8px; color:#555; font-size:1.2rem;">'
            '📷 Camera not started</div>',
            unsafe_allow_html=True,
        )

    if label:
        sign_slot.markdown(
            f'<div class="sv-sign">{label}</div>'
            f'<div class="sv-conf">Confidence: {confidence:.0%}</div>',
            unsafe_allow_html=True,
        )
    else:
        sign_slot.markdown(
            '<div style="color:#505060; font-size:1.1rem; padding:0.5rem 0;">'
            "Show your hand to the camera …</div>",
            unsafe_allow_html=True,
        )

    sentence_display = buf.sentence if buf.sentence else "—"
    sentence_slot.markdown(
        f'<div class="sv-sentence">{sentence_display}</div>',
        unsafe_allow_html=True,
    )

    # ── Live refresh: sleep then force a re-run ────────────────────────────────
    # Only keep the loop running while the camera is active.
    if st.session_state["camera_active"]:
        time.sleep(0.07)     # ~14 fps UI refresh rate
        st.rerun()


if __name__ == "__main__":
    main()
