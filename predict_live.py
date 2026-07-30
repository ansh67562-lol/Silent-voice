"""
predict_live.py
===============
Silent Voice — Live Inference Engine

Provides real-time ASL recognition from a webcam using MediaPipe hand
landmark extraction and a trained scikit-learn Random Forest classifier.

Public API (used by app.py)
---------------------------
    load_model()               → RandomForestClassifier
    build_hands()              → MediaPipe Hands object
    get_prediction(frame_rgb, hands, model) → (label | None, confidence)
    draw_overlay(frame, hand_results, label, confidence, sentence)
    speak_text(text)           → None  (offline TTS via pyttsx3)
    SentenceBuffer             → stateful word-buffer class
    run_inference_loop(camera_index)  → standalone blocking loop

Word-Buffering Logic
--------------------
Inspired by the stability-threshold pattern in the JaspreetSingh-exe
reference repo:
  - A rolling window of the last BUFFER_SIZE frame predictions is kept.
  - A label is 'committed' to the sentence only when it is the majority
    label in >= STABILITY_THRESHOLD fraction of the window AND the model's
    predicted probability is >= CONFIDENCE_MIN.
  - A per-label cooldown (COOLDOWN_SECONDS) prevents the same sign from
    being repeated by simply holding the hand still.
  - After a commit, the window is cleared to start fresh detection.

Standalone Key Bindings (run_inference_loop)
--------------------------------------------
    SPACE  — speak current sentence via TTS
    D      — delete last character
    C      — clear entire sentence
    Q/ESC  — quit

Author : Silent Voice Team
Date   : 2026
"""

import time
import threading
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pyttsx3

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("model/classifier.pkl")

# ── Inference hyper-parameters ─────────────────────────────────────────────────
BUFFER_SIZE         = 10      # rolling window length — smaller = snappier commits
STABILITY_THRESHOLD = 0.75    # min fraction of window that must share a label
CONFIDENCE_MIN      = 0.40    # min predict_proba for any prediction to count
COOLDOWN_SECONDS    = 1.5     # min gap (s) before the same label commits again

# ── Inference resolution (width, height) used for MediaPipe ──────────────────
# Full 640x480 resolution for maximum landmark precision & finger landmark accuracy.
INFER_W, INFER_H = 640, 480

# ── UI / drawing constants ─────────────────────────────────────────────────────
FONT         = cv2.FONT_HERSHEY_SIMPLEX
COL_WHITE    = (255, 255, 255)
COL_DARK     = (20, 20, 30)
COL_GREEN    = (0, 200, 100)
COL_CYAN     = (100, 220, 255)
COL_GREY     = (140, 140, 140)
COL_ORANGE   = (0, 165, 255)

# Enable OpenCV-wide optimisations (SIMD / multi-thread)
cv2.setUseOptimized(True)
cv2.setNumThreads(4)


# ═══════════════════════════════════════════════════════════════════════════════
# SentenceBuffer — word-buffering / stability logic
# ═══════════════════════════════════════════════════════════════════════════════

class SentenceBuffer:
    """
    Accumulates stable ASL sign predictions into a running sentence string.

    A sign is considered *stable* when the same label dominates the rolling
    window of recent predictions.  This prevents noisy or transitional frames
    from polluting the output.

    Parameters
    ----------
    buffer_size : int
        Number of frames in the rolling window (default BUFFER_SIZE).
    threshold : float
        Fraction of window that must agree for a commit (default STABILITY_THRESHOLD).
    cooldown : float
        Minimum seconds before the same label can commit again (default COOLDOWN_SECONDS).
    """

    def __init__(
        self,
        buffer_size: int   = BUFFER_SIZE,
        threshold: float   = STABILITY_THRESHOLD,
        cooldown: float    = COOLDOWN_SECONDS,
    ) -> None:
        self.buffer_size = buffer_size
        self.threshold   = threshold
        self.cooldown    = cooldown

        self._window: list[str]  = []   # rolling list of recent labels
        self._sentence: str      = ""   # accumulated output text
        self._last_label: str | None = None
        self._last_commit_t: float   = 0.0

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def sentence(self) -> str:
        """The current accumulated sentence string."""
        return self._sentence

    # ── Core update ────────────────────────────────────────────────────────────

    def register_current(self, label: str | None) -> str | None:
        """
        Manually register the currently predicted label into the sentence.
        Called when the user presses TAB or R in Manual Register Mode.
        """
        if not label:
            return None

        if len(label) == 1:
            self._sentence += label
        else:
            self._sentence += label + " "

        self._last_label = label
        self._last_commit_t = time.time()
        self._window.clear()
        return label

    def push(self, label: str | None, confidence: float, manual_mode: bool = False) -> str | None:
        """
        Push latest prediction into window. In automatic mode, commits when stable.
        In manual mode, updates current prediction but waits for manual register.
        """
        if label is None or confidence < CONFIDENCE_MIN:
            self._window.clear()
            return None

        if manual_mode:
            # In manual mode, we track current sign but do not auto-commit
            return None

        # Append to rolling window, evict oldest entry when full
        self._window.append(label)
        if len(self._window) > self.buffer_size:
            self._window.pop(0)

        if len(self._window) < self.buffer_size:
            return None

        counts: dict[str, int] = {}
        for lbl in self._window:
            counts[lbl] = counts.get(lbl, 0) + 1
        top_label = max(counts, key=counts.get)
        top_frac  = counts[top_label] / self.buffer_size

        if top_frac < self.threshold:
            return None

        now = time.time()
        if top_label == self._last_label and (now - self._last_commit_t) < self.cooldown:
            return None

        return self.register_current(top_label)

    # ── Mutation helpers ───────────────────────────────────────────────────────

    def delete_last(self) -> None:
        """Remove the last character (or trailing space of a word) from sentence."""
        if self._sentence:
            self._sentence = self._sentence.rstrip()   # strip trailing space
            self._sentence = self._sentence[:-1]       # remove one char

    def clear(self) -> None:
        """Reset everything: sentence, window, and cooldown state."""
        self._sentence      = ""
        self._last_label    = None
        self._last_commit_t = 0.0
        self._window.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Model & MediaPipe helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """
    Load the trained Random Forest classifier from disk.

    Returns
    -------
    sklearn RandomForestClassifier

    Raises
    ------
    FileNotFoundError if model/classifier.pkl does not exist.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'.  "
            "Run train_model.py first to train and save the classifier."
        )
    return joblib.load(MODEL_PATH)


def build_hands(
    min_detection_conf: float = 0.70,
    min_tracking_conf: float  = 0.50,
    video_mode: bool = False,       # True = VIDEO mode (faster tracking between frames)
):
    """
    Create and return a configured MediaPipe Tasks HandLandmarker.

    Parameters
    ----------
    min_detection_conf : Minimum confidence for initial hand detection.
    min_tracking_conf  : Minimum confidence for landmark tracking.
    video_mode         : Use RunningMode.VIDEO (faster, uses inter-frame tracking).
                         Requires monotonically increasing timestamps passed to
                         get_prediction().  Default False = IMAGE mode (stateless).
    """
    import urllib.request
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = Path("hand_landmarker.task")
    if not model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, str(model_path))

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    run_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=run_mode,
        num_hands=1,
        min_hand_detection_confidence=min_detection_conf,
        min_hand_presence_confidence=min_tracking_conf,
    )
    return vision.HandLandmarker.create_from_options(options)


# ═══════════════════════════════════════════════════════════════════════════════
# Core inference
# ═══════════════════════════════════════════════════════════════════════════════

def get_prediction(
    frame_rgb: np.ndarray,
    hands,
    model,
    timestamp_ms: int | None = None,   # supply for VIDEO mode; None = IMAGE mode
) -> tuple[str | None, float, object | None]:
    """
    Run MediaPipe hand detection and RF classification on a single RGB frame.

    Parameters
    ----------
    frame_rgb    : RGB image (H x W x 3, uint8).  May be any resolution.
    hands        : HandLandmarker created by build_hands().
    model        : Trained RandomForestClassifier.
    timestamp_ms : Monotonically increasing millisecond timestamp.  Supply
                   this when using VIDEO mode (build_hands(video_mode=True)).
                   Pass None (default) for IMAGE mode.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

    if timestamp_ms is not None:
        detection_result = hands.detect_for_video(mp_image, int(timestamp_ms))
    else:
        detection_result = hands.detect(mp_image)

    if not detection_result or not hasattr(detection_result, "hand_landmarks") or not detection_result.hand_landmarks:
        return None, 0.0, detection_result

    # Pixel-space landmark extraction matching reference keypoint dataset preprocessing
    hand_lms = detection_result.hand_landmarks[0]
    h, w = frame_rgb.shape[:2]

    # Convert to pixel space coordinates
    px_coords = []
    for lm in hand_lms:
        px = min(int(lm.x * w), w - 1)
        py = min(int(lm.y * h), h - 1)
        px_coords.append([px, py])

    # Wrist relative subtraction in pixel space
    base_x, base_y = px_coords[0][0], px_coords[0][1]
    rel_flat = []
    for p in px_coords:
        rel_flat.append(p[0] - base_x)
        rel_flat.append(p[1] - base_y)

    # Max abs normalization (-1 to 1)
    max_val = max(map(abs, rel_flat)) if max(map(abs, rel_flat)) > 0 else 1.0

    row: list[float] = []
    for i in range(21):
        x_norm = rel_flat[i*2] / max_val
        y_norm = rel_flat[i*2+1] / max_val
        row.extend([x_norm, y_norm, 0.0])

    if len(row) != 63:
        return None, 0.0, detection_result

    proba      = model.predict_proba([row])[0]
    pred_idx   = int(np.argmax(proba))
    confidence = float(proba[pred_idx])
    label      = str(model.classes_[pred_idx])

    if confidence < CONFIDENCE_MIN:
        return None, 0.0, detection_result

    return label, confidence, detection_result


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing / overlay
# ═══════════════════════════════════════════════════════════════════════════════

def draw_overlay(
    frame: np.ndarray,
    mp_results,
    label: str | None,
    confidence: float,
    sentence: str,
    speaker_on: bool = False,
    manual_mode: bool = True,
) -> np.ndarray:
    """
    Draw a modernized glassmorphic UI onto *frame*:
      - Neon hand skeleton with joint markers
      - Top translucent HUD bar showing sentence text and live status pills
      - Bottom prediction card with confidence meter and register shortcut
      - Styled key bindings footer strip
    """
    h, w = frame.shape[:2]

    # Create transparent overlay buffer
    overlay = frame.copy()

    # ── Hand skeleton with neon glow ───────────────────────────────────────────
    if mp_results and hasattr(mp_results, "hand_landmarks") and mp_results.hand_landmarks:
        hand_lms = mp_results.hand_landmarks[0]
        CONNECTIONS = [
            (0,1),(1,2),(2,3),(3,4),       # Thumb
            (0,5),(5,6),(6,7),(7,8),       # Index
            (5,9),(9,10),(10,11),(11,12),  # Middle
            (9,13),(13,14),(14,15),(15,16),# Ring
            (13,17),(17,18),(18,19),(19,20),(0,17) # Pinky & Palm
        ]
        # Bone shadow / glow
        for p1, p2 in CONNECTIONS:
            pt1 = (int(hand_lms[p1].x * w), int(hand_lms[p1].y * h))
            pt2 = (int(hand_lms[p2].x * w), int(hand_lms[p2].y * h))
            cv2.line(frame, pt1, pt2, (40, 20, 20), 4, cv2.LINE_AA)
            cv2.line(frame, pt1, pt2, (255, 140, 0), 2, cv2.LINE_AA)
        # Joint points
        for lm in hand_lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # ── Top HUD bar (translucent dark panel) ───────────────────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 75), (15, 12, 28), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Accent line under HUD
    cv2.line(frame, (0, 75), (w, 75), (100, 70, 220), 2)

    # Sentence text
    sentence_hdr = "SENTENCE:"
    cv2.putText(frame, sentence_hdr, (16, 26), FONT, 0.45, (160, 160, 190), 1, cv2.LINE_AA)

    display_text = sentence[-50:] if len(sentence) > 50 else sentence
    cv2.putText(frame, display_text or "_", (16, 56), FONT, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    # Status Pills (Mode & Speaker)
    # Mode Pill
    m_text = "MANUAL [TAB/R]" if manual_mode else "AUTO DETECT"
    m_bg   = (160, 80, 20) if manual_mode else (20, 120, 60)
    cv2.rectangle(frame, (w - 300, 12), (w - 150, 42), m_bg, -1)
    cv2.rectangle(frame, (w - 300, 12), (w - 150, 42), (220, 220, 255), 1)
    cv2.putText(frame, m_text, (w - 292, 32), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Speaker Pill
    s_text = "SPEAKER ON" if speaker_on else "SPEAKER OFF"
    s_bg   = (20, 140, 70) if speaker_on else (40, 40, 60)
    cv2.rectangle(frame, (w - 140, 12), (w - 12, 42), s_bg, -1)
    cv2.rectangle(frame, (w - 140, 12), (w - 12, 42), (220, 220, 255), 1)
    cv2.putText(frame, s_text, (w - 132, 32), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Bottom Prediction Card (Glassmorphic) ──────────────────────────────────
    card_h = 85
    card_overlay = frame.copy()
    cv2.rectangle(card_overlay, (12, h - card_h - 12), (400, h - 12), (18, 18, 30), -1)
    cv2.addWeighted(card_overlay, 0.75, frame, 0.25, 0, frame)

    card_border = (0, 220, 120) if label else (70, 70, 90)
    cv2.rectangle(frame, (12, h - card_h - 12), (400, h - 12), card_border, 2)

    if label:
        # Recognized Sign Label
        cv2.putText(frame, f"SIGN: {label}", (24, h - card_h + 26),
                    FONT, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        # Confidence Bar
        bar_w = int(140 * confidence)
        cv2.rectangle(frame, (24, h - card_h + 38), (164, h - card_h + 46), (40, 40, 60), -1)
        cv2.rectangle(frame, (24, h - card_h + 38), (24 + bar_w, h - card_h + 46), (0, 220, 120), -1)
        cv2.putText(frame, f"{confidence:.0%}", (172, h - card_h + 45),
                    FONT, 0.42, (180, 255, 200), 1, cv2.LINE_AA)

        if manual_mode:
            cv2.rectangle(frame, (24, h - card_h + 52), (380, h - card_h + 74), (100, 60, 20), -1)
            cv2.putText(frame, "PRESS TAB or R TO REGISTER", (32, h - card_h + 67),
                        FONT, 0.45, (255, 220, 140), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Waiting for hand sign...", (24, h - card_h + 40),
                    FONT, 0.70, (140, 140, 160), 1, cv2.LINE_AA)

    # ── Footer Key Hints Strip ────────────────────────────────────────────────
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (415, h - card_h - 12), (w - 12, h - 12), (15, 12, 28), -1)
    cv2.addWeighted(footer_overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (415, h - card_h - 12), (w - 12, h - 12), (70, 70, 90), 1)

    hints = [
        "TAB / R : Register Sign     M : Toggle Mode     1-0 : Shortcuts",
        "SPACE   : Word Space        ENTER : Speak TTS   H   : Cheatsheet",
        "BKSP/D  : Erase Letter      C : Clear All       Q   : Quit",
    ]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (425, h - card_h + 12 + i * 22),
                    FONT, 0.38, (210, 210, 230), 1, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# Cheatsheet panel
# ═══════════════════════════════════════════════════════════════════════════════

def make_cheatsheet() -> np.ndarray:
    """
    Generate and return a 1000x620 BGR visual cheatsheet image with crisp,
    high-contrast 21-point hand skeleton diagrams for all letters & gestures.
    """
    import json
    W, H = 1000, 620
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (16, 16, 26)  # dark background

    landmarks_json = Path("landmarks_summary.json")
    landmark_dict = {}
    if landmarks_json.exists():
        try:
            with open(landmarks_json) as f:
                landmark_dict = json.load(f)
        except Exception:
            pass

    # Add open-palm hello gesture pose for cheatsheet diagram
    if "hello" not in landmark_dict:
        landmark_dict["hello"] = [
            0.0, 0.0, 0.0, -0.25, -0.15, 0.0, -0.45, -0.35, 0.0, -0.60, -0.55, 0.0, -0.70, -0.70, 0.0,
            -0.22, -0.45, 0.0, -0.25, -0.70, 0.0, -0.27, -0.88, 0.0, -0.28, -1.00, 0.0,
            -0.05, -0.45, 0.0, -0.05, -0.72, 0.0, -0.05, -0.90, 0.0, -0.05, -1.00, 0.0,
            0.12, -0.42, 0.0, 0.15, -0.68, 0.0, 0.17, -0.85, 0.0, 0.18, -0.96, 0.0,
            0.28, -0.38, 0.0, 0.32, -0.58, 0.0, 0.35, -0.74, 0.0, 0.38, -0.88, 0.0
        ]

    # ── Title bar ──────────────────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (W, 46), (32, 32, 52), -1)
    cv2.putText(img, "SILENT VOICE  -  VISUAL SIGN SKELETON CHEATSHEET",
                (16, 30), FONT, 0.75, COL_CYAN, 2, cv2.LINE_AA)
    cv2.putText(img, "[H] hide", (W - 95, 30), FONT, 0.45, COL_GREY, 1)
    cv2.line(img, (0, 46), (W, 46), (60, 60, 80), 1)

    def section_header(y: int, text: str, col) -> None:
        cv2.rectangle(img, (8, y - 16), (W - 8, y + 6), (28, 28, 44), -1)
        cv2.putText(img, text, (14, y), FONT, 0.55, col, 1, cv2.LINE_AA)

    # ── ASL Alphabet Hand Skeleton Diagrams ────────────────────────────────────
    section_header(66, "ASL ALPHABET (A-Z) & GESTURES  (21-Point Hand Skeleton Landmarks)", COL_GREEN)
    signs = list("ABCDEFGHIKLMNOPQRSTUVWXY") + ["hello"]
    cols_per_row = 13

    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),       # Thumb
        (0,5),(5,6),(6,7),(7,8),       # Index
        (5,9),(9,10),(10,11),(11,12),  # Middle
        (9,13),(13,14),(14,15),(15,16),# Ring
        (13,17),(17,18),(18,19),(19,20),(0,17) # Pinky & Palm
    ]

    for idx, s in enumerate(signs):
        col = idx % cols_per_row
        row = idx // cols_per_row
        cell_w, cell_h = 72, 90
        x = 10 + col * (cell_w + 4)
        y = 78 + row * (cell_h + 8)

        # Draw card background
        cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), (22, 30, 26), -1)
        cv2.rectangle(img, (x, y), (x + cell_w, y + cell_h), (0, 160, 70), 1)

        # Render high-contrast 21-point hand diagram inside the cell
        if s in landmark_dict:
            coords = landmark_dict[s]
            pts = []
            box_cx, box_cy = x + cell_w // 2, y + cell_h // 2 + 10
            scale = cell_w * 0.40

            for i in range(21):
                raw_x = coords[i*3]
                raw_y = coords[i*3+1]
                px = int(box_cx + raw_x * scale)
                py = int(box_cy + raw_y * scale)
                pts.append((px, py))

            # Draw bones with thick lines for visibility
            for p1, p2 in CONNECTIONS:
                cv2.line(img, pts[p1], pts[p2], (255, 140, 0), 2, cv2.LINE_AA)
            # Draw joints with vibrant yellow/cyan markers
            for px, py in pts:
                cv2.circle(img, (px, py), 3, (0, 220, 255), -1, cv2.LINE_AA)

        # Label character banner
        cv2.rectangle(img, (x, y), (x + cell_w, y + 22), (0, 110, 50), -1)
        lbl_x = x + cell_w // 2 - (6 if len(s) == 1 else 18)
        cv2.putText(img, s, (lbl_x, y + 16),
                    FONT, 0.55 if len(s) > 1 else 0.65, COL_WHITE, 2, cv2.LINE_AA)

    # ── Controls reference ─────────────────────────────────────────────────────
    section_header(465, "CONTROLS & KEYBOARD SHORTCUTS", (255, 220, 80))
    controls = [
        ("TAB / R",  "REGISTER currently detected sign to sentence (Manual Mode)"),
        ("M",        "Toggle Mode: MANUAL (press to register) vs AUTO DETECT"),
        ("SPACE",    "Insert word space"),
        ("ENTER",    "Speak full sentence aloud (TTS)"),
        ("T",        "Toggle auto-speaker ON / OFF"),
        ("D / BSP",  "Erase last letter / word"),
        ("C",        "Clear entire sentence"),
        ("1 - 0",    "Insert shortcut word (I/you/the/is/a/and/to/my/what/please)"),
    ]
    for idx, (k, desc) in enumerate(controls):
        col = idx % 2
        row = idx // 2
        x   = 12  + col * 490
        y   = 495 + row * 28
        cv2.rectangle(img, (x, y - 14), (x + 76, y + 4), (55, 55, 20), -1)
        cv2.putText(img, k, (x + 4, y),
                    FONT, 0.40, (255, 220, 70), 1, cv2.LINE_AA)
        cv2.putText(img, desc, (x + 84, y),
                    FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # Border
    cv2.rectangle(img, (0, 0), (W - 1, H - 1), (60, 60, 90), 1)
    return img



def speak_text(text: str) -> None:
    """
    Speak *text* aloud using pyttsx3 run in an isolated subprocess.

    On Windows, pyttsx3 uses the COM SAPI5 interface which requires the
    calling thread to be the process main thread.  Running in a subprocess
    guarantees a fresh main thread, avoiding the silent-failure that occurs
    when pyttsx3 is called from a Streamlit daemon thread.

    Falls back to PowerShell's built-in SAPI5 speech synthesizer if the
    subprocess approach fails for any reason.

    Parameters
    ----------
    text : The string to be spoken. Silently ignored if empty.
    """
    import subprocess, sys

    text = text.strip()
    if not text:
        return

    # Escape single quotes so they survive PowerShell string embedding
    safe_text = text.replace("'", "''")

    # Primary: pyttsx3 in a fresh subprocess
    script = (
        "import pyttsx3; e=pyttsx3.init(); "
        "e.setProperty('rate',150); "
        f"e.say('{safe_text}'); "
        "e.runAndWait(); e.stop()"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            timeout=30,
            capture_output=True,
        )
        if result.returncode == 0:
            return
    except Exception:
        pass

    # Fallback: Windows PowerShell SAPI5 (always available on Windows)
    try:
        ps_script = (
            f"Add-Type -AssemblyName System.speech; "
            f"$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate=-2; $s.Speak('{safe_text}')"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            timeout=30,
            capture_output=True,
        )
    except Exception:
        pass  # TTS best-effort; never crash the app


def speak_text_async(text: str) -> None:
    """
    Non-blocking wrapper around speak_text().

    Spawns a daemon thread so the caller is never blocked during TTS playback.
    speak_text() itself uses a subprocess internally, so there are no
    COM/main-thread issues even when called from Streamlit's rerun loop.
    """
    t = threading.Thread(target=speak_text, args=(text,), daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone inference loop  (python predict_live.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Common short words the user can insert by pressing a number key ───────────
WORD_SHORTCUTS: dict[int, str] = {
    ord("1"): "I",
    ord("2"): "you",
    ord("3"): "the",
    ord("4"): "is",
    ord("5"): "a",
    ord("6"): "and",
    ord("7"): "to",
    ord("8"): "my",
    ord("9"): "what",
    ord("0"): "please",
}


def run_inference_loop(camera_index: int = 0) -> None:
    """
    Blocking webcam-based inference loop with a live OpenCV window.

    Speed optimisations
    -------------------
    * MediaPipe runs in VIDEO mode — inter-frame landmark tracking is ~2x
      faster than full re-detection on every frame.
    * Inference is performed on a downscaled 320x240 copy of each frame;
      the display window always shows the full-resolution image.
    * OpenCV multi-thread and SIMD optimisations are enabled globally.

    Key Bindings
    ------------
    H           -- show / hide the sign cheatsheet window
    SPACE       -- speak sentence now
    T           -- toggle auto-speaker ON / OFF
    ENTER       -- insert a word space
    BACKSPACE/D -- erase last letter
    C           -- clear entire sentence
    1-9, 0      -- insert shortcut word (I/you/the/is/a/and/to/my/what/please)
    Q / ESC     -- quit
    """
    print("[INFO] Loading model ...")
    model = load_model()

    # VIDEO mode: MediaPipe tracks landmarks between frames — much faster
    print("[INFO] Initialising MediaPipe Hands (VIDEO mode) ...")
    hands = build_hands(video_mode=True)

    # Pre-generate the cheatsheet image (done once, reused every frame)
    cheatsheet_img = make_cheatsheet()
    CHEATSHEET_WIN = "Silent Voice -- Sign Cheatsheet  [H to hide]"

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera {camera_index}.  "
            "Check --camera index or webcam permissions."
        )
    # Hint the driver to use a high-definition capture resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    buffer       = SentenceBuffer()
    speaker_on   = False
    manual_mode  = True          # Manual mode default: user presses TAB or R to register sign
    show_sheet   = True          # cheatsheet visible by default
    start_time   = time.monotonic()
    fps_counter  = 0
    fps_display  = 0.0
    fps_timer    = time.monotonic()

    print("[INFO] Silent Voice live inference started.")
    print("[INFO] Controls:")
    print("  TAB / R     : REGISTER currently detected sign")
    print("  M           : toggle MANUAL / AUTO mode")
    print("  SPACE       : insert word space")
    print("  ENTER       : speak full sentence now")
    print("  T           : toggle auto-speaker ON/OFF")
    print("  BACKSPACE/D : erase last letter")
    print("  C           : clear entire sentence")
    print("  H           : show/hide visual cheatsheet")
    print("  1-9, 0      : insert shortcut word (I/you/the/is/a/and/to/my/what/please)")
    print("  Q / ESC     : quit")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # ── FPS counter ────────────────────────────────────────────────────────
        fps_counter += 1
        now = time.monotonic()
        if now - fps_timer >= 1.0:
            fps_display = fps_counter / (now - fps_timer)
            fps_counter = 0
            fps_timer   = now

        # ── Inference frame matching 4:3 aspect ratio ────────────────────────────
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame_rgb.shape[1], frame_rgb.shape[0]) != (INFER_W, INFER_H):
            infer_frame = cv2.resize(frame_rgb, (INFER_W, INFER_H))
        else:
            infer_frame = frame_rgb

        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        label, confidence, mp_results = get_prediction(
            infer_frame, hands, model, timestamp_ms=timestamp_ms
        )

        # Update sentence buffer
        committed = buffer.push(label, confidence, manual_mode=manual_mode)
        if committed:
            print(f"[BUFFER] Committed: '{committed}'  ->  sentence: '{buffer.sentence}'")
            if speaker_on:
                speak_text_async(committed)

        # ── Draw overlays on display frame ─────────────────────────────────────
        frame = draw_overlay(
            frame, mp_results, label, confidence,
            buffer.sentence, speaker_on=speaker_on, manual_mode=manual_mode,
        )

        cv2.putText(frame, f"FPS {fps_display:.0f}",
                    (10, 90), FONT, 0.50, COL_GREY, 1, cv2.LINE_AA)

        cv2.imshow("Silent Voice -- Live Recognition", frame)

        # ── Cheatsheet window ─────────────────────────────────────────────────
        if show_sheet:
            cv2.imshow(CHEATSHEET_WIN, cheatsheet_img)
        else:
            try:
                cv2.destroyWindow(CHEATSHEET_WIN)
            except Exception:
                pass

        # ── Key handling ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):         # Q / ESC -- quit
            break

        elif key in (9, ord("r"), ord("R")):          # TAB (9) or R -- REGISTER SIGN
            if label:
                reg = buffer.register_current(label)
                print(f"[REGISTER] Registered: '{reg}'  ->  sentence: '{buffer.sentence}'")
                if speaker_on:
                    speak_text_async(reg)
            else:
                print("[REGISTER] No sign detected to register.")

        elif key in (ord("m"), ord("M")):             # M -- toggle MANUAL/AUTO mode
            manual_mode = not manual_mode
            print(f"[MODE] Mode switched to: {'MANUAL (press TAB/R to register)' if manual_mode else 'AUTO DETECT'}")

        elif key in (ord("h"), ord("H")):             # H -- toggle cheatsheet
            show_sheet = not show_sheet
            print(f"[SHEET] Cheatsheet {'shown' if show_sheet else 'hidden'}")

        elif key == 32:                               # SPACE -- word space
            buffer._sentence += " "
            print(f"[SPACE] Added word break -> '{buffer.sentence}'")

        elif key == 13:                               # ENTER -- speak full sentence
            if buffer.sentence.strip():
                speak_text_async(buffer.sentence)
                print(f"[SPEAK] '{buffer.sentence}'")
            else:
                print("[SPEAK] Nothing to speak yet.")

        elif key in (ord("t"), ord("T")):             # T -- toggle auto-speaker
            speaker_on = not speaker_on
            print(f"[SPEAKER] Auto-speak toggled {'ON' if speaker_on else 'OFF'}")

        elif key in (8, ord("d"), ord("D")):          # BACKSPACE or D -- erase
            before = buffer.sentence
            buffer.delete_last()
            print(f"[ERASE] '{before}' -> '{buffer.sentence}'")

        elif key in (ord("c"), ord("C")):             # C -- clear all
            buffer.clear()
            print("[CLEAR] Sentence cleared.")

        elif key in WORD_SHORTCUTS:                   # 1-9, 0 -- shortcut word
            word = WORD_SHORTCUTS[key]
            buffer._sentence += word + " "
            print(f"[WORD] Inserted '{word}' -> '{buffer.sentence}'")
            if speaker_on:
                speak_text_async(word)

    cap.release()
    if hasattr(hands, "close"):
        hands.close()
    cv2.destroyAllWindows()
    print("[INFO] Inference stopped.")


if __name__ == "__main__":
    run_inference_loop()
