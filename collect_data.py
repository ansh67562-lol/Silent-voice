"""
collect_data.py
===============
Silent Voice — Data Collection Tool

Opens a live webcam window, detects hand landmarks with MediaPipe, and
saves captured samples to data/landmarks.csv for training.

Usage
-----
    python collect_data.py [--camera 0] [--samples 200]

Class Selection Keys
--------------------
Alphabet (26 classes):
    A–Z (or a–z)  →  class "A" through "Z"

Word signs (15 classes) — all unshifted, no conflict with letters:
    1 = hello     2 = thanks    3 = yes       4 = no        5 = help
    6 = please    7 = sorry     8 = water      9 = food      0 = stop
    . = more      , = eat       / = drink      ; = love      ' = bathroom

Action Keys
-----------
    SPACE  — capture one sample for the current class (hand must be visible)
    C      — toggle continuous auto-capture (~2 samples/sec)
    D      — delete the last recorded sample for the current class
    Q/ESC  — quit

Inspired by the mode-toggle / class-selection pattern from
references/Sign-Language-Recognition (CodingSamrat).

Author : Silent Voice Team
Date   : 2026
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "landmarks.csv"

# ── Feature schema ─────────────────────────────────────────────────────────────
# MediaPipe provides 21 hand landmarks, each with (x, y, z) → 63 features.
NUM_LANDMARKS = 21
FEATURE_COLS  = [f"{ax}{i}" for i in range(NUM_LANDMARKS) for ax in ("x", "y", "z")]
CSV_HEADER    = FEATURE_COLS + ["label"]   # 63 feature cols + 1 label col

# ── Vocabulary ─────────────────────────────────────────────────────────────────
# 26 alphabet letters: collected with keys A–Z (handled dynamically).
# 15 word signs: mapped to specific non-letter keys.
WORD_KEY_MAP: dict[int, str] = {
    ord("1"): "hello",    ord("2"): "thanks",  ord("3"): "yes",
    ord("4"): "no",       ord("5"): "help",    ord("6"): "please",
    ord("7"): "sorry",    ord("8"): "water",   ord("9"): "food",
    ord("0"): "stop",
    ord("."): "more",     ord(","): "eat",     ord("/"): "drink",
    ord(";"): "love",     ord("'"): "bathroom",
}

# ── UI constants ───────────────────────────────────────────────────────────────
PANEL_W       = 270   # width of the right-hand info sidebar (pixels)
FONT          = cv2.FONT_HERSHEY_SIMPLEX
COL_ACCENT    = (100, 220, 255)   # BGR: light cyan
COL_GREEN     = (0, 200, 100)
COL_RED       = (60, 60, 220)
COL_WHITE     = (255, 255, 255)
COL_GREY      = (140, 140, 140)
COL_DARK      = (20, 20, 30)
COL_ORANGE    = (0, 165, 255)


# ── CSV helpers ────────────────────────────────────────────────────────────────

def ensure_csv_header() -> None:
    """Create data/ directory and write the CSV header if the file is new."""
    DATA_DIR.mkdir(exist_ok=True)
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)


def load_class_counts() -> dict[str, int]:
    """Return {label: sample_count} for every class already in landmarks.csv."""
    counts: dict[str, int] = {}
    if not CSV_PATH.exists():
        return counts
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("label", "").strip()
            if lbl:
                counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def append_sample(landmark_values: list[float], label: str) -> None:
    """Append one 63-feature row (+ label) to landmarks.csv."""
    ensure_csv_header()
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow(landmark_values + [label])


def delete_last_sample(label: str) -> bool:
    """
    Remove the last row whose 'label' column matches *label* from landmarks.csv.

    Returns True if a row was deleted, False otherwise.
    """
    if not CSV_PATH.exists():
        return False
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.reader(f))

    # rows[0] is the header; search backwards through data rows
    for i in range(len(rows) - 1, 0, -1):
        if rows[i] and rows[i][-1].strip() == label:
            rows.pop(i)
            with open(CSV_PATH, "w", newline="") as f:
                csv.writer(f).writerows(rows)
            return True
    return False


# ── Landmark extraction ────────────────────────────────────────────────────────

def extract_landmark_row(hand_landmarks) -> list[float]:
    """
    Flatten MediaPipe hand landmarks into a list of 63 floats:
        [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

    Coordinates are already normalised to [0, 1] by MediaPipe.
    """
    row: list[float] = []
    for lm in hand_landmarks.landmark:
        row.extend([round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)])
    return row   # length 63


# ── UI drawing ─────────────────────────────────────────────────────────────────

def draw_sidebar(frame: "cv2.Mat",
                 current_class: str | None,
                 class_counts: dict[str, int],
                 target_samples: int,
                 continuous: bool,
                 hand_ok: bool,
                 fps: float) -> None:
    """
    Draw a semi-transparent info panel on the right edge of *frame* (in-place).

    Shows: current class, progress bar, hand-detection status, mode, FPS,
    and an abbreviated key-binding reference.
    """
    h, w = frame.shape[:2]
    x0 = w - PANEL_W

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), COL_DARK, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # ── Title ──────────────────────────────────────────────────────────────────
    cv2.putText(frame, "Silent Voice", (x0 + 10, 32), FONT, 0.75, COL_ACCENT, 2)
    cv2.putText(frame, "Data Collector",  (x0 + 10, 55), FONT, 0.45, (150, 150, 200), 1)
    cv2.line(frame, (x0 + 5, 65), (w - 5, 65), (50, 50, 70), 1)

    # ── Current class & sample count ──────────────────────────────────────────
    cls_txt   = current_class if current_class else "None"
    cls_color = COL_GREEN if current_class else (80, 80, 80)
    cv2.putText(frame, f"Class : {cls_txt}", (x0 + 10, 100), FONT, 0.65, cls_color, 2)

    count = class_counts.get(current_class, 0) if current_class else 0
    cv2.putText(frame, f"Samples: {count}/{target_samples}",
                (x0 + 10, 125), FONT, 0.50, COL_WHITE, 1)

    # Progress bar
    bar_left  = x0 + 10
    bar_right = w - 10
    bar_w     = bar_right - bar_left
    filled    = int(bar_w * min(count / max(target_samples, 1), 1.0))
    cv2.rectangle(frame, (bar_left, 135), (bar_right, 148), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_left, 135), (bar_left + filled, 148), COL_GREEN, -1)

    # ── Hand detection status ──────────────────────────────────────────────────
    hd_color = COL_GREEN if hand_ok else COL_RED
    hd_text  = "Hand : DETECTED" if hand_ok else "Hand : NOT FOUND"
    cv2.putText(frame, hd_text, (x0 + 10, 175), FONT, 0.47, hd_color, 1)

    # ── Capture mode ──────────────────────────────────────────────────────────
    mo_color = COL_ORANGE if continuous else (180, 180, 100)
    mo_text  = "Mode : AUTO-CAPTURE" if continuous else "Mode : MANUAL"
    cv2.putText(frame, mo_text, (x0 + 10, 198), FONT, 0.44, mo_color, 1)

    # ── FPS ───────────────────────────────────────────────────────────────────
    cv2.putText(frame, f"FPS  : {fps:4.0f}", (x0 + 10, 220), FONT, 0.44, COL_GREY, 1)

    # ── Key reference ─────────────────────────────────────────────────────────
    cv2.line(frame, (x0 + 5, 232), (w - 5, 232), (50, 50, 70), 1)
    ref_lines = [
        "-- Key Bindings --",
        "A-Z  : alphabet class",
        "1-9  : hello..food",
        "0    : stop",
        ".    : more   , : eat",
        "/    : drink  ; : love",
        "'    : bathroom",
        "SPACE: capture sample",
        "C    : toggle auto-cap",
        "D    : delete last row",
        "Q/ESC: quit",
    ]
    for idx, line in enumerate(ref_lines):
        color = (160, 160, 160) if idx > 0 else (200, 200, 100)
        cv2.putText(frame, line, (x0 + 8, 252 + idx * 18), FONT, 0.36, color, 1)

    # ── Total samples (bottom) ────────────────────────────────────────────────
    total = sum(class_counts.values())
    cv2.putText(frame, f"Total samples: {total}", (10, h - 12),
                FONT, 0.45, COL_GREY, 1)


# ── CLI argument parsing ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Silent Voice — Data Collection Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera",  type=int, default=0,   help="Webcam device index")
    p.add_argument("--samples", type=int, default=200, help="Target samples per class")
    return p.parse_args()


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── MediaPipe Tasks setup ──────────────────────────────────────────────────
    import urllib.request
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = Path("hand_landmarker.task")
    if not model_path.exists():
        print("[INFO] Downloading MediaPipe hand_landmarker.task model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, str(model_path))

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.70,
        min_hand_presence_confidence=0.50,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # ── Camera ─────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}. "
              "Try --camera 1 or check your webcam connection.")
        return

    # ── State ──────────────────────────────────────────────────────────────────
    current_class: str | None = None        # currently selected class label
    class_counts               = load_class_counts()
    continuous_mode            = False       # auto-capture toggle
    last_capture_t             = 0.0        # last auto-capture timestamp
    CAPTURE_INTERVAL           = 0.50       # seconds between auto-captures
    prev_frame_t               = time.time()

    print("[INFO] Silent Voice Data Collector started.")
    print("[INFO] Press a letter key (A-Z) or number key (1-0 / . , / ; ') "
          "to select a class, then SPACE to capture.")
    print("[INFO] Press 'C' to toggle continuous auto-capture, 'Q' to quit.")
    print(f"[INFO] Target samples per class: {args.samples}")
    print(f"[INFO] CSV output: {CSV_PATH.resolve()}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed — retrying.")
            continue

        # Mirror the frame for natural (selfie-style) interaction
        frame = cv2.flip(frame, 1)

        # ── FPS measurement ────────────────────────────────────────────────────
        now       = time.time()
        fps       = 1.0 / max(now - prev_frame_t, 1e-6)
        prev_frame_t = now

        # ── MediaPipe hand detection ───────────────────────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = landmarker.detect(mp_image)

        hand_detected = len(detection_result.hand_landmarks) > 0
        landmark_row: list[float] | None = None

        if hand_detected:
            hand_lms = detection_result.hand_landmarks[0]
            h, w = frame.shape[:2]
            # Draw landmarks & simple bone connections
            CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),       # Thumb
                (0,5),(5,6),(6,7),(7,8),       # Index
                (5,9),(9,10),(10,11),(11,12),  # Middle
                (9,13),(13,14),(14,15),(15,16),# Ring
                (13,17),(17,18),(18,19),(19,20),(0,17) # Pinky & Palm
            ]
            for p1, p2 in CONNECTIONS:
                pt1 = (int(hand_lms[p1].x * w), int(hand_lms[p1].y * h))
                pt2 = (int(hand_lms[p2].x * w), int(hand_lms[p2].y * h))
                cv2.line(frame, pt1, pt2, (255, 165, 0), 2)
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)

            # Pixel-space landmark extraction matching reference keypoint dataset preprocessing
            px_coords = []
            for lm in hand_lms:
                px = min(int(lm.x * w), w - 1)
                py = min(int(lm.y * h), h - 1)
                px_coords.append([px, py])

            base_x, base_y = px_coords[0][0], px_coords[0][1]
            rel_flat = []
            for p in px_coords:
                rel_flat.append(p[0] - base_x)
                rel_flat.append(p[1] - base_y)

            max_val = max(map(abs, rel_flat)) if max(map(abs, rel_flat)) > 0 else 1.0

            landmark_row = []
            for i in range(21):
                x_norm = rel_flat[i*2] / max_val
                y_norm = rel_flat[i*2+1] / max_val
                landmark_row.extend([round(x_norm, 6), round(y_norm, 6), 0.0])

        # ── Auto-capture logic ─────────────────────────────────────────────────
        if continuous_mode and current_class and hand_detected and landmark_row:
            if now - last_capture_t >= CAPTURE_INTERVAL:
                count = class_counts.get(current_class, 0)
                if count < args.samples:
                    append_sample(landmark_row, current_class)
                    class_counts[current_class] = count + 1
                    last_capture_t = now
                    print(f"[AUTO] {current_class} → "
                          f"{class_counts[current_class]}/{args.samples}")
                else:
                    # Target reached: turn off auto-capture for this class
                    print(f"[INFO] Target reached for '{current_class}'. "
                          "Auto-capture disabled.")
                    continuous_mode = False

        # ── Draw UI sidebar ────────────────────────────────────────────────────
        draw_sidebar(frame, current_class, class_counts,
                     args.samples, continuous_mode, hand_detected, fps)

        cv2.imshow("Silent Voice — Data Collector", frame)

        # ── Key handling ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 255:
            # No key pressed this frame
            continue

        # Quit
        if key in (ord("q"), ord("Q"), 27):     # 27 = ESC
            print("[INFO] Quitting data collector.")
            break

        # Toggle continuous auto-capture
        elif key in (ord("c"), ord("C")):
            continuous_mode = not continuous_mode
            state = "ON" if continuous_mode else "OFF"
            print(f"[INFO] Continuous capture: {state}")

        # Delete last sample for current class
        elif key in (ord("d"), ord("D")):
            if not current_class:
                print("[WARN] No class selected — nothing to delete.")
            elif delete_last_sample(current_class):
                class_counts = load_class_counts()   # reload after deletion
                print(f"[INFO] Deleted last sample for '{current_class}'. "
                      f"Remaining: {class_counts.get(current_class, 0)}")
            else:
                print(f"[WARN] No samples found for '{current_class}'.")

        # Manual single-sample capture (SPACE bar, ASCII 32)
        elif key == 32:
            if not current_class:
                print("[WARN] Select a class first (press A-Z or 1-0 etc.).")
            elif not hand_detected or landmark_row is None:
                print("[WARN] No hand detected — position your hand in frame.")
            else:
                count = class_counts.get(current_class, 0)
                if count < args.samples:
                    append_sample(landmark_row, current_class)
                    class_counts[current_class] = count + 1
                    print(f"[CAPTURE] {current_class} → "
                          f"{class_counts[current_class]}/{args.samples}")
                else:
                    print(f"[INFO] Target {args.samples} already reached "
                          f"for '{current_class}'.")

        # Alphabet class selection (a–z or A–Z)
        elif ord("a") <= key <= ord("z"):
            current_class   = chr(key).upper()
            continuous_mode = False
            print(f"[SELECT] Class → '{current_class}' "
                  f"(existing: {class_counts.get(current_class, 0)})")

        elif ord("A") <= key <= ord("Z"):
            current_class   = chr(key).upper()
            continuous_mode = False
            print(f"[SELECT] Class → '{current_class}' "
                  f"(existing: {class_counts.get(current_class, 0)})")

        # Word-sign class selection via number/punctuation keys
        elif key in WORD_KEY_MAP:
            current_class   = WORD_KEY_MAP[key]
            continuous_mode = False
            print(f"[SELECT] Class → '{current_class}' "
                  f"(existing: {class_counts.get(current_class, 0)})")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    total = sum(class_counts.values())
    print(f"[INFO] Session complete. Total samples in CSV: {total}")
    print(f"[INFO] File saved at: {CSV_PATH.resolve()}")


if __name__ == "__main__":
    main()
