# Silent Voice

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/UI-OpenCV-5C3EE8)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/CV-MediaPipe-00b4d8)](https://google.github.io/mediapipe/)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](LICENSE)
[![Offline](https://img.shields.io/badge/Runs-100%25%20Offline-success)](#project-overview)

> A real-time, fully offline application that translates static ASL sign
> language gestures (A–Z alphabet + 15 common word signs) into text and
> optionally reads it aloud — powered by MediaPipe, scikit-learn, and OpenCV.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements & Setup](#requirements--setup)
3. [Vocabulary](#vocabulary)
4. [Step-by-Step Usage](#step-by-step-usage)
   - [Step 1 — Collect Training Data](#step-1--collect-training-data)
   - [Step 2 — Train the Model](#step-2--train-the-model)
   - [Step 3 — Run Live Inference](#step-3--run-live-inference)
5. [Project Structure](#project-structure)
6. [Key Design Decisions](#key-design-decisions)
7. [Troubleshooting](#troubleshooting)
8. [Roadmap](#roadmap)
9. [Credits & References](#credits--references)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview

| Field | Detail |
|---|---|
| **Name** | Silent Voice |
| **Type** | Software Engineering Mini-Project |
| **Category** | Accessibility / Computer Vision / Applied ML |
| **Mode** | Static gestures only (no motion-based signing) |
| **Connectivity** | 100% offline — no paid APIs, no cloud |

**Functional Requirements covered:**

| FR | Description |
|----|-------------|
| FR1 | Live webcam video capture |
| FR2 | 21-point hand landmark extraction via MediaPipe |
| FR3 | Random Forest classification into 41 classes |
| FR4 | On-screen text display of predicted sign |
| FR5 | Stable-prediction buffering into sentence |
| FR6 | Text-to-speech button (pyttsx3, offline) |
| FR7 | Data collection mode (`collect_data.py`) |
| FR8 | Model retraining from collected data (`train_model.py`) |

---

## Requirements & Setup

### Prerequisites

- Python 3.9 or higher
- A working webcam
- Windows / macOS / Linux

### Installation

```bash
# 1. Clone or download the project
cd SV

# 2. (Recommended) Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **pyttsx3 on Linux** requires `espeak` to be installed:
> ```bash
> sudo apt-get install espeak
> ```

---

## Vocabulary

**Alphabet (26 classes):** `A B C D E F G H I J K L M N O P Q R S T U V W X Y Z`

**Word signs (15 classes):**

| Key | Word | Key | Word | Key | Word |
|-----|------|-----|------|-----|------|
| `1` | hello | `2` | thanks | `3` | yes |
| `4` | no | `5` | help | `6` | please |
| `7` | sorry | `8` | water | `9` | food |
| `0` | stop | `.` | more | `,` | eat |
| `/` | drink | `;` | love | `'` | bathroom |

---

## Step-by-Step Usage

### Step 1 — Collect Training Data

```bash
python collect_data.py [--camera 0] [--samples 200]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--camera` | `0` | Webcam device index |
| `--samples` | `200` | Target number of samples per class |

**In the collection window:**

1. Press a **class key** to select the target class:
   - `A`–`Z` (or `a`–`z`) for alphabet signs
   - `1`–`0` and `.`, `,`, `/`, `;`, `'` for word signs (see table above)
2. Position your hand clearly in frame — the skeleton overlay confirms detection.
3. Press **`SPACE`** to capture one sample.
4. Press **`C`** to toggle **auto-capture mode** (~2 samples/second automatically).
5. Press **`D`** to delete the last recorded sample (undo).
6. Press **`Q`** or **`ESC`** to quit.

Samples are saved to `data/landmarks.csv` — you can re-run the script
multiple times; it appends without overwriting existing data.

> **Tip:** Collect at least 200 samples per class for good accuracy.
> Vary lighting, hand position, and distance slightly for robustness.

---

### Step 2 — Train the Model

```bash
python train_model.py
```

The script:
1. Loads `data/landmarks.csv`
2. Splits 80% train / 20% test (stratified)
3. Trains a `RandomForestClassifier(n_estimators=100)` — no scaling needed
4. Prints accuracy, per-class classification report, and confusion matrix
5. Saves the model to `model/classifier.pkl`

**Expected output:**
```
[INFO] Samples : 8200
[INFO] Features: 63  (21 landmarks × 3 axes)
[INFO] Classes : 41
...
[RESULT] Test Accuracy : 0.9512  (95.12%)
```

---

### Step 3 — Run Live Inference
 image.png

```bash
python predict_live.py
```

Opens an OpenCV window with:
- Live hand skeleton overlay
- Predicted sign name + confidence at the bottom
- Accumulated sentence at the top

**Key bindings:**

| Key | Action |
|-----|--------|
| `SPACE` | Speak current sentence (offline TTS) |
| `D` | Delete last character |
| `C` | Clear entire sentence |
| `Q` / `ESC` | Quit |

---

## Project Structure

```
Silent-voice---sign-language-translator/
├── collect_data.py       # FR7 — webcam data collection (41 classes)
├── train_model.py        # FR8 — Random Forest training + evaluation
├── predict_live.py       # FR1–FR6 — inference engine, SentenceBuffer, TTS, OpenCV UI
├── hand_landmarker.task   # MediaPipe hand-landmark model (bundled)
├── data/
│   ├── landmarks.csv         # Created by collect_data.py
│   └── landmarks_rich.csv    # Extended feature set (see landmarks_summary.json)
├── model/
│   └── classifier.pkl    # Created by train_model.py
├── assets/                # ASL reference charts used in the UI/docs
├── Silent_Voice_PRD.pdf   # Full product requirements document
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── README.md
```

---

## Key Design Decisions

### Feature Representation

Each sample is represented as **63 raw MediaPipe coordinates**
(`x, y, z` per landmark × 21 landmarks).  MediaPipe normalises these to
`[0, 1]` relative to the image frame, so no additional scaling is applied.
Random Forests are scale-invariant, making StandardScaler unnecessary.

### Prediction Stability Buffer

Raw frame-by-frame predictions are noisy.  `predict_live.SentenceBuffer`
uses a **rolling majority-vote window** (default: 15 frames):

- A sign is committed only when **≥ 80% of the window** agrees on a label.
- A **2-second cooldown** prevents the same sign from repeating while the
  hand is held still.
- The window is cleared after each commit, enforcing deliberate re-signing.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `FileNotFoundError: model/classifier.pkl` | Run `python train_model.py` first |
| `No module named 'mediapipe'` | Run `pip install -r requirements.txt` |
| Camera not found | Try `--camera 1` (or `2`) in collect/predict scripts |
| pyttsx3 silent on Linux | Install `espeak`: `sudo apt-get install espeak` |
| Low accuracy | Collect more samples; vary lighting and angles |

---

## Roadmap

Static-gesture recognition for a single sign language is the deliberate
scope of v1 (see the [PRD](Silent_Voice_PRD.pdf) for the full non-goals
list). Directions being considered for later versions:

- [ ] Dynamic/motion-based signs (full ASL grammar, two-handed signs)
- [ ] Additional sign systems (ISL, BSL)
- [ ] Packaged desktop build (PyInstaller) so non-technical users don't need Python installed
- [ ] Mobile-friendly build
- [ ] Confidence-calibrated UI feedback (e.g. "hold steadier" hints)

Have an idea or want to pick one of these up? See [Contributing](#contributing).

---

## Credits & References

Silent Voice was built independently, but a few open-source projects were
studied for implementation patterns during development:

- [VivanRajath/ASL](https://github.com/VivanRajath/ASL) — MediaPipe + Random
  Forest ASL alphabet recognition approach
- [CodingSamrat/Sign-Language-Recognition](https://github.com/CodingSamrat/Sign-Language-Recognition) —
  data-collection and keypoint-dataset preprocessing pattern

No code from these repositories is vendored in this project; they're
credited here for the ideas they contributed.

---

## Contributing

Bug reports, new word-sign data, and doc improvements are all welcome —
see [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and PR guidelines.

---

## License

All rights reserved — see [LICENSE](LICENSE). This code is not open source;
no permission is granted to copy, modify, or redistribute it without the
author's written consent.
