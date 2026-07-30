# Contributing to Silent Voice

Thanks for your interest in improving Silent Voice! This is a small, focused
accessibility project, and contributions of all sizes are welcome.

## Ways to contribute

- **New word signs** — use `collect_data.py` to record samples for a gesture
  that isn't in the current 41-class vocabulary, then open a PR with the new
  data under `data/` and an updated label list.
- **Bug fixes** — anything from webcam edge cases to TTS quirks on a
  particular OS.
- **Docs** — clearer setup steps, platform-specific troubleshooting, etc.
- **New sign languages** — ISL/BSL support is explicitly out of scope for v1
  but tracked as a future direction; a proof-of-concept is welcome as a
  separate module rather than a rewrite of the ASL path.

## Development setup

```bash
git clone https://github.com/ansh67562-lol/Silent-voice---sign-language-translator.git
cd Silent-voice---sign-language-translator
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Before opening a PR

1. Run the app locally (`python predict_live.py`) and confirm your change
   doesn't break the live prediction loop.
2. If you touched `train_model.py` or the feature extraction logic in
   `predict_live.py`/`collect_data.py`, retrain the model and report the
   validation accuracy in your PR description.
3. Keep commits scoped — one logical change per PR is easier to review.

## Code style

The codebase favors explicit, well-commented functions over cleverness
(see the module docstrings at the top of each file for the intent). Please
match that style rather than introducing new abstractions unless they
clearly pay for themselves.

## Attribution

If you bring in ideas or code patterns from another open-source project,
credit it in your PR description and in a code comment, the way the
existing codebase credits its inspirations in the README.
