# Hand Gesture Caption App

Real-time assistive communication for people who cannot speak. Show hand gestures to the camera and your message appears as **live captions** — designed to work alongside **Zoom**, **Google Meet**, and **Microsoft Teams**.

## Two ways to run it

| | Web app (`web/`) | Desktop app (`main.py`) |
|---|---|---|
| **For** | Anyone, via a URL | Capturing data, training, benchmarking |
| **Install** | None | Python + MediaPipe + OpenCV |
| **Inference** | In-browser (WASM + exported MLP) | Python |
| **Camera data** | Never leaves the device | Local |

The web app is the shippable product; the Python side is the training toolchain
that produces the model it runs.

### Web app features

- **Live captions** with a large serif display, plus the in-progress buffer shown inline
- **Speech output** — speak each committed word, or read the whole transcript aloud
- **Recognition modes** — All / Words / Letters; narrowing to Letters stops `2` and `V` competing
- **Presenter window** — a clean caption-only window to screen-share into a call
- **Session log** with timestamps, copy, and download
- **Undo** (⌘Z) on every edit, plus adjustable hold and pause timing
- **Gesture guide** with searchable hand-shape descriptions, flagged for motion letters and look-alike shapes
- Light/dark themes, camera picker, and settings that persist across reloads

### Run the web app locally

```bash
python3 tools/devserver.py
```

Then open <http://localhost:4173>. Camera access needs `localhost` or HTTPS.

**If the page looks stale after an update, hard-reload once** — `Cmd+Shift+R`
(macOS) or `Ctrl+Shift+R` (Windows/Linux). A cache entry created before the dev
server started sending `no-store` is never revalidated, so the browser will keep
serving it until you bypass the cache once. After that it stays fresh: the dev
server sends `no-store` and asset URLs carry a `?v=` version.

### Deploy it

It is a static site — no server, no build step. `vercel.json` is included:

```bash
npx vercel deploy --prod
```

Netlify, Cloudflare Pages, or GitHub Pages work the same way; publish the `web/`
directory. Hosting cost is zero and it scales to any number of users, because
every visitor's browser does its own inference.

### Regenerate the web model after retraining

The browser loads JSON, not joblib. After training, re-export and re-verify:

```bash
python tools/export_web_assets.py            # joblib -> web/public/model/*.json
python tools/check_web_parity.py             # asserts JS == Python
```

`check_web_parity.py` runs 200 random hands through both pipelines and fails if
features, confidences, or labels diverge. Run it before any deploy.

## What You Can Sign

| Category | Coverage |
|----------|----------|
| **Letters** | Full alphabet **A–Z** (ASL fingerspelling) |
| **Numbers** | **0–9** |
| **Words & phrases** | 50+ essentials: Hello, Thank you, Help, Water, Emergency, I love you, and more |

Spell any word letter-by-letter, use quick word gestures, or combine both in one message.

## How It Works

1. **Custom recognizer** combines rules with the trained landmark model.
2. **Letters** you sign are buffered into a word (shown as `spelling: HEL_`).
3. **Auto-space** — pause briefly (~0.7s) and the word is committed with a space after it.
4. **Auto-correct** — buffered letters are spell-checked (`H E L P` → `help`).
5. **Captions** update live for your video call.

## Standout Features

- **Personal profile calibration**: collect profile-specific samples and train a personalized model.
- **Active-learning loop in live app**: save confirmed predictions or relabel mistakes as new training samples.
- **Benchmark report generation**: export confusion matrix and metrics, with automatic delta vs previous benchmark.

## Quick Start

```bash
cd "/Applications/Projects/Hand Gesture Project"
source venv/bin/activate
pip install -r requirements.txt

# Best for video calls — large caption window + minimal overlay
python main.py --conference --caption-only

# Run with a personal profile model/dataset
python main.py --profile myname --conference --caption-only
```

## Using With Zoom / Meet / Teams

1. **Start the app** with conference mode:
   ```bash
   python main.py --conference --caption-only
   ```

2. **Position the "Captions for Meet/Zoom" window** at the bottom of your screen (like built-in meeting captions).

3. **Join your call** and share your camera as usual. Others see your video; you read/sign into the app.

4. **Optional — screen share captions**: Share only the caption window so everyone sees your text:
   - Zoom: Share → Window → select **Captions for Meet/Zoom**
   - Meet: Present → A window → select the caption window

5. **Copy transcript** anytime with `V` to paste into chat.

### Recommended Setup

- Plain background, good lighting, hand fully in frame
- Use **SPELL mode** (`M` to cycle) when spelling names or uncommon words
- Use **WORDS mode** for fast phrases (Hello, Thank you, Help)
- Use **ALL mode** (default) for mixed conversation

## ML + Auto-Correct

The app uses **machine learning** to recognize gestures more accurately:

| Component | Role |
|-----------|------|
| `gesture_mlp.joblib` | Custom neural network on 21 hand landmarks |
| Rule-based classifier | ASL letters, numbers, phrases (fallback) |

**Auto-spacing:** Pause briefly (~0.7s) after spelling letters — the app commits the word and adds a space.

**Auto-correct:** Fingerspelled buffers are corrected against an English dictionary (e.g. `T H A N K` → `thank`).

Retrain the custom model anytime:
```bash
python train_gesture_model.py
```

Profile-specific training:
```bash
python train_gesture_model.py --profile myname --quality-check --show-report
```

## Add Your Own Custom Gestures

Use this workflow for fully custom labels and model behavior.

1. Capture samples for each gesture label (collect at least 100-200 per label):

```bash
python capture_gesture_dataset.py --label NEED_HELP --target 220
python capture_gesture_dataset.py --label CALL_DOCTOR --target 220
python capture_gesture_dataset.py --label THANK_YOU --target 220

# Collect into a specific profile
python capture_gesture_dataset.py --profile myname --label NEED_HELP --target 220
```

2. Train from the captured dataset:

```bash
python train_gesture_model.py --min-samples-per-label 20 --show-report
```

Training now applies automatic balancing + augmentation on the training split:

- Minority labels are upsampled to `--balance-target-per-label` (default `220`, capped by the largest class).
- Synthetic samples are created by adding feature-space noise controlled by `--augment-noise-std` (default `0.08`).

Example with explicit tuning:

```bash
python train_gesture_model.py \
    --min-samples-per-label 20 \
    --balance-target-per-label 260 \
    --augment-noise-std 0.06 \
    --show-report
```

Run a quality check before training:

```bash
python check_gesture_dataset.py --min-samples-per-label 20
```

Integrate quality checks directly into training:

```bash
python train_gesture_model.py --quality-check
python train_gesture_model.py --quality-check --strict-quality
```

Guided profile calibration (multi-label session):

```bash
python calibrate_profile.py --profile myname --per-label 90
```

Generate a benchmark report and confusion matrix:

```bash
python generate_benchmark_report.py --profile myname
```

3. Run the app normally. Any dataset label not in the built-in vocabulary is still shown as readable text (for example, `CALL_DOCTOR` appears as `Call Doctor`).

### Dataset Format

Samples are stored in `data/gesture_landmarks.jsonl`, one JSON object per line:

```json
{"label":"NEED_HELP","handedness":"Right","landmarks":[[x,y,z], ... 21 points]}
```

### Capture Controls

- `C`: capture one sample when a hand is detected
- `A`: toggle auto-capture mode
- `Q`: quit capture window

### Live Active-Learning Controls

- `K`: save current predicted gesture as a confirmed training sample
- `R`: open relabel mode, type correct label, then press Enter to save correction

## Controls

| Key | Action |
|-----|--------|
| Hold gesture | Add letter or word to captions |
| `Space` | Insert a space |
| `Backspace` | Delete last character |
| `C` | Clear transcript |
| `V` | Copy transcript to clipboard |
| `M` | Cycle mode: ALL → WORDS → SPELL |
| `T` | Toggle caption-only window (for Meet/Zoom overlay) |
| `F` | Toggle conference mode (cleaner view) |
| `H` | Show/hide hand skeleton overlay |
| `G` | Show/hide gesture reference guide |
| `K` | Save current prediction as a training sample |
| `R` | Relabel current frame and save correction sample |
| `Q` | Quit |

## Recognition Modes

| Mode | Best for |
|------|----------|
| **ALL** | Everyday mixed use — words, letters, numbers |
| **WORDS** | Fast conversation with common phrases |
| **SPELL** | Fingerspelling names, places, any word |

## Full Alphabet (A–Z)

| Letter | Hand shape |
|--------|------------|
| A | Fist, thumb to side |
| B | Flat hand, fingers up, thumb tucked |
| C | Curved hand |
| D | Index up, others curled |
| E | Fingers curled, thumb tucked |
| F | Thumb–index touch, 3 fingers up |
| G | Index points sideways |
| H | Index + middle sideways together |
| I | Pinky up |
| J | Pinky up (move for J motion) |
| K | Index + middle up, thumb between |
| L | Thumb + index (L shape) |
| M | Thumb under 3 fingers |
| N | Thumb under 2 fingers |
| O | Fingertips touch (O shape) |
| P | Like K, pointing down |
| Q | Thumb + index pointing down |
| R | Index + middle crossed |
| S | Closed fist |
| T | Thumb between index and middle |
| U | Index + middle together up |
| V | Index + middle spread (peace) |
| W | Index + middle + ring up |
| X | Index hooked/bent |
| Y | Thumb + pinky (shaka) |
| Z | Index up (move for Z motion) |

## Numbers (0–9)

| Number | Hand shape |
|--------|------------|
| 0 | O-shape (fingertips touch) |
| 1 | Index finger |
| 2 | Peace sign (V) |
| 3 | Three fingers up |
| 4 | Four fingers, thumb tucked |
| 5 | Open palm |
| 6–9 | Standard ASL number shapes |

## Common Words

Hello, Hi, Goodbye, Thank you, Please, Sorry, Yes, No, Help, Water, Food, Bathroom, Emergency, I love you, I understand, I don't understand, One moment please, Can you hear me, and more. Press `G` in the app for the on-screen guide.

## Project Structure

```
Hand Gesture Project/
├── main.py
├── capture_gesture_dataset.py
├── calibrate_profile.py
├── check_gesture_dataset.py
├── generate_benchmark_report.py
├── requirements.txt
├── README.md
└── src/
    ├── asl_alphabet.py      # A–Z recognition
    ├── asl_numbers.py       # 0–9 recognition
    ├── common_phrases.py    # 50+ word/phrase gestures
    ├── finger_geometry.py   # Hand shape analysis
    ├── gesture_ensemble.py  # Rules + custom model voting
    ├── ml_gesture_model.py  # Custom neural network
    ├── word_processor.py    # Auto-space + autocorrect
    ├── hand_tracker.py
    ├── caption_renderer.py
    └── conference_ui.py     # Zoom/Meet-friendly UI
models/
    └── gesture_mlp.joblib   # Trained landmark classifier
data/
    └── gesture_landmarks.jsonl  # Custom training samples (created by capture script)
profiles/
    └── <profile>/
        ├── gesture_landmarks.jsonl
        ├── gesture_mlp.joblib
        └── reports/
```

## Requirements

- Python 3.9+
- Webcam
- macOS, Linux, or Windows

## Recognition

### The bundled model is broken

`models/gesture_mlp.joblib` is degenerate. Probed across 3000 hand poses:

| measure | value |
|---|---|
| inputs answered `HELP` | **61%** |
| labels never predicted | **20 of 47** (incl. A D E M O R S T U W X 0 2 6 9) |
| mean top probability | 0.570 (gate is 0.56) |

It answers `HELP` at 0.94 confidence for a plain V sign. Its training set is
missing and its architecture does not match `_build_classifier()`, so it can be
neither retrained nor evaluated. Reproduce with `tools/diagnose_model.py`.

**The web app therefore does not use it by default.** Recognition runs on the
ported geometric rules, with the model available behind a toggle.

### What the web app uses instead

1. **Your calibration samples**, when you have recorded any (highest priority)
2. **Geometric rules** — `web/src/rules.js`, a port of `asl_alphabet.py`,
   `asl_numbers.py` and the phrase rules, verified against the Python originals
   by `tools/check_rules_parity.py` (0 mismatches over 600 hands)
3. **The bundled model** — off by default, and even when on must clear 0.8

### Personal calibration

1. Pick a gesture under *Teach your hand*.
2. Hold the shape and press **Record** — 15 samples, moving your hand slightly.
3. From 5 samples up, your own examples are matched first. The live readout
   shows `· yours` when a reading came from them.

Samples are 73-d feature vectors in `localStorage` — the same features the model
uses, so the two are directly comparable. Nothing is uploaded.

The gesture guide flags the shapes a still-image recogniser genuinely cannot
separate (`O`/`0`, `V`/`2`, `F`/`9`, a held `J`/`I`) and notes that the shipped
rules pinch the wrong finger for `7` and `8`. Switching to Letters or Words mode
removes the cross-category collisions.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q          # feature invariants, buffering, datasets
node --test "web/test/*.test.mjs"   # calibration k-NN, fusion policy
python tools/check_web_parity.py    # JS features/model == Python
python tools/check_rules_parity.py  # JS rules == Python rules
python tools/diagnose_model.py      # is a trained model actually usable?
```

Covers feature-extraction invariants (translation/scale/handedness), the
fingerspelling buffer and autocorrect, dataset validation, and profile path
sanitization.
