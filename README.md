# SV95C Gait Analysis App 👟

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A Streamlit web application providing a user-friendly laboratory tool to run the **gaitmap** gait-analysis pipeline on Apple Watch IMU data and calculate **SV95C (Stride Velocity 95th Centile)** without writing Python code.

---

## 📌 Context: MIT THINK Proposal

* **Application:** Ankle-worn Apple Watch measuring SV95C for passive home mobility monitoring in Duchenne muscular dystrophy (DMD).
* **Role:** Pre-clinical pilot analysis tool for researchers and students.
* **Disclaimer:** **Research and pilot tool only.** Not a medical device and not a clinical diagnostic system. The SV95C values shown are computed estimates for research purposes.

---

## ✨ Features

1. **Slice 1 — Demo Mode (Instant Smoke Test):**
   * One-click demo loading gaitmap's built-in healthy adult dataset (2x20 m walk, foot-worn IMUs, 204.8 Hz, 38.7 s).
   * Exact reproducibility of verified benchmark numbers: Left SV95C `1.373 m/s`, Right SV95C `1.410 m/s`, Pooled SV95C `1.393 m/s` (52 parameterized strides).
2. **Slice 2 — SensorLog CSV Upload:**
   * Accepts CSV exports from the iOS **SensorLog** app on Apple Watch.
   * Auto-validates accelerometer (`accelerometerAccelerationX/Y/Z` in G) and gyroscope (`gyroRotationX/Y/Z` in rad/s) columns with timestamp alignment.
   * Converts acceleration to $m/s^2$ ($\times 9.81$).
   * Interactive axis-mapping dropdown (defaulting to identity labelled *"orientation unvalidated"*).
   * File summary panel (sampling rate, duration, row count, time gaps, NaNs).
   * Size guard: warns and requires time window selection if file exceeds 2,000,000 rows (~5.5 hours at 100 Hz).
3. **Slice 3 — Algorithm Selection & Method Comparison:**
   * Selectable segmentation (`BarthDtw`, `ConstrainedBarthDtw`), gait-event detection (`RamppEventDetection`), and trajectory reconstruction (`RtsKalman with ZUPT`, `ForwardBackwardIntegration`).
   * **"Compare All Methods"** button executes the 4 standard configurations (Methods A–D) side-by-side, reports parameter spread (`max - min`), and evaluates consistency against the **0.10 m/s Minimum Clinically Important Difference (MCID)** (Servais et al. 2022).
4. **Slice 4 — Export & Data Quality:**
   * Download per-stride CSV (`s_id`, start/end sample & time, stride length, stride time, gait velocity).
   * Download summary JSON (sampling rates, per-sensor & pooled metrics, settings used).
   * Diagnostic data-quality panel detailing detected vs parameterized strides, NaNs, sampling rates, and resampling methods.
   * Graceful zero-stride detection banner with 4 specific diagnostic recommendations.

---

## 📐 Pipeline Specifications (Methods A–D)

* **Method A (Main Chain):** `BarthDtw` + `RamppEventDetection` + `RtsKalman` (ZUPT)
* **Method B:** `ConstrainedBarthDtw` + `RamppEventDetection` + `RtsKalman`
* **Method C:** `BarthDtw` + `RamppEventDetection` + `StrideLevelTrajectory` default (`ForwardBackwardIntegration`, no Kalman/ZUPT)
* **Method D:** Main chain executed on data polyphase-resampled to 100 Hz

### ⚠️ Resampling Normative Rule
* Resample **ONLY** with polyphase filtering (`scipy.signal.resample_poly`).
* For 204.8 Hz $\to$ 100 Hz: `up=125, down=256` (exact $204.8 / 100 = 256 / 125$).
* **Never use linear interpolation** (`numpy.interp` or `pandas.DataFrame.interpolate`) on IMU data. Linear interpolation smooths out angular velocity peaks, causing DTW stride segmentation to detect zero strides.

---

## 📂 Repository Structure

```
sv95c-app/
├── app.py                 # Streamlit entry point, layout, and UI state
├── src/
│   ├── __init__.py
│   ├── sensorlog.py       # SensorLog CSV parser, validation, and unit conversion
│   ├── resampling.py      # Strict polyphase resampling helpers
│   ├── pipeline.py        # gaitmap pipeline execution and method definitions
│   ├── analysis.py        # Summary statistics, MCID spread, and export formatters
│   └── plots.py           # Plotly interactive histogram and IMU stride plots
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py   # Full test suite covering acceptance criteria 9.1-9.6
├── requirements.txt       # Pinned dependencies for Python 3.12
└── README.md
```

---

## 🚀 Quickstart & Local Setup

### Prerequisites
* Python 3.12 (recommended)

### 1. Clone the repository
```bash
git clone https://github.com/hutianle/sv95c-app.git
cd sv95c-app
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv

# Windows (Command Prompt / PowerShell):
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Acceptance Tests

Run the test suite verifying all 6 acceptance criteria from section 9:
```bash
pytest -v tests/test_pipeline.py
```

Expected verification results:
* `test_9_1_method_a_reference_numbers`: Left SV95C `1.373 m/s`, Right SV95C `1.410 m/s`, Pooled SV95C `1.393 m/s` (exact match to 3 decimals).
* `test_9_2_method_comparison_spread_and_mcid`: Method B pooled `1.393 m/s`, C pooled `1.374 m/s`, D pooled `1.392 m/s`; spread `0.019 m/s` < `0.10 m/s` MCID.
* `test_9_3_single_sensor_left_path`: Left-only single sensor yields 28 detected, 26 parameterized strides, SV95C `1.373 m/s`.
* `test_9_4_resampling_regression_and_no_linear_interp`: 100 Hz polyphase yields pooled SV95C `1.392 m/s` and zero linear interpolation calls anywhere in codebase.
* `test_9_5_upload_validation`: Missing SensorLog columns raise descriptive error; valid 100 Hz data preserves sampling grid without re-resampling.
* `test_9_6_zero_stride_handling`: Zero detected strides produces diagnostic advice banner without application crashes.

---

## ☁️ Deployment: Streamlit Community Cloud

1. Push your repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"** and select your repository:
   * **Repository:** `hutianle/sv95c-app`
   * **Branch:** `main`
   * **Main file path:** `app.py`
   * **Python version:** `3.12`
4. Click **"Deploy!"**. No secrets, API keys, or environment variables are required.

---

## 📄 License and Open Source Attribution

* **gaitmap core:** MIT License (FAU Machine Learning and Data Analytics Lab).
* **gaitmap_mad:** GNU Affero General Public License v3.0 (AGPL-3.0).
* **sv95c-app:** AGPL-3.0.
* In compliance with Section 13 of the AGPL-3.0, if this application is made accessible as a public network service, the full source code is made freely available to all users through the UI footer link and this repository.
