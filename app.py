"""SV95C Analysis App — Gait Analysis for Ankle IMU Data.

A Streamlit web application providing a user-friendly interface to run the
gaitmap analysis pipeline on Apple Watch IMU data and calculate SV95C
(Stride Velocity 95th Centile) without writing code.

Context: MIT THINK proposal — ankle-worn Apple Watch measuring SV95C for home
mobility monitoring in Duchenne muscular dystrophy (DMD).
"""

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
from gaitmap.example_data import get_healthy_example_imu_data

from src.resampling import resample_imu_dataframe
from src.sensorlog import (
    AXIS_MAPPINGS,
    compute_validation_summary,
    parse_sensorlog_to_gaitmap,
    validate_sensorlog_columns,
)
from src.pipeline import (
    PipelineResult,
    run_pipeline,
)
from src.analysis import (
    MCID_MPS,
    compare_all_methods,
    format_summary_json,
)
from src.plots import (
    plot_raw_imu_with_strides,
    plot_velocity_histogram,
)

# Page configuration
st.set_page_config(
    page_title="SV95C Analysis App | Ankle IMU Gait Analysis",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_example_data_cached() -> Tuple[pd.DataFrame, float]:
    """Cache loading the built-in healthy example dataset."""
    df = get_healthy_example_imu_data()
    return df, 204.8


@st.cache_data
def run_pipeline_cached(
    _data: pd.DataFrame,
    sampling_rate_hz: float,
    segmenter_type: str,
    trajectory_method_type: str,
    sensor_side: str,
    method_name: str,
) -> PipelineResult:
    """Cached pipeline execution keyed by parameters and data shape/values."""
    return run_pipeline(
        data=_data,
        sampling_rate_hz=sampling_rate_hz,
        segmenter_type=segmenter_type,
        trajectory_method_type=trajectory_method_type,
        sensor_side=sensor_side,
        method_name=method_name,
    )


@st.cache_data
def run_comparison_cached(
    _data: pd.DataFrame,
    sampling_rate_hz: float,
    sensor_side: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, PipelineResult]]:
    """Cached execution of all four method configurations (A-D)."""
    return compare_all_methods(
        data=_data,
        sampling_rate_hz=sampling_rate_hz,
        sensor_side=sensor_side,
    )


def initialize_session_state():
    """Ensure session state variables exist."""
    if "raw_data" not in st.session_state:
        st.session_state["raw_data"] = None
    if "sampling_rate" not in st.session_state:
        st.session_state["sampling_rate"] = 204.8
    if "data_source_label" not in st.session_state:
        st.session_state["data_source_label"] = "Example Dataset"
    if "metadata" not in st.session_state:
        st.session_state["metadata"] = {}
    if "pipeline_result" not in st.session_state:
        st.session_state["pipeline_result"] = None
    if "comparison_result" not in st.session_state:
        st.session_state["comparison_result"] = None


def main():
    initialize_session_state()

    # App Header
    st.title("👟 SV95C Gait Analysis App")
    st.markdown(
        "**Stride Velocity 95th Centile (SV95C) Analysis for Ankle IMU Data**  \n"
        "*Pilot monitoring tool for the MIT THINK proposal (Duchenne Muscular Dystrophy mobility monitoring).*"
    )

    # Mandatory Research Disclaimer Banner
    st.info(
        "ℹ️ **Research & Pilot Output Notice:** This software is a laboratory tool intended solely for "
        "pre-clinical and pilot research evaluation. The computed SV95C value is an exploratory estimate, "
        "not a clinical diagnosis, certified medical device output, or health indicator.",
        icon="🔬",
    )

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        # 1. Data Source Selection
        st.subheader("1. Data Source")
        data_source = st.radio(
            "Select input data:",
            options=["Try Example Data", "Upload SensorLog CSV"],
            index=0,
            help="Use gaitmap's verified 204.8 Hz healthy adult dataset or upload an Apple Watch SensorLog CSV."
        )

        sensor_side = "left"
        selected_sensor_label = "both"

        if data_source == "Try Example Data":
            st.caption("Healthy adult, 2x20 m walk, foot-worn IMUs (left + right), 204.8 Hz, 38.7 s.")
            example_sensor_choice = st.selectbox(
                "Sensor Selection:",
                options=["Both Sensors (Left + Right)", "Left Sensor Only", "Right Sensor Only"],
                index=0,
            )

            if st.button("📥 Load Example Data", use_container_width=True):
                with st.spinner("Loading example dataset..."):
                    df_full, orig_rate = load_example_data_cached()
                    if example_sensor_choice == "Left Sensor Only":
                        df = df_full[["left_sensor"]]
                        sensor_side = "left"
                        selected_sensor_label = "Left Sensor"
                    elif example_sensor_choice == "Right Sensor Only":
                        df = df_full[["right_sensor"]]
                        sensor_side = "right"
                        selected_sensor_label = "Right Sensor"
                    else:
                        df = df_full
                        sensor_side = "left"
                        selected_sensor_label = "Both Sensors (Left + Right)"

                    st.session_state["raw_data"] = df
                    st.session_state["sampling_rate"] = orig_rate
                    st.session_state["data_source_label"] = f"Example Data ({selected_sensor_label})"
                    st.session_state["metadata"] = {
                        "source": "gaitmap healthy example dataset",
                        "sensors": selected_sensor_label,
                        "resampling_used": "None (original 204.8 Hz)",
                    }
                    st.session_state["pipeline_result"] = None
                    st.session_state["comparison_result"] = None
                    st.success("Example data loaded successfully!")

        else:
            # SensorLog CSV Upload
            st.caption("Upload a CSV file exported by the **SensorLog** iOS app on an Apple Watch.")
            uploaded_file = st.file_uploader(
                "Choose SensorLog CSV:",
                type=["csv"],
                help="Requires accelerometer & gyro timestamp and 3-axis acceleration/rotation columns."
            )

            ankle_side = st.selectbox(
                "Ankle Attachment Side:",
                options=["Left Ankle", "Right Ankle"],
                index=0,
                help="Select which ankle wore the Apple Watch strap."
            )
            sensor_side = "left" if ankle_side == "Left Ankle" else "right"

            axis_mapping_choice = st.selectbox(
                "Watch Axis Alignment:",
                options=list(AXIS_MAPPINGS.keys()),
                index=0,
                help="Watch coordinate frame mapping to gaitmap foot sensor frame. Default is unvalidated identity."
            )
            if not AXIS_MAPPINGS[axis_mapping_choice]["validated"]:
                st.caption("⚠️ *Note: Orientation unvalidated for this mapping. Check DTW detection rate.*")

            if uploaded_file is not None:
                try:
                    raw_csv_df = pd.read_csv(uploaded_file)
                    is_valid, missing_cols = validate_sensorlog_columns(raw_csv_df)
                    if not is_valid:
                        st.error(
                            f"❌ **Invalid SensorLog CSV format.**  \n"
                            f"Missing required column(s): `{', '.join(missing_cols)}`"
                        )
                    else:
                        val_summary = compute_validation_summary(raw_csv_df)
                        st.markdown("##### Upload Validation Summary")
                        col_v1, col_v2 = st.columns(2)
                        col_v1.metric("Row Count", f"{val_summary['row_count']:,}")
                        col_v2.metric("Detected Rate", f"{val_summary['detected_rate_hz']:.1f} Hz")
                        col_v1.metric("Duration", f"{val_summary['duration_s']:.1f} s")
                        col_v2.metric("Time Gaps (>2x dt)", f"{val_summary['time_gaps_count']}")

                        if val_summary["missing_value_count"] > 0:
                            st.warning(f"File contains {val_summary['missing_value_count']} NaN values in IMU columns (will be dropped).")

                        # Size guard (> 2,000,000 rows)
                        win_start = None
                        win_len = None
                        if val_summary["needs_windowing"]:
                            st.warning("⚠️ **File exceeds 2,000,000 rows (~5.5 hrs).** Full continuous multi-day recordings are out of scope for v1.")
                            st.markdown("**Select time window to analyze:**")
                            win_start = st.number_input("Start Minute", min_value=0.0, max_value=float(val_summary["duration_s"] / 60.0), value=0.0, step=1.0)
                            win_len = st.number_input("Window Length (minutes)", min_value=0.5, max_value=120.0, value=10.0, step=1.0)

                        if st.button("🔄 Process & Load CSV", use_container_width=True):
                            with st.spinner("Converting units, remapping axes, and resampling..."):
                                s_name = "left_sensor" if sensor_side == "left" else "right_sensor"
                                gait_df, final_rate, meta = parse_sensorlog_to_gaitmap(
                                    df=raw_csv_df,
                                    sensor_name=s_name,
                                    axis_mapping=axis_mapping_choice,
                                    window_start_min=win_start,
                                    window_length_min=win_len,
                                    target_sampling_rate_hz=100.0,
                                )
                                st.session_state["raw_data"] = gait_df
                                st.session_state["sampling_rate"] = final_rate
                                st.session_state["data_source_label"] = f"Uploaded CSV ({uploaded_file.name})"
                                st.session_state["metadata"] = meta
                                st.session_state["pipeline_result"] = None
                                st.session_state["comparison_result"] = None
                                st.success("SensorLog data prepared and ready for analysis!")

                except Exception as e:
                    st.error(f"Error parsing uploaded file: {str(e)}")

        st.divider()

        # 2. Algorithm Selection
        st.subheader("2. Pipeline Algorithms")
        segmenter_choice = st.selectbox(
            "Stride Segmentation:",
            options=["BarthDtw", "ConstrainedBarthDtw"],
            index=0,
            help="Dynamic Time Warping template matching on body-frame gyroscope signal."
        )
        st.selectbox(
            "Gait-Event Detection:",
            options=["RamppEventDetection"],
            index=0,
            disabled=True,
            help="Detects initial contact, terminal contact, and minimum velocity points."
        )
        trajectory_choice = st.selectbox(
            "Trajectory Reconstruction:",
            options=["RtsKalman with ZUPT", "ForwardBackwardIntegration (No Kalman/ZUPT)"],
            index=0,
            help="Kalman filter with zero-velocity updates (ZUPT) or double integration."
        )

        st.divider()

        # 3. Sampling Rate Control
        st.subheader("3. Sampling Rate")
        current_sr = st.session_state.get("sampling_rate", 204.8)
        chosen_sr = st.number_input(
            "Sampling Rate (Hz):",
            min_value=10.0,
            max_value=1000.0,
            value=float(current_sr),
            step=1.0,
            format="%.1f",
            help="Explicit sampling rate passed to gaitmap. Must match the data grid."
        )
        st.session_state["sampling_rate"] = chosen_sr

        st.divider()

        # 4. Execution Buttons
        st.subheader("4. Run")
        col_btn1, col_btn2 = st.columns(2)
        run_clicked = col_btn1.button("🚀 Run Analysis", type="primary", use_container_width=True)
        compare_clicked = col_btn2.button("📊 Compare Methods", use_container_width=True)

    # --- MAIN CONTENT AREA ---
    data = st.session_state.get("raw_data")

    # If no data loaded yet, show initial empty state with quick demo button
    if data is None:
        st.info("👋 **No dataset loaded.** Click **'Load Example Data'** in the sidebar to run the smoke test demo, or upload an Apple Watch SensorLog CSV.")
        demo_col1, demo_col2 = st.columns([1, 3])
        if demo_col1.button("⚡ Quick Start: Load Example Data & Run", type="primary"):
            with st.spinner("Loading example data..."):
                df_full, orig_rate = load_example_data_cached()
                st.session_state["raw_data"] = df_full
                st.session_state["sampling_rate"] = orig_rate
                st.session_state["data_source_label"] = "Example Data (Both Sensors)"
                st.session_state["metadata"] = {
                    "source": "gaitmap healthy example dataset",
                    "sensors": "Both Sensors",
                    "resampling_used": "None (original 204.8 Hz)",
                }
                # Run immediately
                traj_key = "RtsKalman"
                result = run_pipeline_cached(
                    _data=df_full,
                    sampling_rate_hz=orig_rate,
                    segmenter_type="BarthDtw",
                    trajectory_method_type=traj_key,
                    sensor_side="left",
                    method_name="Method A (Main Chain)",
                )
                st.session_state["pipeline_result"] = result
                st.rerun()
        return

    # Status Bar
    num_rows = len(data)
    active_rate = st.session_state["sampling_rate"]
    duration_sec = num_rows / active_rate if active_rate > 0 else 0
    sensors_present = list(data.columns.get_level_values(0).unique()) if isinstance(data.columns, pd.MultiIndex) else ["Single Sensor"]

    st.markdown(
        f"**Data Loaded:** `{st.session_state['data_source_label']}` &nbsp;|&nbsp; "
        f"**Rows:** `{num_rows:,}` &nbsp;|&nbsp; "
        f"**Duration:** `{duration_sec:.1f} s` &nbsp;|&nbsp; "
        f"**Rate:** `{active_rate:.1f} Hz` &nbsp;|&nbsp; "
        f"**Sensors:** `{', '.join(sensors_present)}`"
    )

    # Handle Analysis Execution
    if run_clicked:
        traj_key = "RtsKalman" if "Kalman" in trajectory_choice else "ForwardBackwardIntegration"
        with st.spinner(f"Running gaitmap pipeline ({segmenter_choice} + {traj_key})..."):
            result = run_pipeline_cached(
                _data=data,
                sampling_rate_hz=active_rate,
                segmenter_type=segmenter_choice,
                trajectory_method_type=traj_key,
                sensor_side=sensor_side,
                method_name=f"{segmenter_choice} + {traj_key}",
            )
            st.session_state["pipeline_result"] = result
            st.session_state["comparison_result"] = None

    if compare_clicked:
        with st.spinner("Running all 4 method configurations (Methods A, B, C, D)..."):
            comp_df, spread_metrics, method_results = run_comparison_cached(
                _data=data,
                sampling_rate_hz=active_rate,
                sensor_side=sensor_side,
            )
            st.session_state["comparison_result"] = (comp_df, spread_metrics, method_results)
            # Default pipeline result to Method A for the primary display
            st.session_state["pipeline_result"] = method_results["A"]

    res: Optional[PipelineResult] = st.session_state.get("pipeline_result")

    # If no run yet, prompt the user
    if res is None:
        st.write("")
        st.write("👉 Click **'Run Analysis'** in the sidebar to execute the gait analysis chain.")
        return

    st.divider()

    # --- ZERO STRIDES HANDLING (Criterion 9.6) ---
    if res.zero_strides:
        st.error("🚨 **Zero Strides Detected**")
        st.markdown(
            "The segmentation algorithm could not identify any valid stride templates in this recording. "
            "Please review the diagnostic checks below before re-running:"
        )
        for cause in res.possible_causes:
            st.markdown(f"- {cause}")
        return

    # --- PRIMARY RESULTS DISPLAY ---
    p_stats = res.pooled_stats
    sv95_val = p_stats["sv95c"]

    st.markdown("### 📊 Primary Gait Metrics")

    # Prominent Headline Metrics Cards
    top_col1, top_col2 = st.columns([1.5, 1])
    with top_col1:
        st.metric(
            label="🏆 SV95C (Stride Velocity 95th Centile)",
            value=f"{sv95_val:.3f} m/s" if pd.notna(sv95_val) else "N/A",
            help="Primary mobility outcome: 95th percentile of valid stride velocities."
        )
    with top_col2:
        param_n = p_stats['parameterized_strides']
        det_n = p_stats['detected_strides']
        pct = (param_n / det_n * 100) if det_n > 0 else 0
        st.metric(
            label="Stride Reconstruction Yield",
            value=f"{param_n} / {det_n}",
            delta=f"{pct:.0f}% valid parameterized",
            delta_color="normal"
        )

    sec_col1, sec_col2, sec_col3, sec_col4 = st.columns(4)
    sec_col1.metric(
        label="Mean Velocity",
        value=f"{p_stats['mean_velocity']:.3f} m/s" if pd.notna(p_stats["mean_velocity"]) else "N/A",
    )
    sec_col2.metric(
        label="Median Velocity",
        value=f"{p_stats['median_velocity']:.3f} m/s" if pd.notna(p_stats["median_velocity"]) else "N/A",
    )
    sec_col3.metric(
        label="Mean Stride Length",
        value=f"{p_stats['mean_stride_length']:.3f} m" if pd.notna(p_stats["mean_stride_length"]) else "N/A",
    )
    sec_col4.metric(
        label="Mean Stride Time",
        value=f"{p_stats['mean_stride_time']:.3f} s" if pd.notna(p_stats["mean_stride_time"]) else "N/A",
    )

    # Per-Sensor breakdown if multiple sensors present
    if len(res.per_sensor_stats) > 1:
        with st.expander("🔍 View Per-Sensor Breakdown (Left vs Right)", expanded=False):
            sensor_cols = st.columns(len(res.per_sensor_stats))
            for i, (s_name, s_stat) in enumerate(res.per_sensor_stats.items()):
                with sensor_cols[i]:
                    st.markdown(f"**{s_name.replace('_', ' ').title()}**")
                    st.write(f"- **SV95C:** `{s_stat['sv95c']:.3f} m/s`" if pd.notna(s_stat['sv95c']) else "- **SV95C:** N/A")
                    st.write(f"- **Mean Velocity:** `{s_stat['mean_velocity']:.3f} m/s`" if pd.notna(s_stat['mean_velocity']) else "- **Mean Velocity:** N/A")
                    st.write(f"- **Strides:** `{s_stat['parameterized_strides']} valid / {s_stat['detected_strides']} detected`")
                    st.write(f"- **Mean Length:** `{s_stat['mean_stride_length']:.3f} m`" if pd.notna(s_stat['mean_stride_length']) else "- **Mean Length:** N/A")
                    st.write(f"- **Mean Time:** `{s_stat['mean_stride_time']:.3f} s`" if pd.notna(s_stat['mean_stride_time']) else "- **Mean Time:** N/A")

    st.divider()

    # --- CHARTS SECTION ---
    st.markdown("### 📈 Visualizations")
    chart_tab1, chart_tab2 = st.tabs(["Stride Velocity Histogram", "Raw IMU Signals & Stride Markers"])

    with chart_tab1:
        # Extract valid velocities for histogram
        valid_vels = res.per_stride_df["gait_velocity_mps"].dropna().tolist()
        hist_fig = plot_velocity_histogram(valid_vels, sv95_val)
        st.plotly_chart(hist_fig, use_container_width=True)

    with chart_tab2:
        sensor_to_plot = st.selectbox(
            "Select sensor signal to inspect:",
            options=list(res.stride_list.keys()),
            index=0,
            key="plot_sensor_select"
        )
        signal_fig = plot_raw_imu_with_strides(
            imu_data=data,
            stride_df=res.per_stride_df,
            sampling_rate_hz=active_rate,
            sensor_name=sensor_to_plot,
            max_duration_s=30.0,
        )
        st.plotly_chart(signal_fig, use_container_width=True)

    st.divider()

    # --- METHOD COMPARISON SECTION (Slice 3) ---
    comp_data = st.session_state.get("comparison_result")
    if comp_data is not None:
        comp_df, spread_metrics, _ = comp_data
        st.markdown("### 🔬 Method Comparison (Methods A - D)")
        st.caption(
            "Evaluation across DTW constraints, trajectory algorithms, and 100 Hz polyphase resampling."
        )

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        spread_val = spread_metrics["spread_mps"]
        is_sub_mcid = spread_metrics["is_below_mcid"]

        c_col1, c_col2 = st.columns([1, 2.5])
        c_col1.metric(
            label="Algorithm Spread (Max - Min)",
            value=f"{spread_val:.3f} m/s",
            delta=f"Threshold MCID: {MCID_MPS:.2f} m/s",
            delta_color="inverse" if is_sub_mcid else "normal",
        )
        c_col2.info(f"💡 {spread_metrics['interpretation']}")
        st.divider()

    # --- PER-STRIDE DATA TABLE ---
    st.markdown("### 📋 Per-Stride Results Table")
    st.dataframe(
        res.per_stride_df,
        use_container_width=True,
        hide_index=True,
    )

    # --- EXPORT & DOWNLOAD (Slice 4) ---
    st.markdown("### 💾 Export Results")
    down_col1, down_col2, _ = st.columns([1, 1, 2])

    # 1. Per-stride CSV export
    csv_bytes = res.per_stride_df.to_csv(index=False).encode("utf-8")
    down_col1.download_button(
        label="📥 Download Strides (CSV)",
        data=csv_bytes,
        file_name="sv95c_strides.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # 2. Summary JSON export
    summary_dict = format_summary_json(
        result=res,
        extra_metadata=st.session_state.get("metadata", {})
    )
    json_bytes = json.dumps(summary_dict, indent=2).encode("utf-8")
    down_col2.download_button(
        label="📥 Download Summary (JSON)",
        data=json_bytes,
        file_name="sv95c_summary.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()

    # --- DATA QUALITY PANEL (Slice 4) ---
    with st.expander("🛡️ Data Quality & Pipeline Diagnostics", expanded=False):
        dq_col1, dq_col2, dq_col3 = st.columns(3)
        dq_col1.write(f"- **Detected Strides:** {p_stats['detected_strides']}")
        dq_col1.write(f"- **Parameterized Strides:** {p_stats['parameterized_strides']}")
        dq_col1.write(f"- **NaN Strides:** {p_stats['nan_strides']}")

        dq_col2.write(f"- **Sampling Rate Used:** {res.sampling_rate_hz:.1f} Hz")
        meta = st.session_state.get("metadata", {})
        dq_col2.write(f"- **Resampling:** {meta.get('resampling_used', 'None')}")
        dq_col2.write(f"- **Axis Orientation:** {meta.get('axis_mapping', 'Standard FSF')}")

        dq_col3.write(f"- **Segmenter:** {res.method_name}")
        dq_col3.write(f"- **Orientation Validated:** {meta.get('orientation_validated', False)}")
        dq_col3.write(f"- **Zero-Stride Detected:** {res.zero_strides}")

    # Footer and License
    st.markdown("---")
    st.caption(
        "SV95C Analysis App v1.0.0 | Built for the MIT THINK Proposal pilot study.  \n"
        "Core gait analysis algorithms provided by [gaitmap](https://github.com/mad-lab-fau/gaitmap) (MIT License) "
        "and [gaitmap_mad](https://github.com/mad-lab-fau/gaitmap_mad) (AGPL-3.0 License).  \n"
        "In compliance with AGPL-3.0, source code for this application is accessible at: "
        "[https://github.com/mit-think-dmd/sv95c-app](https://github.com/mit-think-dmd/sv95c-app)"
    )


if __name__ == "__main__":
    main()
