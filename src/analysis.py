"""Statistical analysis, method comparison, and data export formatting."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.pipeline import (
    PipelineResult,
    run_method_a,
    run_method_b,
    run_method_c,
    run_method_d,
)

# Minimum Clinically Important Difference for SV95C in DMD (Servais et al. 2022)
MCID_MPS: float = 0.10


def format_summary_json(
    result: PipelineResult,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Format full pipeline run metadata and results into a JSON-serializable dictionary."""
    summary = {
        "app": "SV95C Analysis App",
        "version": "1.0.0",
        "project": "MIT THINK Proposal — Apple Watch Ankle SV95C Monitoring",
        "disclaimer": "Research and pilot tool only. Not a medical device or diagnostic system.",
        "method_name": result.method_name,
        "sampling_rate_hz": result.sampling_rate_hz,
        "zero_strides": result.zero_strides,
        "pooled_results": {
            "sv95c_mps": round(result.pooled_stats["sv95c"], 4) if pd.notna(result.pooled_stats["sv95c"]) else None,
            "mean_velocity_mps": round(result.pooled_stats["mean_velocity"], 4) if pd.notna(result.pooled_stats["mean_velocity"]) else None,
            "median_velocity_mps": round(result.pooled_stats["median_velocity"], 4) if pd.notna(result.pooled_stats["median_velocity"]) else None,
            "detected_strides": result.pooled_stats["detected_strides"],
            "parameterized_strides": result.pooled_stats["parameterized_strides"],
            "nan_strides": result.pooled_stats["nan_strides"],
            "mean_stride_length_m": round(result.pooled_stats["mean_stride_length"], 4) if pd.notna(result.pooled_stats["mean_stride_length"]) else None,
            "mean_stride_time_s": round(result.pooled_stats["mean_stride_time"], 4) if pd.notna(result.pooled_stats["mean_stride_time"]) else None,
        },
        "per_sensor_results": {
            sensor: {
                "sv95c_mps": round(stats["sv95c"], 4) if pd.notna(stats["sv95c"]) else None,
                "mean_velocity_mps": round(stats["mean_velocity"], 4) if pd.notna(stats["mean_velocity"]) else None,
                "median_velocity_mps": round(stats["median_velocity"], 4) if pd.notna(stats["median_velocity"]) else None,
                "detected_strides": stats["detected_strides"],
                "parameterized_strides": stats["parameterized_strides"],
                "nan_strides": stats["nan_strides"],
                "mean_stride_length_m": round(stats["mean_stride_length"], 4) if pd.notna(stats["mean_stride_length"]) else None,
                "mean_stride_time_s": round(stats["mean_stride_time"], 4) if pd.notna(stats["mean_stride_time"]) else None,
            }
            for sensor, stats in result.per_sensor_stats.items()
        }
    }

    if extra_metadata:
        summary["metadata"] = extra_metadata

    return summary


def compare_all_methods(
    data: pd.DataFrame,
    sampling_rate_hz: float,
    sensor_side: str = "left"
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, PipelineResult]]:
    """Execute Methods A, B, C, and D on the dataset and return comparison metrics.

    Parameters
    ----------
    data : pd.DataFrame
        IMU dataset in gaitmap MultiSensorData format.
    sampling_rate_hz : float
        Sampling rate in Hz.
    sensor_side : str, default 'left'
        'left' or 'right' side for single-sensor data.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any], Dict[str, PipelineResult]]
        - comparison_table_df: summary comparison table
        - spread_metrics: max, min, spread, mcid comparison
        - results_by_method: dictionary mapping method key to PipelineResult
    """
    res_a = run_method_a(data, sampling_rate_hz, sensor_side=sensor_side)
    res_b = run_method_b(data, sampling_rate_hz, sensor_side=sensor_side)
    res_c = run_method_c(data, sampling_rate_hz, sensor_side=sensor_side)
    res_d = run_method_d(data, sampling_rate_hz, sensor_side=sensor_side)

    methods = [
        ("Method A (BarthDtw + RtsKalman)", res_a),
        ("Method B (ConstrainedBarthDtw + RtsKalman)", res_b),
        ("Method C (BarthDtw + ForwardBackward)", res_c),
        ("Method D (Main chain @ 100 Hz)", res_d),
    ]

    rows = []
    sv95_values = []
    for label, res in methods:
        p_stats = res.pooled_stats
        sv95 = p_stats["sv95c"]
        if pd.notna(sv95):
            sv95_values.append(sv95)

        rows.append({
            "Method": label,
            "Sampling Rate (Hz)": f"{res.sampling_rate_hz:.1f}",
            "Detected Strides": p_stats["detected_strides"],
            "Parameterized Strides": p_stats["parameterized_strides"],
            "Mean Velocity (m/s)": round(p_stats["mean_velocity"], 3) if pd.notna(p_stats["mean_velocity"]) else np.nan,
            "Median Velocity (m/s)": round(p_stats["median_velocity"], 3) if pd.notna(p_stats["median_velocity"]) else np.nan,
            "SV95C (m/s)": round(sv95, 3) if pd.notna(sv95) else np.nan,
            "Mean Length (m)": round(p_stats["mean_stride_length"], 3) if pd.notna(p_stats["mean_stride_length"]) else np.nan,
            "Mean Time (s)": round(p_stats["mean_stride_time"], 3) if pd.notna(p_stats["mean_stride_time"]) else np.nan,
        })

    comparison_df = pd.DataFrame(rows)

    if len(sv95_values) > 0:
        max_sv95 = float(np.max(sv95_values))
        min_sv95 = float(np.min(sv95_values))
        spread = float(max_sv95 - min_sv95)
        is_below_mcid = spread < MCID_MPS
    else:
        max_sv95 = np.nan
        min_sv95 = np.nan
        spread = np.nan
        is_below_mcid = False

    spread_metrics = {
        "max_sv95c": max_sv95,
        "min_sv95c": min_sv95,
        "spread_mps": spread,
        "mcid_mps": MCID_MPS,
        "is_below_mcid": is_below_mcid,
        "interpretation": (
            f"The spread across methods is {spread:.3f} m/s, which is well below the "
            f"{MCID_MPS:.2f} m/s Minimum Clinically Important Difference (MCID). "
            "This confirms algorithmic consistency across DTW constraints, trajectory algorithms, and 100 Hz resampling."
            if is_below_mcid else
            f"The spread across methods is {spread:.3f} m/s (MCID threshold is {MCID_MPS:.2f} m/s)."
        )
    }

    results_by_method = {
        "A": res_a,
        "B": res_b,
        "C": res_c,
        "D": res_d,
    }

    return comparison_df, spread_metrics, results_by_method
