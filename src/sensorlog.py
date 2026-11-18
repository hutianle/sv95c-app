"""SensorLog CSV parsing, validation, and conversion to gaitmap DataFrame format.

Apple Watch SensorLog format specifications:
Accelerometer:
  - accelerometerTimestamp_sinceReboot (s)
  - accelerometerAccelerationX (G)
  - accelerometerAccelerationY (G)
  - accelerometerAccelerationZ (G)
Gyroscope:
  - gyroTimestamp_sinceReboot (s)
  - gyroRotationX (rad/s)
  - gyroRotationY (rad/s)
  - gyroRotationZ (rad/s)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.resampling import resample_imu_dataframe


REQUIRED_COLUMNS: List[str] = [
    "accelerometerTimestamp_sinceReboot",
    "accelerometerAccelerationX",
    "accelerometerAccelerationY",
    "accelerometerAccelerationZ",
    "gyroTimestamp_sinceReboot",
    "gyroRotationX",
    "gyroRotationY",
    "gyroRotationZ",
]

# Standard gravity constant for G -> m/s^2
G_TO_MPS2: float = 9.81

# Size guard limit (rows): ~5.5 hours at 100 Hz
MAX_UNWINDOWED_ROWS: int = 2_000_000

# Predefined axis mapping options
AXIS_MAPPINGS: Dict[str, Dict[str, str]] = {
    "Identity (Default - orientation unvalidated)": {
        "acc_x": "acc_x", "acc_y": "acc_y", "acc_z": "acc_z",
        "gyr_x": "gyr_x", "gyr_y": "gyr_y", "gyr_z": "gyr_z",
        "validated": False,
        "description": "Standard watch coordinate frame without axis remapping. Orientation is unvalidated."
    },
    "Lateral Ankle - Crown Forward (+Y -> +X, +X -> +Y, +Z -> +Z)": {
        "acc_x": "acc_y", "acc_y": "acc_x", "acc_z": "acc_z",
        "gyr_x": "gyr_y", "gyr_y": "gyr_x", "gyr_z": "gyr_z",
        "validated": False,
        "description": "Swaps X and Y axes for lateral ankle placement with crown facing forward."
    },
    "Anterior Ankle - Screen Facing Forward (+X -> +X, +Y -> +Y, -Z -> +Z)": {
        "acc_x": "acc_x", "acc_y": "acc_y", "acc_z": "-acc_z",
        "gyr_x": "gyr_x", "gyr_y": "gyr_y", "gyr_z": "-gyr_z",
        "validated": False,
        "description": "Inverts Z axis for anterior ankle mount with watch face pointing forward."
    },
    "Invert Vertical (+Z Inverted)": {
        "acc_x": "acc_x", "acc_y": "acc_y", "acc_z": "-acc_z",
        "gyr_x": "gyr_x", "gyr_y": "gyr_y", "gyr_z": "-gyr_z",
        "validated": False,
        "description": "Inverts the Z axis."
    }
}


def validate_sensorlog_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Check if all required SensorLog columns exist in DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded CSV DataFrame.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, missing_columns)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def compute_validation_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute data summary, sampling rate, duration, and time gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Valid SensorLog DataFrame.

    Returns
    -------
    Dict[str, Any]
        Summary dictionary containing row_count, duration_s, detected_rate_hz,
        time_gaps_count, max_gap_s, missing_value_count, and needs_windowing.
    """
    is_valid, missing = validate_sensorlog_columns(df)
    if not is_valid:
        raise ValueError(f"Invalid SensorLog format. Missing required column(s): {', '.join(missing)}")

    row_count = len(df)
    t_acc = df["accelerometerTimestamp_sinceReboot"].to_numpy(dtype=float)

    # Missing values in required columns
    missing_value_count = int(df[REQUIRED_COLUMNS].isna().sum().sum())

    # Duration and sampling rate
    if row_count > 1:
        dt = np.diff(t_acc)
        # Filter non-positive or extreme dt for rate detection
        positive_dt = dt[dt > 0]
        median_dt = float(np.median(positive_dt)) if len(positive_dt) > 0 else 0.01
        detected_rate_hz = float(1.0 / median_dt) if median_dt > 0 else 0.0
        duration_s = float(t_acc[-1] - t_acc[0])

        # Gaps where dt exceeds 2x the median dt
        gap_mask = dt > (2.0 * median_dt)
        time_gaps_count = int(np.sum(gap_mask))
        max_gap_s = float(np.max(dt)) if len(dt) > 0 else 0.0
    else:
        duration_s = 0.0
        detected_rate_hz = 0.0
        time_gaps_count = 0
        max_gap_s = 0.0

    needs_windowing = row_count > MAX_UNWINDOWED_ROWS

    return {
        "row_count": row_count,
        "duration_s": duration_s,
        "detected_rate_hz": detected_rate_hz,
        "time_gaps_count": time_gaps_count,
        "max_gap_s": max_gap_s,
        "missing_value_count": missing_value_count,
        "needs_windowing": needs_windowing,
    }


def apply_axis_mapping(
    acc_x: np.ndarray,
    acc_y: np.ndarray,
    acc_z: np.ndarray,
    gyr_x: np.ndarray,
    gyr_y: np.ndarray,
    gyr_z: np.ndarray,
    mapping_name: str = "Identity (Default - orientation unvalidated)"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply axis remapping to accelerometer and gyroscope signals."""
    if mapping_name not in AXIS_MAPPINGS:
        mapping_name = "Identity (Default - orientation unvalidated)"

    raw_signals = {
        "acc_x": acc_x, "acc_y": acc_y, "acc_z": acc_z,
        "-acc_x": -acc_x, "-acc_y": -acc_y, "-acc_z": -acc_z,
        "gyr_x": gyr_x, "gyr_y": gyr_y, "gyr_z": gyr_z,
        "-gyr_x": -gyr_x, "-gyr_y": -gyr_y, "-gyr_z": -gyr_z,
    }

    mapping = AXIS_MAPPINGS[mapping_name]
    out_acc_x = raw_signals[mapping["acc_x"]]
    out_acc_y = raw_signals[mapping["acc_y"]]
    out_acc_z = raw_signals[mapping["acc_z"]]
    out_gyr_x = raw_signals[mapping["gyr_x"]]
    out_gyr_y = raw_signals[mapping["gyr_y"]]
    out_gyr_z = raw_signals[mapping["gyr_z"]]

    return out_acc_x, out_acc_y, out_acc_z, out_gyr_x, out_gyr_y, out_gyr_z


def parse_sensorlog_to_gaitmap(
    df: pd.DataFrame,
    sensor_name: str = "left_sensor",
    axis_mapping: str = "Identity (Default - orientation unvalidated)",
    window_start_min: Optional[float] = None,
    window_length_min: Optional[float] = None,
    target_sampling_rate_hz: float = 100.0
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """Parse SensorLog DataFrame, convert units, align, remap axes, and resample to 100 Hz.

    Parameters
    ----------
    df : pd.DataFrame
        SensorLog DataFrame from CSV.
    sensor_name : str, default 'left_sensor'
        Name to give the single ankle sensor in the gaitmap MultiIndex.
    axis_mapping : str
        Key from AXIS_MAPPINGS.
    window_start_min : float, optional
        Start minute for window slicing if file is large.
    window_length_min : float, optional
        Duration in minutes for window slicing.
    target_sampling_rate_hz : float, default 100.0
        Target rate for uniform grid.

    Returns
    -------
    Tuple[pd.DataFrame, float, Dict[str, Any]]
        (gaitmap_dataframe, final_sampling_rate_hz, metadata)
    """
    is_valid, missing = validate_sensorlog_columns(df)
    if not is_valid:
        raise ValueError(f"Invalid SensorLog format. Missing required column(s): {', '.join(missing)}")

    # Clean rows with NaNs in required columns
    clean_df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    if len(clean_df) == 0:
        raise ValueError("SensorLog CSV contains no valid non-empty data rows.")

    # Apply windowing if requested
    t_acc = clean_df["accelerometerTimestamp_sinceReboot"].to_numpy(dtype=float)
    t_start = t_acc[0]
    if window_start_min is not None and window_length_min is not None:
        win_start_s = t_start + (window_start_min * 60.0)
        win_end_s = win_start_s + (window_length_min * 60.0)
        mask = (t_acc >= win_start_s) & (t_acc <= win_end_s)
        clean_df = clean_df[mask].copy()
        if len(clean_df) == 0:
            raise ValueError(f"No data in selected time window [{window_start_min} min, {window_start_min + window_length_min} min].")
        t_acc = clean_df["accelerometerTimestamp_sinceReboot"].to_numpy(dtype=float)

    # Unit conversion: G -> m/s^2 (multiply by 9.81)
    acc_x = clean_df["accelerometerAccelerationX"].to_numpy(dtype=float) * G_TO_MPS2
    acc_y = clean_df["accelerometerAccelerationY"].to_numpy(dtype=float) * G_TO_MPS2
    acc_z = clean_df["accelerometerAccelerationZ"].to_numpy(dtype=float) * G_TO_MPS2

    # Gyroscope is already rad/s
    gyr_x = clean_df["gyroRotationX"].to_numpy(dtype=float)
    gyr_y = clean_df["gyroRotationY"].to_numpy(dtype=float)
    gyr_z = clean_df["gyroRotationZ"].to_numpy(dtype=float)

    # Axis remapping
    acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z = apply_axis_mapping(
        acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mapping_name=axis_mapping
    )

    # Detect native rate from accelerometer timestamps
    dt = np.diff(t_acc)
    pos_dt = dt[dt > 0]
    median_dt = float(np.median(pos_dt)) if len(pos_dt) > 0 else (1.0 / target_sampling_rate_hz)
    native_rate = float(1.0 / median_dt) if median_dt > 0 else target_sampling_rate_hz

    # Build flat dictionary
    raw_data = {
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyr_x": gyr_x,
        "gyr_y": gyr_y,
        "gyr_z": gyr_z,
    }
    raw_df = pd.DataFrame(raw_data)

    # Resampling rule (normative):
    # If native rate is already uniform 100 Hz (within 0.5 Hz tolerance), do not resample at all.
    if abs(native_rate - target_sampling_rate_hz) <= 0.5:
        final_df = raw_df
        final_rate = target_sampling_rate_hz
        resampling_used = "None (native uniform ~100 Hz)"
    else:
        # Use polyphase resampling
        final_df, final_rate = resample_imu_dataframe(
            raw_df, orig_rate=native_rate, target_rate=target_sampling_rate_hz
        )
        resampling_used = f"Polyphase resampling ({native_rate:.1f} Hz -> {final_rate:.1f} Hz)"

    # Format as gaitmap MultiSensorData MultiIndex DataFrame
    multi_cols = pd.MultiIndex.from_tuples(
        [(sensor_name, c) for c in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]],
        names=["sensor", "axis"]
    )
    final_df.columns = multi_cols

    metadata = {
        "sensor_name": sensor_name,
        "axis_mapping": axis_mapping,
        "orientation_validated": AXIS_MAPPINGS.get(axis_mapping, {}).get("validated", False),
        "native_sampling_rate_hz": native_rate,
        "final_sampling_rate_hz": final_rate,
        "resampling_used": resampling_used,
        "row_count": len(final_df),
        "duration_s": float(len(final_df) / final_rate) if final_rate > 0 else 0.0,
    }

    return final_df, final_rate, metadata
