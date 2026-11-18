"""Acceptance tests for the SV95C analysis pipeline.

Verifies:
- 9.1 Demo mode Method A reproduces exact verified reference numbers
- 9.2 Method comparison on example data (spread < 0.10 m/s MCID)
- 9.3 Single-sensor path (Left only: 28 detected, 26 parameterized, SV95C 1.373 m/s)
- 9.4 Resampling regression test (100 Hz polyphase, no linear interpolation anywhere)
- 9.5 SensorLog upload validation and error messaging
- 9.6 Zero-stride handling (no crash, clear diagnostic banner)
"""

import os
import re
import numpy as np
import pandas as pd
import pytest
from gaitmap.example_data import get_healthy_example_imu_data

from src.pipeline import (
    run_method_a,
    run_method_b,
    run_method_c,
    run_method_d,
    run_pipeline,
)
from src.analysis import compare_all_methods
from src.resampling import resample_imu_dataframe
from src.sensorlog import (
    REQUIRED_COLUMNS,
    validate_sensorlog_columns,
    compute_validation_summary,
    parse_sensorlog_to_gaitmap,
)


@pytest.fixture(scope="session")
def example_data():
    """Load gaitmap healthy example dataset (204.8 Hz, 2x20m walk)."""
    return get_healthy_example_imu_data()


def test_9_1_method_a_reference_numbers(example_data):
    """Criterion 9.1: Method A reproduces verified reference numbers to 3 decimals."""
    sr = 204.8
    result = run_method_a(example_data, sr)

    left_stats = result.per_sensor_stats["left_sensor"]
    right_stats = result.per_sensor_stats["right_sensor"]
    pooled_stats = result.pooled_stats

    # Left sensor: 28 detected, 26 parameterized, mean 1.252 m/s, SV95C 1.373 m/s
    assert left_stats["detected_strides"] == 28
    assert left_stats["parameterized_strides"] == 26
    assert np.isclose(left_stats["mean_velocity"], 1.252, atol=1e-3)
    assert np.isclose(left_stats["sv95c"], 1.373, atol=1e-3)

    # Right sensor: 28 detected, 26 parameterized, mean 1.257 m/s, SV95C 1.410 m/s
    assert right_stats["detected_strides"] == 28
    assert right_stats["parameterized_strides"] == 26
    assert np.isclose(right_stats["mean_velocity"], 1.257, atol=1e-3)
    assert np.isclose(right_stats["sv95c"], 1.410, atol=1e-3)

    # Pooled: 52 parameterized, mean 1.255 m/s, SV95C 1.393 m/s
    assert pooled_stats["parameterized_strides"] == 52
    assert np.isclose(pooled_stats["mean_velocity"], 1.255, atol=1e-3)
    assert np.isclose(pooled_stats["sv95c"], 1.393, atol=1e-3)


def test_9_2_method_comparison_spread_and_mcid(example_data):
    """Criterion 9.2: B pooled 1.393, C pooled 1.374, D pooled 1.392; spread 0.019 < 0.10 m/s MCID."""
    sr = 204.8
    comp_df, spread_metrics, results = compare_all_methods(example_data, sr)

    res_b = results["B"]
    res_c = results["C"]
    res_d = results["D"]

    assert np.isclose(res_b.pooled_stats["sv95c"], 1.393, atol=1e-3)
    assert np.isclose(res_c.pooled_stats["sv95c"], 1.374, atol=1e-3)
    assert np.isclose(res_d.pooled_stats["sv95c"], 1.392, atol=1e-3)

    assert np.isclose(spread_metrics["spread_mps"], 0.019, atol=1e-3)
    assert spread_metrics["spread_mps"] < 0.10
    assert spread_metrics["is_below_mcid"] is True


def test_9_3_single_sensor_left_path(example_data):
    """Criterion 9.3: Single-sensor left: 28 detected, 26 parameterized, SV95C 1.373 m/s."""
    sr = 204.8
    left_data = example_data[["left_sensor"]]
    result = run_method_a(left_data, sr, sensor_side="left")

    left_stats = result.per_sensor_stats["left_sensor"]
    assert left_stats["detected_strides"] == 28
    assert left_stats["parameterized_strides"] == 26
    assert np.isclose(left_stats["mean_velocity"], 1.252, atol=1e-3)
    assert np.isclose(left_stats["sv95c"], 1.373, atol=1e-3)
    assert np.isclose(result.pooled_stats["sv95c"], 1.373, atol=1e-3)


def test_9_4_resampling_regression_and_no_linear_interp(example_data):
    """Criterion 9.4: Polyphase resample to 100 Hz yields pooled SV95C 1.392 m/s.

    Also verifies no linear interpolation on IMU data exists anywhere in the codebase.
    """
    sr = 204.8
    # 1. Polyphase resample to 100 Hz
    resampled_data, new_sr = resample_imu_dataframe(example_data, orig_rate=sr, target_rate=100.0)
    assert np.isclose(new_sr, 100.0, atol=1e-2)
    assert not resampled_data.isna().any().any()

    # Run Method A
    result = run_method_a(resampled_data, new_sr)
    left_stats = result.per_sensor_stats["left_sensor"]
    right_stats = result.per_sensor_stats["right_sensor"]
    pooled_stats = result.pooled_stats

    assert left_stats["detected_strides"] == 28
    assert left_stats["parameterized_strides"] == 26
    assert right_stats["detected_strides"] == 30
    assert right_stats["parameterized_strides"] == 28
    assert pooled_stats["parameterized_strides"] == 54
    assert np.isclose(pooled_stats["sv95c"], 1.392, atol=1e-3)

    # 2. Grep test: ensure np.interp and .interpolate( do not appear on IMU data in src/
    root_src = os.path.join(os.path.dirname(__file__), "..", "src")
    forbidden_patterns = [r"\bnp\.interp\b", r"\bnumpy\.interp\b", r"\.interpolate\("]
    for root, _, files in os.walk(root_src):
        for f in files:
            if f.endswith(".py"):
                fpath = os.path.join(root, f)
                with open(fpath, "r", encoding="utf-8") as py_file:
                    content = py_file.read()
                    for pattern in forbidden_patterns:
                        matches = re.findall(pattern, content)
                        assert len(matches) == 0, f"Forbidden linear interpolation {pattern} found in {fpath}"


def test_9_5_upload_validation():
    """Criterion 9.5: Missing columns produce clear error; valid format gives summary panel."""
    # 1. Missing columns test
    bad_df = pd.DataFrame({
        "accelerometerTimestamp_sinceReboot": [0.0, 0.01, 0.02],
        "accelerometerAccelerationX": [0.1, 0.1, 0.1],
        # Missing gyro and other columns
    })
    is_valid, missing = validate_sensorlog_columns(bad_df)
    assert not is_valid
    assert "gyroTimestamp_sinceReboot" in missing
    assert "accelerometerAccelerationY" in missing

    with pytest.raises(ValueError) as excinfo:
        compute_validation_summary(bad_df)
    assert "Missing required column(s)" in str(excinfo.value)
    assert "gyroTimestamp_sinceReboot" in str(excinfo.value)

    # 2. Valid format test (native uniform 100 Hz)
    n_rows = 100
    timestamps = 100.0 + np.arange(n_rows) * 0.010
    valid_df = pd.DataFrame({
        "accelerometerTimestamp_sinceReboot": timestamps,
        "accelerometerAccelerationX": np.zeros(n_rows),
        "accelerometerAccelerationY": np.zeros(n_rows),
        "accelerometerAccelerationZ": np.ones(n_rows),
        "gyroTimestamp_sinceReboot": timestamps,
        "gyroRotationX": np.zeros(n_rows),
        "gyroRotationY": np.zeros(n_rows),
        "gyroRotationZ": np.zeros(n_rows),
    })

    summary = compute_validation_summary(valid_df)
    assert summary["row_count"] == 100
    assert np.isclose(summary["detected_rate_hz"], 100.0, atol=0.1)
    assert summary["missing_value_count"] == 0
    assert summary["needs_windowing"] is False

    gait_df, rate, meta = parse_sensorlog_to_gaitmap(valid_df)
    assert len(gait_df) == 100
    assert np.isclose(rate, 100.0, atol=0.1)
    assert "None (native uniform ~100 Hz)" in meta["resampling_used"]
    assert isinstance(gait_df.columns, pd.MultiIndex)
    assert ("left_sensor", "acc_x") in gait_df.columns

    # 3. Test non-100 Hz input triggers polyphase resampling
    non_100_timestamps = 100.0 + np.arange(200) * 0.020  # 50 Hz
    non_100_df = pd.DataFrame({
        "accelerometerTimestamp_sinceReboot": non_100_timestamps,
        "accelerometerAccelerationX": np.zeros(200),
        "accelerometerAccelerationY": np.zeros(200),
        "accelerometerAccelerationZ": np.ones(200),
        "gyroTimestamp_sinceReboot": non_100_timestamps,
        "gyroRotationX": np.zeros(200),
        "gyroRotationY": np.zeros(200),
        "gyroRotationZ": np.zeros(200),
    })
    gait_resampled, rate_resampled, meta_resampled = parse_sensorlog_to_gaitmap(non_100_df)
    assert np.isclose(rate_resampled, 100.0, atol=0.1)
    assert "Polyphase" in meta_resampled["resampling_used"]


def test_9_6_zero_stride_handling():
    """Criterion 9.6: Zero strides detected does not crash, returns structured warning."""
    # Constant zero IMU signal where DTW template will detect 0 strides
    zero_df = pd.DataFrame(
        np.zeros((500, 6)),
        columns=pd.MultiIndex.from_tuples(
            [("ankle_sensor", c) for c in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]],
            names=["sensor", "axis"]
        )
    )

    result = run_pipeline(
        data=zero_df,
        sampling_rate_hz=100.0,
        segmenter_type="BarthDtw",
        trajectory_method_type="RtsKalman",
        sensor_side="left"
    )

    assert result.zero_strides is True
    assert result.pooled_stats["detected_strides"] == 0
    assert result.pooled_stats["parameterized_strides"] == 0
    assert np.isnan(result.pooled_stats["sv95c"])
    assert len(result.per_stride_df) == 0
    assert len(result.possible_causes) > 0
    assert "sampling rate" in result.possible_causes[0].lower()
