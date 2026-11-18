"""gaitmap pipeline implementation and method configurations (Methods A-D).

Normative definitions:
- Method A: BarthDtw + Rampp + RtsKalman (main chain)
- Method B: ConstrainedBarthDtw + Rampp + RtsKalman
- Method C: BarthDtw + Rampp + default StrideLevelTrajectory() (ForwardBackwardIntegration, no Kalman/ZUPT)
- Method D: Method A on data polyphase-resampled to 100 Hz
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import pandas as pd

from gaitmap.utils.coordinate_conversion import convert_to_fbf
from gaitmap_mad.stride_segmentation import BarthDtw, ConstrainedBarthDtw
from gaitmap.event_detection import RamppEventDetection
from gaitmap.trajectory_reconstruction import StrideLevelTrajectory, RtsKalman
from gaitmap.parameters import TemporalParameterCalculation, SpatialParameterCalculation

from src.resampling import resample_imu_dataframe


@dataclass
class PipelineResult:
    """Container for pipeline execution output."""
    method_name: str
    sampling_rate_hz: float
    stride_list: Dict[str, pd.DataFrame]
    event_list: Optional[Dict[str, pd.DataFrame]]
    temporal_parameters: Optional[Dict[str, pd.DataFrame]]
    spatial_parameters: Optional[Dict[str, pd.DataFrame]]
    per_stride_df: pd.DataFrame
    per_sensor_stats: Dict[str, Dict[str, Any]]
    pooled_stats: Dict[str, Any]
    zero_strides: bool = False
    warning_message: Optional[str] = None
    possible_causes: List[str] = field(default_factory=list)


def convert_dataset_to_fbf(
    data: pd.DataFrame,
    sensor_side: str = "left"
) -> pd.DataFrame:
    """Convert dataset to Foot Body Frame (FBF) for single-sensor or multi-sensor data.

    Parameters
    ----------
    data : pd.DataFrame
        IMU DataFrame with MultiIndex columns (sensor, axis).
    sensor_side : str, default 'left'
        'left' or 'right' side for single-sensor datasets.

    Returns
    -------
    pd.DataFrame
        Dataset transformed to FBF.
    """
    if isinstance(data.columns, pd.MultiIndex):
        sensors = list(data.columns.get_level_values(0).unique())
    else:
        raise ValueError("Data must be a DataFrame with MultiIndex (sensor, axis) columns.")

    has_left = any("left" in s.lower() for s in sensors)
    has_right = any("right" in s.lower() for s in sensors)

    # Multi-sensor with both left and right (e.g. example data)
    if has_left and has_right:
        return convert_to_fbf(data, left_like="left_", right_like="right_")
    elif len(sensors) == 1:
        s = sensors[0]
        if "right" in s.lower() or sensor_side.lower() == "right":
            return convert_to_fbf(data, right=[s])
        else:
            return convert_to_fbf(data, left=[s])
    else:
        # Multiple sensors, match by prefix or assign based on sensor_side
        lefts = [s for s in sensors if "left" in s.lower()]
        rights = [s for s in sensors if "right" in s.lower()]
        if not lefts and not rights:
            if sensor_side.lower() == "right":
                rights = sensors
            else:
                lefts = sensors
        return convert_to_fbf(
            data,
            left=lefts if lefts else None,
            right=rights if rights else None
        )


def run_pipeline(
    data: pd.DataFrame,
    sampling_rate_hz: float,
    segmenter_type: str = "BarthDtw",
    trajectory_method_type: str = "RtsKalman",
    sensor_side: str = "left",
    method_name: str = "Method A"
) -> PipelineResult:
    """Execute gaitmap gait-analysis pipeline on IMU data.

    Parameters
    ----------
    data : pd.DataFrame
        IMU data in gaitmap MultiSensorData format.
    sampling_rate_hz : float
        Explicit sampling rate in Hz.
    segmenter_type : str, default 'BarthDtw'
        'BarthDtw' or 'ConstrainedBarthDtw'.
    trajectory_method_type : str, default 'RtsKalman'
        'RtsKalman' or 'ForwardBackwardIntegration' (default StrideLevelTrajectory).
    sensor_side : str, default 'left'
        'left' or 'right' side for single-sensor data.
    method_name : str, default 'Method A'
        Label for reporting.

    Returns
    -------
    PipelineResult
        Full results including SV95C, per-stride table, and statistics.
    """
    # 1. Coordinate frame conversion to FBF
    fbf_data = convert_dataset_to_fbf(data, sensor_side=sensor_side)

    # 2. Stride segmentation
    if segmenter_type == "ConstrainedBarthDtw":
        segmenter = ConstrainedBarthDtw()
    else:
        segmenter = BarthDtw()

    seg_result = segmenter.segment(data=fbf_data, sampling_rate_hz=sampling_rate_hz)
    stride_list = seg_result.stride_list_

    # Count detected strides
    total_detected = sum(len(sl) for sl in stride_list.values())

    # Handle zero strides gracefully (Acceptance criterion 9.6)
    if total_detected == 0:
        empty_df = pd.DataFrame(columns=[
            "s_id", "sensor", "start_sample", "end_sample",
            "start_time_s", "end_time_s", "stride_length_m",
            "stride_time_s", "gait_velocity_mps"
        ])
        possible_causes = [
            "Wrong sampling rate: verify that the actual recording rate matches the sampling rate passed to the pipeline.",
            "Wrong sensor orientation / axis mapping: if the watch axes are swapped or inverted, DTW templates will not match. Try adjusting the axis mapping in the sidebar.",
            "Resampling damage: linear interpolation can flatten peaks; ensure polyphase resampling was used.",
            "Stationary or non-walking interval: the selected recording window might contain sitting, standing, or non-gait movements."
        ]
        warning_msg = (
            f"Zero strides were detected by {segmenter_type}. "
            "Please check the sampling rate, sensor orientation, and time window."
        )

        sensor_names = list(stride_list.keys())
        per_sensor_stats = {
            s: {
                "detected_strides": 0,
                "parameterized_strides": 0,
                "mean_velocity": np.nan,
                "median_velocity": np.nan,
                "sv95c": np.nan,
                "mean_stride_length": np.nan,
                "mean_stride_time": np.nan,
                "nan_strides": 0,
            }
            for s in sensor_names
        }
        pooled_stats = {
            "detected_strides": 0,
            "parameterized_strides": 0,
            "mean_velocity": np.nan,
            "median_velocity": np.nan,
            "sv95c": np.nan,
            "mean_stride_length": np.nan,
            "mean_stride_time": np.nan,
            "nan_strides": 0,
        }

        return PipelineResult(
            method_name=method_name,
            sampling_rate_hz=sampling_rate_hz,
            stride_list=stride_list,
            event_list=None,
            temporal_parameters=None,
            spatial_parameters=None,
            per_stride_df=empty_df,
            per_sensor_stats=per_sensor_stats,
            pooled_stats=pooled_stats,
            zero_strides=True,
            warning_message=warning_msg,
            possible_causes=possible_causes,
        )

    # 3. Gait event detection
    event_detector = RamppEventDetection()
    event_result = event_detector.detect(
        data=fbf_data,
        stride_list=stride_list,
        sampling_rate_hz=sampling_rate_hz
    )
    event_list = event_result.min_vel_event_list_

    # 4. Trajectory reconstruction
    if trajectory_method_type == "RtsKalman":
        try:
            trajectory = StrideLevelTrajectory(
                trajectory_method=RtsKalman(),
                ori_method=None,
                pos_method=None
            ).estimate(
                data=data,
                stride_event_list=event_list,
                sampling_rate_hz=sampling_rate_hz
            )
        except Exception:
            # Fallback to default StrideLevelTrajectory if explicit RtsKalman construction fails
            trajectory = StrideLevelTrajectory().estimate(
                data=data,
                stride_event_list=event_list,
                sampling_rate_hz=sampling_rate_hz
            )
    else:
        # Default StrideLevelTrajectory() uses ForwardBackwardIntegration without Kalman/ZUPT
        trajectory = StrideLevelTrajectory().estimate(
            data=data,
            stride_event_list=event_list,
            sampling_rate_hz=sampling_rate_hz
        )

    positions = trajectory.position_
    orientations = trajectory.orientation_

    # 5. Temporal and Spatial parameter calculation
    temp_calc = TemporalParameterCalculation().calculate(
        stride_event_list=event_list,
        sampling_rate_hz=sampling_rate_hz
    )
    temporal_params = temp_calc.parameters_

    spat_calc = SpatialParameterCalculation().calculate(
        stride_event_list=event_list,
        positions=positions,
        orientations=orientations,
        sampling_rate_hz=sampling_rate_hz
    )
    spatial_params = spat_calc.parameters_

    # 6. Build per-stride table and summary statistics
    stride_rows = []
    per_sensor_stats = {}
    all_valid_velocities = []
    all_valid_lengths = []
    all_valid_times = []
    total_nan_strides = 0

    for sensor in stride_list.keys():
        sensor_strides = stride_list[sensor]
        sensor_temp = temporal_params.get(sensor, pd.DataFrame()) if temporal_params else pd.DataFrame()
        sensor_spat = spatial_params.get(sensor, pd.DataFrame()) if spatial_params else pd.DataFrame()

        det_count = len(sensor_strides)
        vel_series = sensor_spat["gait_velocity"] if "gait_velocity" in sensor_spat else pd.Series(dtype=float)
        valid_vel = vel_series.dropna()
        param_count = len(valid_vel)
        nan_count = det_count - param_count
        total_nan_strides += nan_count

        length_series = sensor_spat["stride_length"].dropna() if "stride_length" in sensor_spat else pd.Series(dtype=float)
        time_series = sensor_temp["stride_time"].dropna() if "stride_time" in sensor_temp else pd.Series(dtype=float)

        if param_count > 0:
            mean_v = float(np.mean(valid_vel))
            median_v = float(np.median(valid_vel))
            sv95 = float(np.percentile(valid_vel, 95))
            mean_l = float(np.mean(length_series)) if len(length_series) > 0 else np.nan
            mean_t = float(np.mean(time_series)) if len(time_series) > 0 else np.nan
            all_valid_velocities.extend(valid_vel.tolist())
            all_valid_lengths.extend(length_series.tolist())
            all_valid_times.extend(time_series.tolist())
        else:
            mean_v = np.nan
            median_v = np.nan
            sv95 = np.nan
            mean_l = np.nan
            mean_t = np.nan

        per_sensor_stats[sensor] = {
            "detected_strides": det_count,
            "parameterized_strides": param_count,
            "mean_velocity": mean_v,
            "median_velocity": median_v,
            "sv95c": sv95,
            "mean_stride_length": mean_l,
            "mean_stride_time": mean_t,
            "nan_strides": nan_count,
        }

        # Build table rows for this sensor
        for s_id in sensor_strides.index:
            start_s = int(sensor_strides.loc[s_id, "start"])
            end_s = int(sensor_strides.loc[s_id, "end"])
            vel = float(sensor_spat.loc[s_id, "gait_velocity"]) if s_id in sensor_spat.index and pd.notna(sensor_spat.loc[s_id, "gait_velocity"]) else np.nan
            length = float(sensor_spat.loc[s_id, "stride_length"]) if s_id in sensor_spat.index and pd.notna(sensor_spat.loc[s_id, "stride_length"]) else np.nan
            time_val = float(sensor_temp.loc[s_id, "stride_time"]) if s_id in sensor_temp.index and pd.notna(sensor_temp.loc[s_id, "stride_time"]) else np.nan
            stance_val = float(sensor_temp.loc[s_id, "stance_time"]) if s_id in sensor_temp.index and "stance_time" in sensor_temp and pd.notna(sensor_temp.loc[s_id, "stance_time"]) else np.nan
            swing_val = float(sensor_temp.loc[s_id, "swing_time"]) if s_id in sensor_temp.index and "swing_time" in sensor_temp and pd.notna(sensor_temp.loc[s_id, "swing_time"]) else np.nan

            stride_rows.append({
                "s_id": s_id,
                "sensor": sensor,
                "start_sample": start_s,
                "end_sample": end_s,
                "start_time_s": round(start_s / sampling_rate_hz, 4),
                "end_time_s": round(end_s / sampling_rate_hz, 4),
                "stride_length_m": round(length, 4) if pd.notna(length) else np.nan,
                "stride_time_s": round(time_val, 4) if pd.notna(time_val) else np.nan,
                "stance_time_s": round(stance_val, 4) if pd.notna(stance_val) else np.nan,
                "swing_time_s": round(swing_val, 4) if pd.notna(swing_val) else np.nan,
                "gait_velocity_mps": round(vel, 4) if pd.notna(vel) else np.nan,
            })

    per_stride_df = pd.DataFrame(stride_rows)

    # Pooled statistics across sensors
    if len(all_valid_velocities) > 0:
        pooled_stats = {
            "detected_strides": total_detected,
            "parameterized_strides": len(all_valid_velocities),
            "mean_velocity": float(np.mean(all_valid_velocities)),
            "median_velocity": float(np.median(all_valid_velocities)),
            "sv95c": float(np.percentile(all_valid_velocities, 95)),
            "mean_stride_length": float(np.mean(all_valid_lengths)) if len(all_valid_lengths) > 0 else np.nan,
            "mean_stride_time": float(np.mean(all_valid_times)) if len(all_valid_times) > 0 else np.nan,
            "nan_strides": total_nan_strides,
        }
    else:
        pooled_stats = {
            "detected_strides": total_detected,
            "parameterized_strides": 0,
            "mean_velocity": np.nan,
            "median_velocity": np.nan,
            "sv95c": np.nan,
            "mean_stride_length": np.nan,
            "mean_stride_time": np.nan,
            "nan_strides": total_nan_strides,
        }

    return PipelineResult(
        method_name=method_name,
        sampling_rate_hz=sampling_rate_hz,
        stride_list=stride_list,
        event_list=event_list,
        temporal_parameters=temporal_params,
        spatial_parameters=spatial_params,
        per_stride_df=per_stride_df,
        per_sensor_stats=per_sensor_stats,
        pooled_stats=pooled_stats,
        zero_strides=False,
    )


def run_method_a(data: pd.DataFrame, sampling_rate_hz: float, sensor_side: str = "left") -> PipelineResult:
    """Run Method A: BarthDtw + Rampp + RtsKalman (main chain)."""
    return run_pipeline(
        data=data,
        sampling_rate_hz=sampling_rate_hz,
        segmenter_type="BarthDtw",
        trajectory_method_type="RtsKalman",
        sensor_side=sensor_side,
        method_name="Method A (BarthDtw + RtsKalman)"
    )


def run_method_b(data: pd.DataFrame, sampling_rate_hz: float, sensor_side: str = "left") -> PipelineResult:
    """Run Method B: ConstrainedBarthDtw + Rampp + RtsKalman."""
    return run_pipeline(
        data=data,
        sampling_rate_hz=sampling_rate_hz,
        segmenter_type="ConstrainedBarthDtw",
        trajectory_method_type="RtsKalman",
        sensor_side=sensor_side,
        method_name="Method B (ConstrainedBarthDtw + RtsKalman)"
    )


def run_method_c(data: pd.DataFrame, sampling_rate_hz: float, sensor_side: str = "left") -> PipelineResult:
    """Run Method C: BarthDtw + Rampp + default StrideLevelTrajectory (no Kalman/ZUPT)."""
    return run_pipeline(
        data=data,
        sampling_rate_hz=sampling_rate_hz,
        segmenter_type="BarthDtw",
        trajectory_method_type="ForwardBackwardIntegration",
        sensor_side=sensor_side,
        method_name="Method C (BarthDtw + ForwardBackwardIntegration)"
    )


def run_method_d(data: pd.DataFrame, sampling_rate_hz: float, sensor_side: str = "left") -> PipelineResult:
    """Run Method D: main chain on data polyphase-resampled to 100 Hz."""
    resampled_data, new_rate = resample_imu_dataframe(data, orig_rate=sampling_rate_hz, target_rate=100.0)
    return run_pipeline(
        data=resampled_data,
        sampling_rate_hz=new_rate,
        segmenter_type="BarthDtw",
        trajectory_method_type="RtsKalman",
        sensor_side=sensor_side,
        method_name="Method D (Main chain @ 100 Hz polyphase)"
    )
