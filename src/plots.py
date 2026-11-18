"""Visualization components for IMU signals and stride analysis."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_velocity_histogram(
    velocities: List[float],
    sv95c: float,
    title: str = "Stride Velocity Distribution and SV95C"
) -> go.Figure:
    """Create an interactive histogram of stride velocities with a vertical SV95C line.

    Parameters
    ----------
    velocities : List[float]
        Array or list of valid stride velocities in m/s.
    sv95c : float
        Calculated 95th centile stride velocity in m/s.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Plotly Figure object.
    """
    fig = go.Figure()

    if len(velocities) == 0:
        fig.add_annotation(
            text="No parameterized strides to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(title=title)
        return fig

    # Stride velocity histogram
    fig.add_trace(
        go.Histogram(
            x=velocities,
            nbinsx=max(10, min(30, len(velocities))),
            name="Stride Velocity",
            marker_color="#1f77b4",
            opacity=0.75,
            hovertemplate="Velocity: %{x:.3f} m/s<br>Count: %{y}<extra></extra>"
        )
    )

    # Vertical line at SV95C
    fig.add_vline(
        x=sv95c,
        line_width=3,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"<b>SV95C = {sv95c:.3f} m/s</b>",
        annotation_position="top right",
        annotation_font=dict(size=14, color="#d62728"),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Stride Velocity (m/s)",
        yaxis_title="Count of Strides",
        template="plotly_white",
        bargap=0.08,
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_raw_imu_with_strides(
    imu_data: pd.DataFrame,
    stride_df: pd.DataFrame,
    sampling_rate_hz: float,
    sensor_name: Optional[str] = None,
    max_duration_s: float = 30.0,
    title: str = "Raw IMU Signals with Detected Stride Boundaries"
) -> go.Figure:
    """Plot gyroscope and accelerometer signals with shaded stride intervals.

    Parameters
    ----------
    imu_data : pd.DataFrame
        DataFrame with IMU signals (MultiIndex or single-sensor).
    stride_df : pd.DataFrame
        Per-stride DataFrame containing start_time_s and end_time_s.
    sampling_rate_hz : float
        Sampling rate in Hz.
    sensor_name : str, optional
        Name of the sensor to plot if multi-sensor dataset.
    max_duration_s : float, default 30.0
        Maximum duration in seconds to plot initially to prevent UI lag.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with signal traces and shaded stride spans.
    """
    fig = go.Figure()

    # Determine sensor columns
    if isinstance(imu_data.columns, pd.MultiIndex):
        sensors = list(imu_data.columns.levels[0])
        chosen_sensor = sensor_name if sensor_name in sensors else sensors[0]
        sensor_df = imu_data[chosen_sensor]
    else:
        chosen_sensor = sensor_name or "Sensor"
        sensor_df = imu_data

    n_samples = len(sensor_df)
    total_time_s = n_samples / sampling_rate_hz
    display_samples = min(n_samples, int(max_duration_s * sampling_rate_hz))
    sub_df = sensor_df.iloc[:display_samples]
    time_s = np.arange(display_samples) / sampling_rate_hz

    # Plot primary gyroscope axis (gyr_z or gyr_ml/gyr_y)
    for gyr_col, color in [("gyr_z", "#2ca02c"), ("gyr_y", "#1f77b4"), ("gyr_x", "#ff7f0e")]:
        if gyr_col in sub_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=time_s,
                    y=sub_df[gyr_col],
                    name=f"Gyro {gyr_col} (rad/s)",
                    line=dict(width=1.5, color=color),
                    opacity=0.85,
                    hovertemplate="Time: %{x:.2f} s<br>Rate: %{y:.2f} rad/s<extra></extra>"
                )
            )

    # Add shaded rectangular spans for detected strides within the displayed window
    filtered_strides = stride_df[
        (stride_df["sensor"] == chosen_sensor) & (stride_df["start_time_s"] < max_duration_s)
    ] if "sensor" in stride_df.columns else stride_df[stride_df["start_time_s"] < max_duration_s]

    for idx, row in filtered_strides.iterrows():
        s_start = row["start_time_s"]
        s_end = min(row["end_time_s"], max_duration_s)
        vel_text = f"{row['gait_velocity_mps']:.2f} m/s" if pd.notna(row.get("gait_velocity_mps")) else "N/A"

        # Add vertical span for stride
        fig.add_vrect(
            x0=s_start,
            x1=s_end,
            fillcolor="#aec7e8",
            opacity=0.35,
            layer="below",
            line_width=1,
            line_color="#1f77b4",
            annotation_text=f"S{row.get('s_id', idx)} ({vel_text})",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#1f77b4")
        )

    subtitle = f"Showing first {min(max_duration_s, total_time_s):.1f}s of {total_time_s:.1f}s total (Sensor: {chosen_sensor})"
    fig.update_layout(
        title=dict(text=f"{title}<br><sup>{subtitle}</sup>", font=dict(size=16)),
        xaxis_title="Time (seconds)",
        yaxis_title="Angular Velocity (rad/s)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
