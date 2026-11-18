"""Polyphase resampling utilities for IMU data.

Normative rules:
- Resample ONLY with polyphase resampling (scipy.signal.resample_poly).
- For 204.8 -> 100 Hz: up=125, down=256 (204.8/100 = 256/125, exact).
- Assert the resampled frame has no NaNs.
- NEVER use linear interpolation to resample IMU data.
"""

from fractions import Fraction
from typing import Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def get_polyphase_factors(orig_rate: float, target_rate: float = 100.0) -> Tuple[int, int]:
    """Calculate exact or best integer up and down factors for polyphase resampling.

    Parameters
    ----------
    orig_rate : float
        Original sampling rate in Hz.
    target_rate : float, default 100.0
        Target sampling rate in Hz.

    Returns
    -------
    Tuple[int, int]
        (up, down) integer factors for scipy.signal.resample_poly.
    """
    # Exact known ratio for 204.8 -> 100 Hz
    if np.isclose(orig_rate, 204.8, atol=1e-3) and np.isclose(target_rate, 100.0, atol=1e-3):
        return 125, 256

    ratio = Fraction(target_rate / orig_rate).limit_denominator(2000)
    return ratio.numerator, ratio.denominator


def polyphase_resample_array(
    data: np.ndarray,
    orig_rate: float,
    target_rate: float = 100.0,
    axis: int = 0
) -> np.ndarray:
    """Resample an array of IMU signals using polyphase filtering.

    Parameters
    ----------
    data : np.ndarray
        Array of IMU samples.
    orig_rate : float
        Original sampling rate in Hz.
    target_rate : float, default 100.0
        Target sampling rate in Hz.
    axis : int, default 0
        Axis along which to resample.

    Returns
    -------
    np.ndarray
        Resampled signal array.
    """
    if np.isclose(orig_rate, target_rate, atol=1e-2):
        return data

    up, down = get_polyphase_factors(orig_rate, target_rate)
    resampled = resample_poly(data, up=up, down=down, axis=axis)

    # Normative check: assert no NaNs
    assert not np.isnan(resampled).any(), "Resampled signal contains NaNs"
    return resampled


def resample_imu_dataframe(
    df: pd.DataFrame,
    orig_rate: float,
    target_rate: float = 100.0
) -> Tuple[pd.DataFrame, float]:
    """Resample an IMU DataFrame (single-sensor or multi-sensor) to target rate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with IMU signals. Columns can be flat or MultiIndex.
    orig_rate : float
        Original sampling rate in Hz.
    target_rate : float, default 100.0
        Target sampling rate in Hz.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        (resampled_df, actual_new_sampling_rate)
    """
    if np.isclose(orig_rate, target_rate, atol=1e-2):
        return df.copy(), orig_rate

    up, down = get_polyphase_factors(orig_rate, target_rate)

    resampled_dict = {}
    for col in df.columns:
        col_resampled = resample_poly(df[col].to_numpy(dtype=float), up=up, down=down, axis=0)
        assert not np.isnan(col_resampled).any(), f"Resampled column {col} contains NaNs"
        resampled_dict[col] = col_resampled

    new_df = pd.DataFrame(resampled_dict)
    # Preserve original MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        new_df.columns = df.columns

    new_rate = float(orig_rate * up / down)
    return new_df, new_rate
