"""
Anomaly detection model wrapper.
Wraps the statistical detection logic used by the agent tools.

For the MVP the model is purely statistical (Z-score + % deviation),
which is reliable and interpretable without requiring a trained ML file.
A scikit-learn IsolationForest / Prophet model can be plugged in here
for Day-3 / future enhancements.
"""

import numpy as np
import pandas as pd
from config import settings


def z_score_detect(series: pd.Series, current: float, z_thresh_high: float = None,
                   z_thresh_med: float = None, dev_thresh_high: float = None,
                   dev_thresh_med: float = None) -> dict:
    """
    Z-score anomaly detection.

    Parameters
    ----------
    series       : Historical baseline consumption values (pd.Series of floats)
    current      : Current reading to evaluate
    z_thresh_*   : Override thresholds from settings if provided

    Returns
    -------
    dict with keys: status, z_score, deviation_pct, baseline_mean, baseline_std, confidence
    """
    z_high = z_thresh_high or settings.Z_HIGH
    z_med  = z_thresh_med  or settings.Z_MED
    d_high = dev_thresh_high or settings.DEV_HIGH
    d_med  = dev_thresh_med  or settings.DEV_MED

    mean = float(series.mean())
    std  = float(series.std(ddof=1)) or 1.0
    dev  = (current - mean) / mean * 100
    z    = (current - mean) / std

    if z >= z_high and dev >= d_high:
        status = "HIGH"
    elif z >= z_med and dev >= d_med:
        status = "MEDIUM"
    else:
        status = "NORMAL"

    conf = round(min(0.99, 0.5 + 0.5 * (1 - np.exp(-max(z, 0) / 4))), 2) \
           if status != "NORMAL" else 0.90

    return {
        "status": status,
        "z_score": round(float(z), 2),
        "deviation_pct": round(float(dev), 1),
        "baseline_mean": round(mean, 1),
        "baseline_std": round(std, 1),
        "confidence": conf,
    }


def rolling_baseline(series: pd.Series, window: int = 7) -> pd.Series:
    """
    Returns a rolling-window baseline (rolling mean) for trend comparison.
    Useful for visualization and secondary anomaly context.
    """
    return series.rolling(window=window, min_periods=3).mean()
