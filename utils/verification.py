"""Verification metrics utilities using the scores package.

Provides wrappers around the scores library for computing verification metrics.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Optional, Tuple, Any

try:
    import scores
    from scores.continuous import mae, rmse
    from scores.probability import brier_score, crps_for_ensemble
    SCORES_AVAILABLE = True
except ImportError as e:
    SCORES_AVAILABLE = False


class VerificationMetrics:
    """Calculate verification metrics between forecast and observation data."""

    def __init__(self):
        """Initialize the verification metrics calculator."""
        if not SCORES_AVAILABLE:
            raise ImportError(
                "The 'scores' package is required for verification. "
                "Install it with: pip install scores"
            )

    @staticmethod
    def continuous_metrics(
        forecast: xr.DataArray,
        observation: xr.DataArray,
    ) -> Dict[str, float]:
        """Calculate continuous verification metrics.

        Args:
            forecast: Forecast data array
            observation: Observation data array

        Returns:
            Dict with RMSE, MAE, and Bias values
        """
        if not SCORES_AVAILABLE:
            return {"error": "scores package not available"}

        # Ensure arrays are aligned
        forecast, observation = xr.align(forecast, observation, join="inner")

        # Flatten for metric calculation
        fcst_flat = forecast.values.flatten()
        obs_flat = observation.values.flatten()

        # Remove NaN values
        mask = ~(np.isnan(fcst_flat) | np.isnan(obs_flat))
        fcst_clean = fcst_flat[mask]
        obs_clean = obs_flat[mask]

        if len(fcst_clean) == 0:
            return {"rmse": np.nan, "mae": np.nan, "bias": np.nan}

        return {
            "rmse": float(np.sqrt(np.mean((fcst_clean - obs_clean) ** 2))),
            "mae": float(np.mean(np.abs(fcst_clean - obs_clean))),
            "bias": float(np.mean(fcst_clean - obs_clean)),
        }

    @staticmethod
    def categorical_metrics(
        forecast: xr.DataArray,
        observation: xr.DataArray,
        threshold: float,
    ) -> Dict[str, float]:
        """Calculate categorical verification metrics.

        Args:
            forecast: Forecast data array
            observation: Observation data array
            threshold: Threshold for converting to binary

        Returns:
            Dict with POD, FAR, and CSI values
        """
        # Convert to binary based on threshold
        fcst_binary = (forecast >= threshold).astype(int)
        obs_binary = (observation >= threshold).astype(int)

        # Flatten and clean
        fcst_flat = fcst_binary.values.flatten()
        obs_flat = obs_binary.values.flatten()
        mask = ~(np.isnan(fcst_flat) | np.isnan(obs_flat))
        fcst_clean = fcst_flat[mask].astype(int)
        obs_clean = obs_flat[mask].astype(int)

        if len(fcst_clean) == 0:
            return {"pod": np.nan, "far": np.nan, "csi": np.nan}

        # Calculate contingency table elements
        hits = np.sum((fcst_clean == 1) & (obs_clean == 1))
        misses = np.sum((fcst_clean == 0) & (obs_clean == 1))
        false_alarms = np.sum((fcst_clean == 1) & (obs_clean == 0))

        # Calculate metrics
        pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else np.nan

        return {
            "pod": float(pod),
            "far": float(far),
            "csi": float(csi),
        }

    @staticmethod
    def probabilistic_metrics(
        forecast_prob: xr.DataArray,
        observation: xr.DataArray,
        threshold: float,
    ) -> Dict[str, float]:
        """Calculate probabilistic verification metrics.

        Args:
            forecast_prob: Forecast probability data (0-1 or 0-100%)
            observation: Observation data array
            threshold: Threshold for binary observation

        Returns:
            Dict with Brier score and reliability
        """
        # Convert observations to binary
        obs_binary = (observation >= threshold).astype(float)

        # Ensure probability is 0-1
        if forecast_prob.max() > 1:
            forecast_prob = forecast_prob / 100.0

        # Flatten and clean
        prob_flat = forecast_prob.values.flatten()
        obs_flat = obs_binary.values.flatten()
        mask = ~(np.isnan(prob_flat) | np.isnan(obs_flat))
        prob_clean = prob_flat[mask]
        obs_clean = obs_flat[mask]

        if len(prob_clean) == 0:
            return {"brier_score": np.nan}

        # Brier score
        brier = float(np.mean((prob_clean - obs_clean) ** 2))

        return {
            "brier_score": brier,
        }

    @staticmethod
    def spatial_metrics(
        forecast: xr.DataArray,
        observation: xr.DataArray,
        threshold: float,
        window_sizes: List[int] = [1, 3, 5, 11, 21],
    ) -> Dict[str, Dict[int, float]]:
        """Calculate spatial verification metrics (FSS).

        Args:
            forecast: 2D forecast data array
            observation: 2D observation data array
            threshold: Threshold for binary conversion
            window_sizes: List of neighborhood window sizes

        Returns:
            Dict with FSS values for each window size
        """
        fss_results = {}

        # Ensure 2D arrays
        fcst_2d = forecast.squeeze().values
        obs_2d = observation.squeeze().values

        if fcst_2d.ndim != 2 or obs_2d.ndim != 2:
            return {"fss": {}, "error": "Data must be 2D"}

        for window in window_sizes:
            try:
                fss = VerificationMetrics._calculate_fss(
                    fcst_2d, obs_2d, threshold, window
                )
                fss_results[window] = float(fss)
            except Exception as e:
                fss_results[window] = np.nan

        return {"fss": fss_results}

    @staticmethod
    def _calculate_fss(
        forecast: np.ndarray,
        observation: np.ndarray,
        threshold: float,
        window_size: int,
    ) -> float:
        """Calculate Fractions Skill Score for a single window size.

        Args:
            forecast: 2D forecast array
            observation: 2D observation array
            threshold: Threshold for binary conversion
            window_size: Neighborhood window size

        Returns:
            FSS value
        """
        from scipy.ndimage import uniform_filter

        # Convert to binary
        fcst_binary = (forecast >= threshold).astype(float)
        obs_binary = (observation >= threshold).astype(float)

        # Calculate fractions
        fcst_frac = uniform_filter(fcst_binary, size=window_size, mode="constant")
        obs_frac = uniform_filter(obs_binary, size=window_size, mode="constant")

        # FSS numerator and denominator
        mse = np.nanmean((fcst_frac - obs_frac) ** 2)
        mse_ref = np.nanmean(fcst_frac ** 2) + np.nanmean(obs_frac ** 2)

        if mse_ref == 0:
            return 1.0 if mse == 0 else 0.0

        fss = 1 - (mse / mse_ref)
        return fss

    @staticmethod
    def roc_curve(
        forecast_prob: xr.DataArray,
        observation: xr.DataArray,
        threshold: float,
        n_bins: int = 11,
    ) -> Dict[str, List[float]]:
        """Calculate ROC curve data.

        Args:
            forecast_prob: Forecast probability data
            observation: Observation data
            threshold: Threshold for binary observation
            n_bins: Number of probability bins

        Returns:
            Dict with false_alarm_rate and hit_rate lists
        """
        obs_binary = (observation >= threshold).astype(float)

        # Ensure probability is 0-1
        if forecast_prob.max() > 1:
            forecast_prob = forecast_prob / 100.0

        prob_flat = forecast_prob.values.flatten()
        obs_flat = obs_binary.values.flatten()
        mask = ~(np.isnan(prob_flat) | np.isnan(obs_flat))
        prob_clean = prob_flat[mask]
        obs_clean = obs_flat[mask]

        if len(prob_clean) == 0:
            return {"false_alarm_rate": [], "hit_rate": []}

        far_list = [0.0]
        hit_rate_list = [0.0]

        prob_thresholds = np.linspace(1, 0, n_bins)

        for prob_thresh in prob_thresholds:
            fcst_binary = (prob_clean >= prob_thresh).astype(int)

            hits = np.sum((fcst_binary == 1) & (obs_clean == 1))
            misses = np.sum((fcst_binary == 0) & (obs_clean == 1))
            false_alarms = np.sum((fcst_binary == 1) & (obs_clean == 0))
            correct_negatives = np.sum((fcst_binary == 0) & (obs_clean == 0))

            hr = hits / (hits + misses) if (hits + misses) > 0 else 0
            far = false_alarms / (false_alarms + correct_negatives) if (false_alarms + correct_negatives) > 0 else 0

            hit_rate_list.append(hr)
            far_list.append(far)

        far_list.append(1.0)
        hit_rate_list.append(1.0)

        return {
            "false_alarm_rate": far_list,
            "hit_rate": hit_rate_list,
        }

    @staticmethod
    def summary_table(
        metrics: Dict[str, float],
    ) -> pd.DataFrame:
        """Create a summary table from metrics dictionary.

        Args:
            metrics: Dictionary of metric names and values

        Returns:
            DataFrame with metrics summary
        """
        df = pd.DataFrame([metrics]).T
        df.columns = ["Value"]
        df.index.name = "Metric"
        return df


def calculate_all_metrics(
    forecast: xr.DataArray,
    observation: xr.DataArray,
    threshold: Optional[float] = None,
    forecast_prob: Optional[xr.DataArray] = None,
) -> Dict[str, Any]:
    """Calculate all applicable verification metrics.

    Args:
        forecast: Deterministic forecast data
        observation: Observation data
        threshold: Optional threshold for categorical/probabilistic metrics
        forecast_prob: Optional probability forecast for probabilistic metrics

    Returns:
        Dictionary with all calculated metrics
    """
    verifier = VerificationMetrics()
    results = {}

    # Continuous metrics
    results["continuous"] = verifier.continuous_metrics(forecast, observation)

    # Categorical metrics (if threshold provided)
    if threshold is not None:
        results["categorical"] = verifier.categorical_metrics(
            forecast, observation, threshold
        )

        # Spatial metrics
        results["spatial"] = verifier.spatial_metrics(
            forecast, observation, threshold
        )

        # Probabilistic metrics (if probability forecast provided)
        if forecast_prob is not None:
            results["probabilistic"] = verifier.probabilistic_metrics(
                forecast_prob, observation, threshold
            )
            results["roc"] = verifier.roc_curve(
                forecast_prob, observation, threshold
            )

    return results
