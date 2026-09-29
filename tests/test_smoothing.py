"""Unit tests for parkinson.smoothing.PatientSmoothedRegressor.

- With a dummy estimator that returns a noisy parabola, the smoothed output
  is closer to the clean parabola than the raw output.
- A patient with <= deg+1 distinct ages keeps its raw predictions unchanged.
- Row order is preserved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from parkinson.smoothing import PatientSmoothedRegressor


# ---------------------------------------------------------------------------
# Dummy estimator
# ---------------------------------------------------------------------------

class ConstantPredictor(BaseEstimator, RegressorMixin):
    """Stores a fixed prediction array and replays it on predict."""

    def __init__(self, predictions: np.ndarray):
        self.predictions = predictions

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.predictions.copy()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parabola_df():
    """Two patients: patient 1 has 10 visits (parabola + noise), patient 2 has 2."""
    rng = np.random.default_rng(42)
    ages_p1 = np.linspace(50, 60, 10)
    # True parabola: -(age-55)^2 + 40
    true_p1 = -(ages_p1 - 55) ** 2 + 40
    noisy_p1 = true_p1 + rng.normal(0, 3, size=10)

    ages_p2 = np.array([60.0, 61.0])
    true_p2 = np.array([35.0, 36.0])
    noisy_p2 = true_p2 + rng.normal(0, 1, size=2)

    n = 12
    df = pd.DataFrame({
        "patient_id": [1] * 10 + [2] * 2,
        "age": np.concatenate([ages_p1, ages_p2]),
    })
    return df, np.concatenate([noisy_p1, noisy_p2]), np.concatenate([true_p1, true_p2])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPatientSmoothedRegressor:
    def test_fit_returns_self(self, parabola_df):
        df, noisy, _ = parabola_df
        dummy = ConstantPredictor(noisy)
        reg = PatientSmoothedRegressor(dummy)
        result = reg.fit(df, noisy)
        assert result is reg

    def test_smoothed_closer_to_truth_for_p1(self, parabola_df):
        """Patient 1 (10 visits): smoothed predictions should be closer to true parabola."""
        df, noisy, true = parabola_df
        dummy = ConstantPredictor(noisy)
        reg = PatientSmoothedRegressor(dummy, deg=2, weight=1.0)
        reg.fit(df, noisy)
        smoothed = reg.predict(df)

        p1_mask = df["patient_id"] == 1
        raw_rmse = np.sqrt(np.mean((noisy[p1_mask] - true[p1_mask]) ** 2))
        smooth_rmse = np.sqrt(np.mean((smoothed[p1_mask] - true[p1_mask]) ** 2))
        assert smooth_rmse < raw_rmse, (
            f"Smoothed RMSE ({smooth_rmse:.3f}) should be < raw RMSE ({raw_rmse:.3f})"
        )

    def test_two_visit_patient_unchanged(self, parabola_df):
        """Patient 2 has 2 visits (≤ deg+1=3 distinct ages) → raw predictions kept."""
        df, noisy, _ = parabola_df
        dummy = ConstantPredictor(noisy)
        reg = PatientSmoothedRegressor(dummy, deg=2, weight=1.0)
        reg.fit(df, noisy)
        smoothed = reg.predict(df)

        p2_mask = (df["patient_id"] == 2).values
        np.testing.assert_array_almost_equal(
            smoothed[p2_mask], noisy[p2_mask],
            decimal=10,
            err_msg="Patient with 2 visits should keep raw predictions",
        )

    def test_row_order_preserved(self, parabola_df):
        """Output array must match input row order."""
        df, noisy, _ = parabola_df
        dummy = ConstantPredictor(noisy)
        reg = PatientSmoothedRegressor(dummy, deg=2, weight=1.0)
        reg.fit(df, noisy)
        smoothed = reg.predict(df)
        assert smoothed.shape == noisy.shape

    def test_weight_zero_returns_raw(self, parabola_df):
        """weight=0.0 → output should equal raw predictions exactly."""
        df, noisy, _ = parabola_df
        dummy = ConstantPredictor(noisy)
        reg = PatientSmoothedRegressor(dummy, deg=2, weight=0.0)
        reg.fit(df, noisy)
        smoothed = reg.predict(df)
        np.testing.assert_array_almost_equal(smoothed, noisy, decimal=10)

    def test_partial_weight_between_raw_and_smooth(self, parabola_df):
        """weight=0.5 → output should be between raw and fully smoothed."""
        df, noisy, true = parabola_df
        dummy = ConstantPredictor(noisy)

        reg_full = PatientSmoothedRegressor(dummy, deg=2, weight=1.0)
        reg_full.fit(df, noisy)
        full_smooth = reg_full.predict(df)

        reg_half = PatientSmoothedRegressor(dummy, deg=2, weight=0.5)
        reg_half.fit(df, noisy)
        half_smooth = reg_half.predict(df)

        p1_mask = (df["patient_id"] == 1).values
        expected = 0.5 * full_smooth[p1_mask] + 0.5 * noisy[p1_mask]
        np.testing.assert_array_almost_equal(half_smooth[p1_mask], expected, decimal=10)
