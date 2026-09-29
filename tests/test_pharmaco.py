"""Unit tests for parkinson.pharmaco.DrugTimingFeatures.

Uses small hand-crafted DataFrames to verify:
- fit works with y=None.
- The curve is globally decreasing (first bin > plateau).
- transform preserves row count and index.
- A row without ``on`` gets on_corrige=NaN and off_estime=off.
- A missing delay gives ratio_attendu equal to plateau_.
- A patient with a single off_estime gets off_estime_tendance=NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from parkinson.pharmaco import DrugTimingFeatures


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_training_df(n_per_bin: int = 50) -> pd.DataFrame:
    """Synthetic training frame with enough rows per bin to learn a curve.

    Ratio on/off decreases from ≈0.95 (delay≈0.1 h) to ≈0.55 (delay≈2 h).
    """
    rng = np.random.default_rng(0)
    rows = []
    # bins: (0,.25), (.25,.5), (.5,.75), (.75,1), (1,1.25), (1.25,1.5), (1.5,2.5)
    bin_centers = [0.12, 0.37, 0.62, 0.87, 1.12, 1.37, 2.0]
    true_ratios = [0.95, 0.88, 0.80, 0.73, 0.65, 0.59, 0.55]
    pid = 0
    for delay, ratio in zip(bin_centers, true_ratios):
        for _ in range(n_per_bin):
            off = rng.uniform(20, 60)
            on = off * ratio + rng.normal(0, 0.5)
            rows.append({
                "patient_id": pid % 20,
                "age": 60 + rng.normal(0, 5),
                "time_since_intake_on": delay + rng.uniform(-0.05, 0.05),
                "off": off,
                "on": on,
                "ledd": 500.0,
            })
            pid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_transformer(training_df):
    t = DrugTimingFeatures(min_count=5)
    t.fit(training_df)
    return t


@pytest.fixture
def training_df():
    return _make_training_df()


@pytest.fixture
def simple_transform_df():
    """Small DataFrame with a mix of missing on/off/delay values."""
    return pd.DataFrame(
        {
            "patient_id":            [1,    1,    1,    2,    2,    2],
            "age":                   [50.0, 51.0, 52.0, 40.0, 41.0, 42.0],
            "time_since_intake_on":  [0.1,  np.nan, 1.8,  0.3,  0.7,  2.1],
            # row idx=1: no on → on_corrige should be NaN
            "on":                    [30.0, np.nan, 25.0, 20.0, 18.0, 16.0],
            # row idx=0: no off → off_estime should fall back to on_corrige
            "off":                   [np.nan, 35.0, np.nan, 22.0, np.nan, 17.0],
            "ledd":                  [500., 500., np.nan, 400., 400., np.nan],
        },
        index=[10, 11, 12, 20, 21, 22],
    )


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_with_y_none(self, training_df):
        t = DrugTimingFeatures(min_count=5)
        result = t.fit(training_df, y=None)
        assert result is t  # returns self

    def test_curve_attribute_set(self, fitted_transformer):
        assert hasattr(fitted_transformer, "curve_")
        assert hasattr(fitted_transformer, "plateau_")

    def test_curve_is_series(self, fitted_transformer):
        assert isinstance(fitted_transformer.curve_, pd.Series)

    def test_plateau_is_scalar(self, fitted_transformer):
        assert np.isfinite(fitted_transformer.plateau_)

    def test_curve_globally_decreasing(self, fitted_transformer):
        """First bin ratio must be greater than the plateau (curve is falling)."""
        first_bin_ratio = fitted_transformer.curve_.iloc[0]
        assert first_bin_ratio > fitted_transformer.plateau_, (
            f"First bin ratio {first_bin_ratio:.3f} should be > plateau "
            f"{fitted_transformer.plateau_:.3f}"
        )

    def test_plateau_value_reasonable(self, fitted_transformer):
        """Plateau should be between 0.4 and 0.7 based on the EDA."""
        assert 0.4 < fitted_transformer.plateau_ < 0.7


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

class TestTransform:
    def test_row_count_preserved(self, fitted_transformer, simple_transform_df):
        result = fitted_transformer.transform(simple_transform_df)
        assert len(result) == len(simple_transform_df)

    def test_index_preserved(self, fitted_transformer, simple_transform_df):
        result = fitted_transformer.transform(simple_transform_df)
        pd.testing.assert_index_equal(result.index, simple_transform_df.index)

    def test_input_not_mutated(self, fitted_transformer, simple_transform_df):
        cols_before = list(simple_transform_df.columns)
        fitted_transformer.transform(simple_transform_df)
        assert list(simple_transform_df.columns) == cols_before

    def test_missing_on_gives_nan_on_corrige(self, fitted_transformer, simple_transform_df):
        """Row where on=NaN must yield on_corrige=NaN."""
        result = fitted_transformer.transform(simple_transform_df)
        assert np.isnan(result.loc[11, "on_corrige"]), (
            "on_corrige should be NaN when on is missing"
        )

    def test_missing_on_off_estime_equals_off(self, fitted_transformer, simple_transform_df):
        """Row where on=NaN but off is present → off_estime = off."""
        result = fitted_transformer.transform(simple_transform_df)
        assert result.loc[11, "off_estime"] == pytest.approx(
            simple_transform_df.loc[11, "off"]
        )

    def test_missing_off_estime_uses_on_corrige(self, fitted_transformer, simple_transform_df):
        """Row where off=NaN but on is present → off_estime = on_corrige."""
        result = fitted_transformer.transform(simple_transform_df)
        # idx=10: off is NaN, on is present → off_estime should equal on_corrige
        assert result.loc[10, "off_estime"] == pytest.approx(
            result.loc[10, "on_corrige"]
        )

    def test_missing_delay_gives_plateau(self, fitted_transformer, simple_transform_df):
        """Row with time_since_intake_on=NaN must get ratio_attendu = plateau_."""
        result = fitted_transformer.transform(simple_transform_df)
        assert result.loc[11, "ratio_attendu"] == pytest.approx(
            fitted_transformer.plateau_
        )

    def test_indicator_flags_binary(self, fitted_transformer, simple_transform_df):
        result = fitted_transformer.transform(simple_transform_df)
        for col in ("traite", "on_manquant", "off_manquant"):
            assert set(result[col].unique()).issubset({0, 1}), (
                f"{col} must be binary (0/1)"
            )

    def test_off_estime_moy_constant_per_patient(self, fitted_transformer, simple_transform_df):
        """off_estime_moy_patient must be the same on every visit of a patient."""
        result = fitted_transformer.transform(simple_transform_df)
        for pid, grp in result.groupby("patient_id"):
            unique = grp["off_estime_moy_patient"].dropna().unique()
            assert len(unique) <= 1, (
                f"patient {pid}: off_estime_moy_patient varies: {unique}"
            )

    def test_off_estime_tendance_nan_single_point(self, fitted_transformer):
        """Patient with only one finite off_estime gets off_estime_tendance=NaN."""
        df = pd.DataFrame({
            "patient_id":           [1,      1,       2],
            "age":                  [50.0,   51.0,    60.0],
            "time_since_intake_on": [0.1,    np.nan,  2.0],
            "on":                   [30.0,   np.nan,  20.0],
            "off":                  [np.nan, 35.0,    np.nan],  # patient 2: off=NaN, on present
            "ledd":                 [500.,   500.,    400.],
        })
        # Patient 2 has on_corrige at one visit only (off_estime from on_corrige) and
        # both off_estime values are non-NaN for patient 1, but we craft patient 2
        # to have a single visit → can't fit a line.
        result = fitted_transformer.transform(df)
        p2 = result[result["patient_id"] == 2]
        assert p2["off_estime_tendance"].isna().all(), (
            "Single-visit patient should get NaN off_estime_tendance"
        )

    def test_expected_new_columns(self, fitted_transformer, simple_transform_df):
        result = fitted_transformer.transform(simple_transform_df)
        expected_cols = {
            "ratio_attendu", "on_corrige", "off_estime", "traite",
            "on_manquant", "off_manquant",
            "off_estime_moy_patient", "off_estime_tendance", "part_traite_patient",
        }
        missing = expected_cols - set(result.columns)
        assert not missing, f"Missing columns after transform: {missing}"
