"""Unit tests for parkinson.features.

Uses a small hand-crafted DataFrame (2–3 patients) to verify:
- ``add_patient_features`` preserves row count, index and ``patient_id``.
- ``off_moy_patient`` is the same on every visit of a patient, including
  visits where ``off`` is missing.
- ``patient_trend`` on off=10,20 at ages 50,51 predicts 30 at age 52.
- A patient with only one non-NaN ``off`` gets ``off_tendance_patient = NaN``.
- The functions work on a frame with no ``target`` column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from parkinson.features import add_patient_features, patient_trend


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """Return a minimal feature frame with 3 patients, no target column."""
    return pd.DataFrame(
        {
            "patient_id":        [1,    1,    1,    2,    2,    3],
            "age":               [50.0, 51.0, 52.0, 40.0, 41.0, 60.0],
            "age_at_diagnosis":  [45.0, 45.0, 45.0, 35.0, 35.0, 55.0],
            # Patient 1: off at 50→10, 51→20, 52→NaN (trend should predict 30).
            # Patient 2: two off values (no NaN visits).
            # Patient 3: only one off value → slope undefined → NaN trend.
            "off":               [10.0, 20.0, np.nan, 5.0,  7.0,  3.0],
            "on":                [1.0,  2.0,  3.0,    4.0,  5.0,  6.0],
            "ledd":              [100., 110., 120.,   90.,  95.,  80.],
        },
        index=[10, 20, 30, 40, 50, 60],   # non-default index to test alignment
    )


# ---------------------------------------------------------------------------
# add_patient_features
# ---------------------------------------------------------------------------

class TestAddPatientFeatures:
    def test_row_count_preserved(self, simple_df):
        result = add_patient_features(simple_df)
        assert len(result) == len(simple_df)

    def test_index_preserved(self, simple_df):
        result = add_patient_features(simple_df)
        pd.testing.assert_index_equal(result.index, simple_df.index)

    def test_patient_id_preserved(self, simple_df):
        result = add_patient_features(simple_df)
        assert "patient_id" in result.columns
        pd.testing.assert_series_equal(
            result["patient_id"], simple_df["patient_id"], check_names=True
        )

    def test_input_not_mutated(self, simple_df):
        cols_before = list(simple_df.columns)
        add_patient_features(simple_df)
        assert list(simple_df.columns) == cols_before

    def test_no_target_column_required(self, simple_df):
        """Function must work on a frame that has no target column."""
        assert "target" not in simple_df.columns   # sanity check
        result = add_patient_features(simple_df)
        assert result is not None

    def test_off_moy_patient_constant_per_patient(self, simple_df):
        """off_moy_patient must be the same on every visit of a patient."""
        result = add_patient_features(simple_df)
        for pid, grp in result.groupby("patient_id"):
            unique_vals = grp["off_moy_patient"].dropna().unique()
            assert len(unique_vals) <= 1, (
                f"patient {pid}: off_moy_patient varies within patient: {unique_vals}"
            )

    def test_off_moy_patient_filled_on_nan_off_visit(self, simple_df):
        """Visits where off is NaN must still have a valid off_moy_patient."""
        result = add_patient_features(simple_df)
        # Patient 1, visit at index 30 has off=NaN but belongs to a patient
        # with other off readings → off_moy_patient should be non-NaN.
        assert not np.isnan(result.loc[30, "off_moy_patient"]), (
            "off_moy_patient should be filled even when off is NaN for that visit"
        )

    def test_off_moy_patient_correct_value(self, simple_df):
        """Patient 1: mean(off) ignoring NaN = (10+20)/2 = 15."""
        result = add_patient_features(simple_df)
        p1 = result[result["patient_id"] == 1]
        np.testing.assert_allclose(
            p1["off_moy_patient"].values,
            15.0,
            err_msg="off_moy_patient for patient 1 should be 15.0",
        )

    def test_n_visites_correct(self, simple_df):
        result = add_patient_features(simple_df)
        assert (result[result["patient_id"] == 1]["n_visites"] == 3).all()
        assert (result[result["patient_id"] == 2]["n_visites"] == 2).all()
        assert (result[result["patient_id"] == 3]["n_visites"] == 1).all()

    def test_duree_maladie_correct(self, simple_df):
        result = add_patient_features(simple_df)
        expected = simple_df["age"] - simple_df["age_at_diagnosis"]
        pd.testing.assert_series_equal(
            result["duree_maladie"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_age_depuis_1re_visite_zero_for_first(self, simple_df):
        """age_depuis_1re_visite must be 0 on the earliest visit of each patient."""
        result = add_patient_features(simple_df)
        for pid, grp in result.groupby("patient_id"):
            assert grp["age_depuis_1re_visite"].min() == 0.0, (
                f"patient {pid}: min age_depuis_1re_visite should be 0"
            )

    def test_off_tendance_patient_prediction(self, simple_df):
        """Patient 1: off=10 at age 50, off=20 at age 51 → trend at age 52 = 30."""
        result = add_patient_features(simple_df)
        val = result.loc[30, "off_tendance_patient"]   # index 30 is patient 1 at age 52
        np.testing.assert_allclose(val, 30.0, rtol=1e-6,
                                   err_msg="Trend prediction at age 52 should be 30")

    def test_off_tendance_nan_for_single_off(self, simple_df):
        """Patient 3 has only one non-NaN off → trend must be NaN."""
        result = add_patient_features(simple_df)
        p3 = result[result["patient_id"] == 3]
        assert p3["off_tendance_patient"].isna().all(), (
            "Patient with a single off value should get NaN trend"
        )

    def test_pente_off_nan_for_single_off(self, simple_df):
        result = add_patient_features(simple_df)
        p3 = result[result["patient_id"] == 3]
        assert p3["pente_off_patient"].isna().all()


# ---------------------------------------------------------------------------
# patient_trend (standalone)
# ---------------------------------------------------------------------------

class TestPatientTrend:
    def test_linear_extrapolation(self, simple_df):
        """off=10 at 50, off=20 at 51 → fitted at 52 should be 30."""
        result = patient_trend(simple_df, "off")
        np.testing.assert_allclose(result.loc[30], 30.0, rtol=1e-6)

    def test_aligned_on_df_index(self, simple_df):
        result = patient_trend(simple_df, "off")
        pd.testing.assert_index_equal(result.index, simple_df.index)

    def test_nan_for_single_point(self, simple_df):
        """Patient 3 has a single non-NaN off → trend is NaN."""
        result = patient_trend(simple_df, "off")
        p3_idx = simple_df[simple_df["patient_id"] == 3].index
        assert result.loc[p3_idx].isna().all()

    def test_nan_when_missing_col(self):
        """All off=NaN for a patient → trend is NaN."""
        df = pd.DataFrame({
            "patient_id": [1, 1],
            "age":        [50.0, 51.0],
            "off":        [np.nan, np.nan],
        })
        result = patient_trend(df, "off")
        assert result.isna().all()
