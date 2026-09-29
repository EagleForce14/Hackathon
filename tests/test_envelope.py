"""Unit tests for parkinson.envelope.

Uses small hand-crafted DataFrames to verify:
- off_q90 > off_q50 at every visit of a patient with enough data.
- off_enveloppe is in the upper range (≥ mean) of the measurements.
- Row count and index are preserved.
- A visit without off still receives patient-level values.
- A patient with one measurement has off_max_patient but NaN for the rest.
- The function works without a target column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from parkinson.envelope import UpperEnvelope, patient_envelope_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def five_visit_df():
    """Patient with off = [40, 25, 41, 30, 39] at ages [60.0, 60.2, 60.4, 60.6, 60.8].

    Mean = 35.  Upper envelope should be well above 35.
    Also includes a second patient with only one off measurement.
    Also includes a NaN-off visit for patient 1 (index=5).
    """
    return pd.DataFrame(
        {
            "patient_id": [1,    1,    1,    1,    1,    1,    2],
            "age":        [60.0, 60.2, 60.4, 60.6, 60.8, 61.0, 55.0],
            # idx=5 (age=61.0): off is NaN → envelope should still be filled from patient 1
            "off":        [40.0, 25.0, 41.0, 30.0, 39.0, np.nan, 50.0],
            "on":         [20.0, 18.0, 21.0, 19.0, 20.0, 18.0,  25.0],
            "ledd":       [500.] * 7,
        },
        index=list(range(10, 17)),
    )


@pytest.fixture
def single_off_df():
    """Patient with exactly one non-NaN off — most derived cols should be NaN."""
    return pd.DataFrame(
        {
            "patient_id": [3,    3],
            "age":        [70.0, 71.0],
            "off":        [30.0, np.nan],
            "on":         [15.0, 14.0],
            "ledd":       [300., 300.],
        },
        index=[100, 101],
    )


# ---------------------------------------------------------------------------
# patient_envelope_features
# ---------------------------------------------------------------------------

class TestPatientEnvelopeFeatures:
    def test_row_count_preserved(self, five_visit_df):
        result = patient_envelope_features(five_visit_df, "off")
        assert len(result) == len(five_visit_df)

    def test_index_preserved(self, five_visit_df):
        result = patient_envelope_features(five_visit_df, "off")
        pd.testing.assert_index_equal(result.index, five_visit_df.index)

    def test_q90_gt_q50(self, five_visit_df):
        """off_q90 must be strictly greater than off_q50 at each visit of patient 1."""
        result = patient_envelope_features(five_visit_df, "off")
        p1 = result[five_visit_df["patient_id"] == 1]
        assert (p1["off_q90"] > p1["off_q50"]).all(), (
            "off_q90 should be > off_q50 for patient 1"
        )

    def test_enveloppe_above_mean(self, five_visit_df):
        """off_enveloppe must be ≥ mean(off) of patient 1 on measured visits."""
        result = patient_envelope_features(five_visit_df, "off")
        mean_off = five_visit_df[five_visit_df["patient_id"] == 1]["off"].mean()  # 35
        # Measured visits (idx 10–14)
        p1_measured = result.loc[10:14, "off_enveloppe"]
        assert (p1_measured >= mean_off - 1e-9).all(), (
            f"off_enveloppe ({p1_measured.values}) should be >= mean ({mean_off})"
        )

    def test_enveloppe_in_expected_range(self, five_visit_df):
        """off_enveloppe should be between 38 and 42 for patient 1."""
        result = patient_envelope_features(five_visit_df, "off")
        p1 = result[five_visit_df["patient_id"] == 1]["off_enveloppe"]
        assert (p1 >= 38).all() and (p1 <= 43).all(), (
            f"off_enveloppe out of [38, 42] range: {p1.values}"
        )

    def test_nan_off_visit_gets_envelope_value(self, five_visit_df):
        """Visit at index=15 (age=61.0, off=NaN for patient 1) must get a non-NaN envelope."""
        result = patient_envelope_features(five_visit_df, "off")
        assert not np.isnan(result.loc[15, "off_enveloppe"]), (
            "Visit with NaN off should still get off_enveloppe from patient trend"
        )

    def test_max_patient_always_filled(self, five_visit_df):
        """off_max_patient must be non-NaN for all patients with ≥ 1 observation."""
        result = patient_envelope_features(five_visit_df, "off")
        assert result["off_max_patient"].notna().all()

    def test_single_off_max_filled_rest_nan(self, single_off_df):
        """Patient with one off: max is filled, quantile/envelope cols are NaN."""
        result = patient_envelope_features(single_off_df, "off")
        # max should be filled
        assert result["off_max_patient"].notna().all()
        # derived cols (need ≥ 2 points) should be NaN
        for col in ("off_q50", "off_q75", "off_q90", "off_enveloppe"):
            assert result[col].isna().all(), f"{col} should be NaN for 1-point patient"

    def test_no_target_column_required(self, five_visit_df):
        """Function must work on a frame with no target column."""
        assert "target" not in five_visit_df.columns
        result = patient_envelope_features(five_visit_df, "off")
        assert result is not None


# ---------------------------------------------------------------------------
# UpperEnvelope transformer
# ---------------------------------------------------------------------------

class TestUpperEnvelope:
    def test_fit_returns_self(self, five_visit_df):
        t = UpperEnvelope(cols=("off",))
        assert t.fit(five_visit_df) is t

    def test_transform_row_count(self, five_visit_df):
        t = UpperEnvelope(cols=("off",))
        t.fit(five_visit_df)
        result = t.transform(five_visit_df)
        assert len(result) == len(five_visit_df)

    def test_transform_index_preserved(self, five_visit_df):
        t = UpperEnvelope(cols=("off",))
        t.fit(five_visit_df)
        result = t.transform(five_visit_df)
        pd.testing.assert_index_equal(result.index, five_visit_df.index)

    def test_transform_input_not_mutated(self, five_visit_df):
        cols_before = list(five_visit_df.columns)
        t = UpperEnvelope(cols=("off",))
        t.fit(five_visit_df)
        t.transform(five_visit_df)
        assert list(five_visit_df.columns) == cols_before

    def test_transform_adds_expected_columns(self, five_visit_df):
        t = UpperEnvelope(cols=("off",), quantiles=(0.5, 0.75, 0.9))
        t.fit(five_visit_df)
        result = t.transform(five_visit_df)
        for col in ("off_max_patient", "off_q50", "off_q75", "off_q90",
                    "off_enveloppe", "off_env2"):
            assert col in result.columns, f"Expected column {col} in output"

    def test_missing_col_skipped_gracefully(self, five_visit_df):
        """If a requested col is not in X, it should be silently skipped."""
        t = UpperEnvelope(cols=("off", "off_estime"))  # off_estime not present
        t.fit(five_visit_df)
        result = t.transform(five_visit_df)
        assert "off_max_patient" in result.columns  # off was processed
        assert "off_estime_max_patient" not in result.columns  # off_estime skipped
