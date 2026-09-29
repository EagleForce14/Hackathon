"""Smoke test for experiments/05_upper_envelope.py.

Fits on a patient-disjoint train subset with all 05c options active
(patient_features + drug_timing + envelope + tuned HGBR + smoothing),
predicts on a held-out subset, and asserts that the prediction count equals
the held-out row count.
"""

from __future__ import annotations

import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline

from parkinson import PROJECT_ROOT
from parkinson.data import GROUP_COL, TARGET_COL, load_raw
from parkinson.features import add_patient_features
from parkinson.pharmaco import DrugTimingFeatures
from parkinson.envelope import UpperEnvelope
from parkinson.smoothing import PatientSmoothedRegressor

DATA_DIR = PROJECT_ROOT / "data"

# Soft-assertion baseline: deliberately loose — update once 05 CV run lands.
CV_MAE_MEAN_PLACEHOLDER = 100.0

HGB_PARAMS = dict(
    max_iter=1000,
    learning_rate=0.03,
    early_stopping=False,
    l2_regularization=1.0,
    random_state=0,
)


@pytest.fixture
def train_predict_split():
    """Return a patient-disjoint train/predict split of the training data."""
    df = load_raw(DATA_DIR)
    patients = sorted(df[GROUP_COL].unique())
    n_holdout = max(1, len(patients) // 5)
    holdout = set(patients[-n_holdout:])
    return (
        df[~df[GROUP_COL].isin(holdout)].copy(),
        df[df[GROUP_COL].isin(holdout)].copy(),
    )


def test_05_upper_envelope_predict_row_count(train_predict_split):
    """Fit with full 05c stack; predict on held-out patients.

    Verifies structural correctness (row count) and a loose MAE bound.
    """
    import skrub
    from sklearn.ensemble import HistGradientBoostingRegressor

    train_df, predict_df = train_predict_split
    n_predict = len(predict_df)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_predict = predict_df.drop(columns=[TARGET_COL])
    y_true = predict_df[TARGET_COL]

    inner_pipeline = make_pipeline(
        FunctionTransformer(add_patient_features),
        DrugTimingFeatures(),
        UpperEnvelope(),
        skrub.DropCols(["patient_id"]),
        skrub.TableVectorizer(),
        HistGradientBoostingRegressor(**HGB_PARAMS),
    )
    model = PatientSmoothedRegressor(inner_pipeline, deg=2, weight=1.0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_predict)

    # HARD: structural row-count assertion.
    assert len(predictions) == n_predict, (
        f"got {len(predictions)} predictions for {n_predict} rows."
    )

    # SOFT: predictions are not garbage (expect MAE around 3.5 based on prototype).
    smoke_mae = mean_absolute_error(y_true, predictions)
    assert smoke_mae < 3 * CV_MAE_MEAN_PLACEHOLDER, (
        f"smoke MAE {smoke_mae:.1f} > 3 × placeholder ({CV_MAE_MEAN_PLACEHOLDER:.1f})."
    )
