"""Smoke test for experiments/03_patient_features.py.

Fits on a patient-disjoint train subset with ``patient_features=True``,
predicts on a held-out subset, and asserts that the prediction count equals
the held-out row count.
"""

from __future__ import annotations

import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from parkinson import PROJECT_ROOT
from parkinson.data import GROUP_COL, TARGET_COL, load_raw
from parkinson.features import add_patient_features

DATA_DIR = PROJECT_ROOT / "data"

# Soft-assertion baseline: deliberately loose — update once 03 CV run lands.
CV_MAE_MEAN_PLACEHOLDER = 100.0


@pytest.fixture
def train_predict_split():
    """Return a patient-disjoint train/predict split of the training data.

    Returns
    -------
    tuple of (train_df, predict_df)
        ``train_df`` and ``predict_df`` have no patient overlap.
        ``predict_df`` retains the ``target`` column for the soft assertion.
    """
    df = load_raw(DATA_DIR)
    patients = sorted(df[GROUP_COL].unique())
    n_holdout = max(1, len(patients) // 5)
    holdout = set(patients[-n_holdout:])
    return (
        df[~df[GROUP_COL].isin(holdout)].copy(),
        df[df[GROUP_COL].isin(holdout)].copy(),
    )


def test_03_patient_features_predict_row_count(train_predict_split):
    """Fit with patient features on train patients; predict on held-out patients.

    Parameters
    ----------
    train_predict_split : tuple of (pandas.DataFrame, pandas.DataFrame)
        Patient-disjoint train and predict DataFrames (both include target).
    """
    import skrub
    from sklearn.ensemble import HistGradientBoostingRegressor

    train_df, predict_df = train_predict_split
    n_predict = len(predict_df)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_predict = predict_df.drop(columns=[TARGET_COL])
    y_true = predict_df[TARGET_COL]

    model = make_pipeline(
        FunctionTransformer(add_patient_features),
        skrub.DropCols(["patient_id"]),
        skrub.TableVectorizer(),
        HistGradientBoostingRegressor(random_state=0),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_predict)

    # HARD: structural row-count assertion.
    assert len(predictions) == n_predict, (
        f"got {len(predictions)} predictions for {n_predict} rows."
    )

    # SOFT: predictions are not garbage (expect MAE around 5–6 based on prototype).
    smoke_mae = mean_absolute_error(y_true, predictions)
    assert smoke_mae < 3 * CV_MAE_MEAN_PLACEHOLDER, (
        f"smoke MAE {smoke_mae:.1f} > 3 × placeholder ({CV_MAE_MEAN_PLACEHOLDER:.1f})."
    )
