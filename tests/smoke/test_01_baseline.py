"""Smoke test for experiments/01_baseline.py.

IID pipeline with no cross-row features. Fits on a subset of training rows
and predicts on a disjoint held-out subset, asserting that the prediction
count equals the held-out row count.

The soft MAE bound (``CV_MAE_MEAN_PLACEHOLDER``) is a loose placeholder.
Update it from ``journal/01_baseline.md`` § Status.headline after the first
CV run completes.
"""

from __future__ import annotations

import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

from parkinson import PROJECT_ROOT
from parkinson.data import GROUP_COL, TARGET_COL, load_raw

DATA_DIR = PROJECT_ROOT / "data"

# Soft-assertion baseline: update from journal/01_baseline.md § Headline.
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


def test_01_baseline_predict_row_count(train_predict_split):
    """Fit on train patients, predict on held-out patients; check row count.

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

    # Use the same encoding stack as the pipeline: TableVectorizer + HGBR.
    model = make_pipeline(
        skrub.TableVectorizer(),
        HistGradientBoostingRegressor(random_state=0),
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_predict)

    # HARD: structural row-count assertion.
    assert len(predictions) == n_predict, (
        f"got {len(predictions)} predictions for {n_predict} rows."
    )

    # SOFT: predictions are not garbage.
    smoke_mae = mean_absolute_error(y_true, predictions)
    assert smoke_mae < 3 * CV_MAE_MEAN_PLACEHOLDER, (
        f"smoke MAE {smoke_mae:.1f} > 3 × placeholder ({CV_MAE_MEAN_PLACEHOLDER:.1f})."
    )
