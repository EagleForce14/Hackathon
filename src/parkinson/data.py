"""Data loading for the Parkinson's motor score prediction workspace.

Loads `X_train.csv` and `y_train.csv` from a given directory, joins them on
the shared index, and returns `(X, y)` with the patient ID column kept so
that `GroupKFold` can use it as the grouping key in cross-validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COL = "target"
GROUP_COL = "patient_id"


def load_raw(data_dir: str | Path) -> pd.DataFrame:
    """Load and join the training features and target into one DataFrame.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``X_train.csv`` and ``y_train.csv``.

    Returns
    -------
    pandas.DataFrame
        Full training frame with all feature columns plus ``target``.
    """
    data_dir = Path(data_dir)
    X = pd.read_csv(data_dir / "X_train.csv", index_col="Index")
    y = pd.read_csv(data_dir / "y_train.csv", index_col="Index")
    return X.join(y)


def load_dataset(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` ready for sklearn-style fits.

    Parameters
    ----------
    data_dir : str or Path or None, optional
        Path to the data directory. Defaults to ``<PROJECT_ROOT>/data``.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix (includes ``patient_id`` for grouped CV).
    y : pandas.Series
        Target series (true OFF MDS-UPDRS score).
    """
    if data_dir is None:
        from parkinson import PROJECT_ROOT

        data_dir = PROJECT_ROOT / "data"
    df = load_raw(data_dir)
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]
