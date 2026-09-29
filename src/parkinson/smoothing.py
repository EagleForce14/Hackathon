"""Patient-level prediction smoothing for the Parkinson's motor score experiments.

After the HGBR produces raw predictions, we know from the EDA that the true
OFF follows a smooth parabolic trajectory per patient.  Fitting a low-degree
polynomial to the per-patient predictions and blending it back reduces
visit-to-visit noise without touching the between-patient ordering.

``PatientSmoothedRegressor`` wraps any sklearn regressor and applies this
post-processing in ``predict``.  It is itself a valid sklearn estimator so
skore can evaluate it end-to-end in a single ``skore.evaluate`` call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone


class PatientSmoothedRegressor(RegressorMixin, BaseEstimator):
    """Wrap an estimator and smooth per-patient predictions with a polynomial.

    Parameters
    ----------
    estimator : sklearn estimator
        A fitted or unfitted regressor.  Must accept a DataFrame as X (the
        internal pipeline is responsible for encoding).
    deg : int, optional
        Polynomial degree for smoothing.  Default 2 (parabola).
    weight : float, optional
        Blend weight for the smoothed prediction.
        ``output = weight * smoothed + (1 - weight) * raw``.
        Default 1.0 (fully smoothed).
    group_col : str, optional
        Column in X that identifies the patient.  Default ``"patient_id"``.
    x_col : str, optional
        Column in X used as the polynomial covariate.  Default ``"age"``.

    Notes
    -----
    - ``fit`` trains the wrapped estimator on the full X, y.
    - ``predict`` applies the estimator, then for each patient with
      ``> deg + 1`` distinct ``x_col`` values fits a polynomial of degree
      ``deg`` on ``(x_col, raw_predictions)`` and blends the result.
    - Patients with ``<= deg + 1`` distinct ages keep their raw predictions
      unchanged (no extrapolation risk).
    - ``y`` is **never** used in ``predict``.
    - Row order is preserved.
    """

    def __init__(
        self,
        estimator,
        deg: int = 2,
        weight: float = 1.0,
        group_col: str = "patient_id",
        x_col: str = "age",
    ) -> None:
        self.estimator = estimator
        self.deg = deg
        self.weight = weight
        self.group_col = group_col
        self.x_col = x_col

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """Clone and train the wrapped estimator on (X, y).

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix (must contain ``group_col`` and ``x_col``).
        y : array-like
            Target values.

        Returns
        -------
        self
        """
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Predict and apply per-patient polynomial smoothing.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.  All visits of a patient must be present together
            so the per-patient polynomial is computed on the correct set.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Blended predictions, same order as input rows.
        """
        raw = self.estimator_.predict(X)  # shape (n,)
        output = raw.copy()

        if not isinstance(X, pd.DataFrame):
            return output  # can't group without column names

        if self.group_col not in X.columns or self.x_col not in X.columns:
            return output

        ages = X[self.x_col].to_numpy(dtype=float)

        for pid, grp in X.groupby(self.group_col, sort=False):
            iloc_pos = grp.index  # pandas index labels
            # Resolve to positional indices for numpy array indexing.
            pos = X.index.get_indexer(iloc_pos)

            a = ages[pos]
            p = raw[pos]

            if np.unique(a).size <= self.deg + 1:
                continue  # too few distinct ages — keep raw

            coeffs = np.polyfit(a, p, self.deg)
            smoothed = np.polyval(coeffs, a)
            output[pos] = self.weight * smoothed + (1 - self.weight) * p

        return output
