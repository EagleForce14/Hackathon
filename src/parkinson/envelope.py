"""Upper-envelope feature engineering for the Parkinson's motor score experiments.

The core insight (``journal/05_upper_envelope.md``) is that the measured OFF
systematically *under*-estimates the true OFF (asymmetric noise, median bias
≈ 15 %).  Summarising each patient's trajectory by its **upper** quantiles and
by an iteratively re-weighted upper-envelope line corrects this downward pull.

All functions are **stateless** — they operate only on X (never y) and can be
safely used inside a cross-validation fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _fit_line(ages: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Return polyfit degree-1 coefficients, or None if < 2 distinct ages."""
    if len(ages) < 2 or np.unique(ages).size < 2:
        return None
    return np.polyfit(ages, vals, 1)


def _fit_parabola(ages: np.ndarray, vals: np.ndarray) -> np.ndarray | None:
    """Return polyfit degree-2 coefficients, or None if < 3 distinct ages or < 6 points."""
    if len(ages) < 6 or np.unique(ages).size < 3:
        return None
    return np.polyfit(ages, vals, 2)


def _upper_envelope_coeffs(
    ages: np.ndarray,
    vals: np.ndarray,
    deg: int,
    min_points: int,
    min_ages: int,
    n_iter: int = 2,
) -> np.ndarray | None:
    """Iteratively refit a polynomial keeping points above the median residual.

    Parameters
    ----------
    ages, vals : arrays of floats (no NaN).
    deg : polynomial degree (1 = line, 2 = parabola).
    min_points : minimum number of kept points to accept a new fit.
    min_ages : minimum number of distinct age values needed to accept new fit.
    n_iter : number of reweighting iterations.

    Returns
    -------
    coefficients or None if initial fit is impossible.
    """
    coeffs = np.polyfit(ages, vals, deg)
    for _ in range(n_iter):
        residuals = vals - np.polyval(coeffs, ages)
        med_res = np.median(residuals)
        keep = residuals >= med_res
        ka, kv = ages[keep], vals[keep]
        if ka.size < min_points or np.unique(ka).size < min_ages:
            break  # keep current coeffs
        coeffs = np.polyfit(ka, kv, deg)
    return coeffs


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def patient_envelope_features(
    df: pd.DataFrame,
    col: str,
    quantiles: tuple[float, ...] = (0.5, 0.75, 0.9),
) -> pd.DataFrame:
    """Compute upper-envelope statistics for *col* and return them as a DataFrame.

    All statistics are computed per patient on non-missing values of *col* and
    then broadcast to **every** visit of that patient (including visits where
    *col* is NaN).

    Parameters
    ----------
    df : pandas.DataFrame
        Feature frame containing ``patient_id``, ``age``, and ``col``.
    col : str
        Column to analyse (e.g. ``"off"``, ``"off_estime"``).
    quantiles : tuple of float, optional
        Quantiles for the quantile-shifted line features.

    Returns
    -------
    pandas.DataFrame
        New columns only, aligned on ``df.index``.  Columns created:

        ``{col}_max_patient``
            Per-patient maximum of *col* (needs ≥ 1 non-NaN value).
        ``{col}_q{q*100:.0f}`` for each q in *quantiles*
            Linear trend evaluated at the visit age, shifted up by the
            empirical quantile of residuals (needs ≥ 2 points, ≥ 2 distinct ages).
        ``{col}_enveloppe``
            Iterative upper-envelope line (degree 1) evaluated at visit age
            (needs ≥ 2 points, ≥ 2 distinct ages).
        ``{col}_env2``
            Iterative upper-envelope parabola (degree 2) evaluated at visit age
            (needs ≥ 6 points, ≥ 3 distinct ages).
    """
    n = len(df)
    q_names = [f"{col}_q{int(q * 100)}" for q in quantiles]
    out = pd.DataFrame(
        np.nan,
        index=df.index,
        columns=[f"{col}_max_patient", *q_names, f"{col}_enveloppe", f"{col}_env2"],
    )

    for pid, grp in df.groupby("patient_id", sort=False):
        idx = grp.index
        ages_all = grp["age"].to_numpy(dtype=float)

        mask = grp[col].notna()
        a = grp.loc[mask, "age"].to_numpy(dtype=float)
        v = grp.loc[mask, col].to_numpy(dtype=float)

        if len(v) == 0:
            continue

        # max_patient: always available if ≥ 1 value
        out.loc[idx, f"{col}_max_patient"] = v.max()

        # quantile lines and envelope: need ≥ 2 points, ≥ 2 distinct ages
        base_coeffs = _fit_line(a, v)
        if base_coeffs is None:
            continue

        base_vals = np.polyval(base_coeffs, a)
        residuals = v - base_vals

        # Quantile-shifted lines
        for q, qname in zip(quantiles, q_names):
            shift = np.quantile(residuals, q)
            fitted = np.polyval(base_coeffs, ages_all) + shift
            out.loc[idx, qname] = fitted

        # Upper-envelope line (deg=1, min 2 points & 2 ages)
        env_coeffs = _upper_envelope_coeffs(a, v, deg=1, min_points=2, min_ages=2)
        if env_coeffs is not None:
            out.loc[idx, f"{col}_enveloppe"] = np.polyval(env_coeffs, ages_all)

        # Upper-envelope parabola (deg=2, min 6 points & 3 ages)
        env2_coeffs = _upper_envelope_coeffs(a, v, deg=2, min_points=4, min_ages=3)
        if env2_coeffs is not None and len(a) >= 6 and np.unique(a).size >= 3:
            out.loc[idx, f"{col}_env2"] = np.polyval(env2_coeffs, ages_all)

    return out


# ---------------------------------------------------------------------------
# Sklearn transformer
# ---------------------------------------------------------------------------

class UpperEnvelope(TransformerMixin, BaseEstimator):
    """Add upper-envelope features for multiple columns.

    Parameters
    ----------
    cols : tuple of str, optional
        Columns to process.  Defaults to ``("off", "off_estime", "on_corrige")``.
        ``off_estime`` and ``on_corrige`` are produced by
        :class:`~parkinson.pharmaco.DrugTimingFeatures` (experiment 04).
    quantiles : tuple of float, optional
        Quantiles passed to :func:`patient_envelope_features`.

    Notes
    -----
    ``fit`` is a no-op — all statistics are computed from X in ``transform``
    (no target involved).  ``envelope=True`` in ``build_learner`` requires
    ``drug_timing=True`` because it uses ``off_estime`` and ``on_corrige``.
    """

    def __init__(
        self,
        cols: tuple[str, ...] = ("off", "off_estime", "on_corrige"),
        quantiles: tuple[float, ...] = (0.5, 0.75, 0.9),
    ) -> None:
        self.cols = cols
        self.quantiles = quantiles

    def fit(self, X: pd.DataFrame, y=None) -> "UpperEnvelope":
        """No-op — envelope features require no fitting."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add envelope features for each configured column.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature frame.  All visits of a patient must be present together.

        Returns
        -------
        pandas.DataFrame
            Copy of X with new envelope columns appended.
        """
        df = X.copy()
        for col in self.cols:
            if col not in df.columns:
                continue
            new_cols = patient_envelope_features(df, col, quantiles=self.quantiles)
            for c in new_cols.columns:
                df[c] = new_cols[c].values
        return df
