"""Patient-level feature engineering for the Parkinson's motor score experiments.

All functions are **pure** (stateless, no ``fit``/``transform`` API) so they
can be wrapped with :class:`sklearn.preprocessing.FunctionTransformer` and
placed at the head of any sklearn pipeline.

.. important::
   These functions operate **only on columns present in X** — never on the
   target — so they are safe to call inside a cross-validation fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def patient_trend(
    df: pd.DataFrame,
    col: str,
    by: str = "patient_id",
    x: str = "age",
) -> pd.Series:
    """Fit a linear trend ``col ~ x`` per patient and return the fitted value.

    For each patient the function fits a degree-1 polynomial on the rows where
    ``col`` is not missing.  It then evaluates that line at **every** visit of
    the patient (including those where ``col`` is missing), so the returned
    Series is fully aligned with ``df``.

    Returns ``NaN`` for a patient when:
    - fewer than 2 non-missing observations of ``col``, or
    - all non-missing observations share the same ``x`` value (slope is
      undefined).

    Parameters
    ----------
    df : pandas.DataFrame
        Input frame (X).  Must contain ``col``, ``by``, and ``x``.
    col : str
        Column to model (e.g. ``"off"``).
    by : str, optional
        Patient identifier column.  Default ``"patient_id"``.
    x : str, optional
        Covariate for the linear fit.  Default ``"age"``.

    Returns
    -------
    pandas.Series
        Fitted values aligned on ``df.index``.  Name is
        ``f"{col}_tendance_{by.split('_')[0]}"``.
    """
    result = pd.Series(np.nan, index=df.index, name=f"{col}_tendance_{by.split('_')[0]}")

    for pid, grp in df.groupby(by, sort=False):
        mask = grp[col].notna()
        xs = grp.loc[mask, x].to_numpy(dtype=float)
        ys = grp.loc[mask, col].to_numpy(dtype=float)

        if len(xs) < 2 or np.unique(xs).size < 2:
            continue  # leave NaN

        coeffs = np.polyfit(xs, ys, 1)          # [slope, intercept]
        fitted = np.polyval(coeffs, grp[x].to_numpy(dtype=float))
        result.loc[grp.index] = fitted

    return result


def add_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* augmented with per-patient aggregate features.

    ``patient_id`` is **preserved** in the output — the caller (pipeline) is
    responsible for dropping it before the vectoriser.

    No target column is required or used.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw feature frame (X) with at least the columns ``patient_id``,
        ``age``, ``age_at_diagnosis``, ``off``, ``on``, ``ledd``.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with extra columns listed below.

    New columns
    -----------
    duree_maladie         : age - age_at_diagnosis
    off_moy_patient       : mean(off) per patient (filled on all visits)
    on_moy_patient        : mean(on) per patient
    ledd_moy_patient      : mean(ledd) per patient
    n_off_patient         : count of non-NaN off values per patient
    n_visites             : number of visits per patient
    age_depuis_1re_visite : age - min(age) per patient
    off_tendance_patient  : fitted value of linear off~age trend per patient
    pente_off_patient     : slope of that linear trend (points / year)
    """
    df = df.copy()
    grp = df.groupby("patient_id")

    df["duree_maladie"] = df["age"] - df["age_at_diagnosis"]

    df["off_moy_patient"] = grp["off"].transform("mean")
    df["on_moy_patient"] = grp["on"].transform("mean")
    df["ledd_moy_patient"] = grp["ledd"].transform("mean")

    df["n_off_patient"] = grp["off"].transform("count")
    df["n_visites"] = grp["patient_id"].transform("count")

    df["age_depuis_1re_visite"] = df["age"] - grp["age"].transform("min")

    df["off_tendance_patient"] = patient_trend(df, "off")

    # Slope per patient (scalar, broadcast to all visits of the patient).
    slopes = pd.Series(np.nan, index=df.index, name="pente_off_patient")
    for pid, g in df.groupby("patient_id"):
        mask = g["off"].notna()
        xs = g.loc[mask, "age"].to_numpy(dtype=float)
        ys = g.loc[mask, "off"].to_numpy(dtype=float)
        if len(xs) < 2 or np.unique(xs).size < 2:
            continue
        slope = np.polyfit(xs, ys, 1)[0]
        slopes.loc[g.index] = slope
    df["pente_off_patient"] = slopes

    return df
