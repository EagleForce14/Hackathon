"""Pharmacokinetic feature transformer for the Parkinson's motor score experiments.

The key insight (from ``data/eda_correlations.html`` §4) is that the ratio
``on / off`` depends strongly on the delay since the last levodopa dose:
it is close to 1 near intake and drops to a plateau of ≈ 0.56 after 1.5 h.
Correcting ``on`` for this delay gives an estimate of the OFF motor score even
for the 42 % of visits where ``off`` was not measured.

All pharmacokinetic learning happens inside ``fit`` using **only X** (never y),
so there is no target leakage.  The transformer is stateful (stores ``curve_``
and ``plateau_`` after fitting) and is fully sklearn-compatible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from parkinson.features import patient_trend


class DrugTimingFeatures(TransformerMixin, BaseEstimator):
    """Learn an on/off correction curve from timing data and add derived features.

    Parameters
    ----------
    bins : tuple of float
        Bin edges (in hours) used to discretise ``time_since_intake_on`` when
        estimating the on/off ratio curve.  The last edge is treated as
        right-open (``pd.cut`` with ``include_lowest=True``).
    plateau_from : float
        Delays >= this value (hours) define the "plateau" — the steady-state
        ratio used when a delay falls outside the observed bins or is missing.
    min_count : int
        Minimum number of rows per bin required to use the observed median;
        bins with fewer rows fall back to ``plateau_``.

    Attributes
    ----------
    curve_ : pandas.Series
        Median on/off ratio indexed by ``pd.Interval`` bin.
    plateau_ : float
        Median on/off ratio for delays >= ``plateau_from``.
    bins_ : list of float
        Bin edges actually used (copy of ``bins`` parameter).
    """

    def __init__(
        self,
        bins: tuple[float, ...] = (0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 7),
        plateau_from: float = 1.5,
        min_count: int = 30,
    ) -> None:
        self.bins = bins
        self.plateau_from = plateau_from
        self.min_count = min_count

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "DrugTimingFeatures":
        """Learn the on/off ratio curve from the training fold.

        Uses only rows where ``on``, ``off``, and ``time_since_intake_on`` are
        all present and ``off > 0``.  Never uses ``y``.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix (must contain ``on``, ``off``,
            ``time_since_intake_on``).
        y : ignored
            Present for sklearn API compatibility only.

        Returns
        -------
        self
        """
        mask = (
            X["on"].notna()
            & X["off"].notna()
            & (X["off"] > 0)
            & X["time_since_intake_on"].notna()
        )
        sub = X.loc[mask].copy()
        sub["_ratio"] = sub["on"] / sub["off"]
        sub["_bin"] = pd.cut(
            sub["time_since_intake_on"],
            bins=list(self.bins),
            include_lowest=True,
        )

        # Plateau: median ratio for delays >= plateau_from.
        plateau_mask = sub["time_since_intake_on"] >= self.plateau_from
        if plateau_mask.sum() > 0:
            self.plateau_ = float(sub.loc[plateau_mask, "_ratio"].median())
        else:
            self.plateau_ = float(sub["_ratio"].median())

        # Per-bin median; fall back to plateau_ when bin has too few rows.
        curve = sub.groupby("_bin", observed=True)["_ratio"].agg(
            ["median", "count"]
        )
        curve["median"] = curve.apply(
            lambda row: row["median"] if row["count"] >= self.min_count else self.plateau_,
            axis=1,
        )
        self.curve_ = curve["median"]
        self.bins_ = list(self.bins)
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add pharmacokinetic and derived features to X.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.  All rows of a patient must be passed together so
            that the per-patient aggregates (``off_estime_moy_patient``,
            ``off_estime_tendance``) are computed correctly.

        Returns
        -------
        pandas.DataFrame
            Copy of X with extra columns (see class docstring).
        """
        df = X.copy()

        # --- ratio_attendu -------------------------------------------
        bins_cut = pd.cut(
            df["time_since_intake_on"],
            bins=self.bins_,
            include_lowest=True,
        )
        ratio_attendu = bins_cut.map(self.curve_)
        # Missing delay or out-of-range → use plateau.
        ratio_attendu = ratio_attendu.where(ratio_attendu.notna(), other=self.plateau_)
        df["ratio_attendu"] = ratio_attendu.values

        # --- on_corrige ----------------------------------------------
        df["on_corrige"] = df["on"] / df["ratio_attendu"]

        # --- off_estime: use measured off when available -------------
        df["off_estime"] = df["off"].where(df["off"].notna(), other=df["on_corrige"])

        # --- indicator flags -----------------------------------------
        df["traite"] = ((df["ledd"].notna()) | (df["on"].notna())).astype(int)
        df["on_manquant"] = df["on"].isna().astype(int)
        df["off_manquant"] = df["off"].isna().astype(int)

        # --- per-patient aggregates on off_estime --------------------
        grp = df.groupby("patient_id")
        df["off_estime_moy_patient"] = grp["off_estime"].transform("mean")
        df["part_traite_patient"] = grp["traite"].transform("mean")

        # --- off_estime_tendance: reuse patient_trend from features.py --
        df["off_estime_tendance"] = patient_trend(df, "off_estime")

        return df
