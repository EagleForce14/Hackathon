"""Learner declaration for the Parkinson's motor score prediction experiments.

Builds a skrub DataOps graph that:
1. Loads the raw training data from a source-bound ``data_dir`` variable.
2. Optionally drops columns before marking the feature matrix as X
   (keeping ``patient_id`` for grouped CV).
3. Applies a ``TableVectorizer`` to encode categorical columns automatically.
4. Fits a ``HistGradientBoostingRegressor``, which handles NaN natively —
   no imputation step required given the high missingness in this dataset.
"""

from __future__ import annotations

from pathlib import Path

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from parkinson.data import GROUP_COL, TARGET_COL, load_raw


def build_learner(
    data_dir_preview: str | Path | None = None,
    drop_cols: tuple[str, ...] = (),
):
    """Return the unfit learner (skrub SkrubLearner).

    Parameters
    ----------
    data_dir_preview : str or Path or None, optional
        Preview value for the ``data_dir`` source variable. Pass an absolute
        path (e.g. ``PROJECT_ROOT / "data"``) when iterating interactively
        so ``learner.skb.preview()`` works. Leave as ``None`` for fit /
        cross-validate runs — ``skore.evaluate(data={"data_dir": ...})``
        supplies the binding.
    drop_cols : tuple of str, optional
        Feature columns to drop before marking X. Default ``()`` preserves
        the full column set (01_baseline behaviour). Pass
        ``("off", "time_since_intake_off")`` for 02_no_off_feature.

    Returns
    -------
    skrub.SkrubLearner
        Unfit learner ready to be passed to ``skore.evaluate``.
    """
    if data_dir_preview is not None:
        data_dir = skrub.var("data_dir", value=str(data_dir_preview))
    else:
        data_dir = skrub.var("data_dir")

    # Layer 1: load raw data from the source-bound directory.
    data = data_dir.skb.apply_func(load_raw)

    # Layer 2: mark X and y on the source frame (IID — no cross-row features).
    # cv + split_kwargs together wire GroupKFold on patient_id at the marker.
    feature_frame = data.drop(columns=[TARGET_COL, *drop_cols])
    X = feature_frame.skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data[GROUP_COL]},
    )
    y = data[TARGET_COL].skb.mark_as_y()

    # Layer 3: TableVectorizer encodes categoricals; HGBR handles NaN natively.
    predictions = X.skb.apply(
        make_pipeline(
            skrub.TableVectorizer(),
            HistGradientBoostingRegressor(random_state=0),
        ),
        y=y,
    )
    return predictions.skb.make_learner()
