"""Learner declaration for the Parkinson's motor score prediction experiments.

Builds a skrub DataOps graph that:
1. Loads the raw training data from a source-bound ``data_dir`` variable.
2. Optionally drops columns before marking the feature matrix as X
   (keeping ``patient_id`` for grouped CV).
3. Optionally computes per-patient aggregate features (experiment 03+).
4. Optionally applies pharmacokinetic timing correction (experiment 04+).
5. Optionally adds upper-envelope features (experiment 05a+).
6. Drops ``patient_id`` with ``skrub.DropCols`` so the vectoriser never
   sees it (patients in the test set are disjoint from train).
7. Applies a ``TableVectorizer`` to encode categorical columns automatically.
8. Fits a ``HistGradientBoostingRegressor`` (optionally tuned, experiment 05b).
9. Optionally wraps the pipeline in ``PatientSmoothedRegressor`` (experiment 05c).
"""

from __future__ import annotations

from pathlib import Path

import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from parkinson.data import GROUP_COL, TARGET_COL, load_raw


def build_learner(
    data_dir_preview: str | Path | None = None,
    drop_cols: tuple[str, ...] = (),
    patient_features: bool = False,
    drug_timing: bool = False,
    envelope: bool = False,
    hgb_params: dict | None = None,
    smooth: bool = False,
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
    patient_features : bool, optional
        When ``True``, insert :func:`parkinson.features.add_patient_features`
        at the head of the pipeline and drop ``patient_id`` with
        ``skrub.DropCols`` before the vectoriser.  ``patient_id`` is kept in
        X up to that point so the GroupKFold wiring still works.
        Default ``False`` (01/02 baseline behaviour — note: ``patient_id`` is
        still dropped by ``DropCols`` when this flag is ``True``; with
        ``False`` the original behaviour is preserved and ``patient_id`` passes
        through to ``TableVectorizer`` as before).
    drug_timing : bool, optional
        When ``True``, insert :class:`parkinson.pharmaco.DrugTimingFeatures`
        after the patient-features step (requires ``patient_features=True``
        for the full 04 setup, but works independently).  Learns an on/off
        ratio curve from the training fold and adds ``on_corrige``,
        ``off_estime``, ``off_estime_moy_patient``, ``off_estime_tendance``,
        ``traite``, ``on_manquant``, ``off_manquant``, and
        ``part_traite_patient``.  Default ``False``.
    envelope : bool, optional
        When ``True``, insert :class:`parkinson.envelope.UpperEnvelope` after
        ``DrugTimingFeatures``.  Requires ``drug_timing=True`` (uses
        ``off_estime`` and ``on_corrige``).  Raises ``ValueError`` otherwise.
        Default ``False``.
    hgb_params : dict or None, optional
        Extra keyword arguments forwarded to
        ``HistGradientBoostingRegressor``.  Use
        ``dict(max_iter=1000, learning_rate=0.03, early_stopping=False,
        l2_regularization=1.0)`` for the tuned 05b configuration.
        Default ``None`` (keeps ``random_state=0`` only).
    smooth : bool, optional
        When ``True``, wrap the whole sklearn pipeline in
        :class:`parkinson.smoothing.PatientSmoothedRegressor` so that
        per-patient predictions are smoothed with a degree-2 polynomial.
        Default ``False``.

    Returns
    -------
    skrub.SkrubLearner
        Unfit learner ready to be passed to ``skore.evaluate``.

    Raises
    ------
    ValueError
        If ``envelope=True`` but ``drug_timing=False``.
    """
    if envelope and not drug_timing:
        raise ValueError(
            "envelope=True requires drug_timing=True "
            "(UpperEnvelope uses off_estime and on_corrige from DrugTimingFeatures)."
        )
    if data_dir_preview is not None:
        data_dir = skrub.var("data_dir", value=str(data_dir_preview))
    else:
        data_dir = skrub.var("data_dir")

    # Layer 1: load raw data from the source-bound directory.
    data = data_dir.skb.apply_func(load_raw)

    # Layer 2: mark X and y.
    # When patient_features=True, patient_id must remain in X so that
    # add_patient_features can group by it; DropCols removes it later.
    # When patient_features=False, we replicate the original behaviour:
    # patient_id flows into TableVectorizer unchanged.
    feature_frame = data.drop(columns=[TARGET_COL, *drop_cols])
    X = feature_frame.skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": data[GROUP_COL]},
    )
    y = data[TARGET_COL].skb.mark_as_y()

    # Layer 3: build the sklearn pipeline steps.
    steps = []
    if patient_features:
        from parkinson.features import add_patient_features  # lazy import

        steps.append(FunctionTransformer(add_patient_features))

    if drug_timing:
        from parkinson.pharmaco import DrugTimingFeatures  # lazy import

        steps.append(DrugTimingFeatures())

    if envelope:
        from parkinson.envelope import UpperEnvelope  # lazy import

        steps.append(UpperEnvelope())

    if patient_features or drug_timing or envelope:
        # Drop patient_id once all steps that need it have run.
        steps.append(skrub.DropCols(["patient_id"]))

    steps += [
        skrub.TableVectorizer(),
        HistGradientBoostingRegressor(random_state=0, **(hgb_params or {})),
    ]

    model = make_pipeline(*steps)

    if smooth:
        from parkinson.smoothing import PatientSmoothedRegressor  # lazy import

        model = PatientSmoothedRegressor(model)

    predictions = X.skb.apply(model, y=y)
    return predictions.skb.make_learner()
