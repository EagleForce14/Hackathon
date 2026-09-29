"""Cross-validation strategy for the Parkinson's motor score experiments.

The test set is held out by patient (``patient_id``), so the cross-validator
must keep each patient's visits entirely within either the training or the
validation fold. ``GroupKFold`` achieves this: it groups rows by
``patient_id`` so no patient's data appears in both folds simultaneously.

This module only declares the splitter; running the evaluation and
persisting the report happen in the experiment scripts (``experiments/``).
"""

from __future__ import annotations

from sklearn.model_selection import GroupKFold

# GroupKFold on patient_id — matches the test-by-patient evaluation setup.
# split_kwargs={"groups": ...} wired at the X marker in pipeline.py feeds
# the groups array to GroupKFold.split(X, y, groups=...) automatically.
splitter = GroupKFold(n_splits=5)
