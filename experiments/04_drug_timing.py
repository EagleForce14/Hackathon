# %% [markdown]
# # Experiment 04 — Drug Timing Correction
#
# **Date:** 2025-07-14
# **Goal:** Add a pharmacokinetic correction layer on top of experiment 03.
# DrugTimingFeatures learns the on/off ratio curve per delay bin from the
# training fold, then estimates OFF for visits where only ON was measured.
# Patient features (03) are still active. GroupKFold(5) on patient_id, same HGBR.
# **Hypothesis:** Correcting ON for levodopa delay narrows the gap between ON
# and OFF measurements and improves predictions on ON-only visits (42% of data).
# Expected: R² ≈ 0.938 / RMSE ≈ 4.10 (RMSE −7% vs 03).
# **Result:** fill in after the run.

# %%
import skore
from skore import login

from parkinson import PROJECT_ROOT
from parkinson.data import load_dataset
from parkinson.hub import load_skore_credentials
from parkinson.pipeline import build_learner

# %% [markdown]
# ## Paths

# %%
DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Project

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="parkinson-drug-timing",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data and learner

# %%
X, y = load_dataset()
learner = build_learner(
    data_dir_preview=DATA_DIR,
    patient_features=True,
    drug_timing=True,
)

# %% [markdown]
# ## Evaluate
#
# GroupKFold(n_splits=5) on patient_id — patients in the test set are
# disjoint from those in training. The split_kwargs wired at the X marker
# supply the groups array to the splitter automatically.

# %%
# No explicit splitter= : skore reuses the GroupKFold + split_kwargs
# (groups=patient_id) wired at the X marker in pipeline.py.
report = skore.evaluate(
    learner,
    data={"data_dir": str(DATA_DIR)},
)
report

# %% [markdown]
# ## Persist

# %%
project.put("04_drug_timing", report)
