# %% [markdown]
# # Experiment 02 — No OFF Feature
#
# **Date:** 2025-07-14
# **Goal:** Measure the true performance floor of the model without the
# ``off`` and ``time_since_intake_off`` columns. These are the near-leakage
# features identified in 01_baseline (Pearson 0.89 with target, 42% missing).
# Same model and GroupKFold as baseline — results are directly comparable.
# **Result:** fill in after the run.

# %%
import skore
from skore import login

from parkinson import PROJECT_ROOT
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
    name="parkinson-no-off",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data and learner
#
# ``drop_cols`` removes ``off`` and ``time_since_intake_off`` before the
# X marker. Everything else (TableVectorizer, HGBR, GroupKFold) is unchanged.

# %%
learner = build_learner(
    data_dir_preview=DATA_DIR,
    drop_cols=("off", "time_since_intake_off"),
)

# %% [markdown]
# ## Evaluate
#
# GroupKFold(n_splits=5) on patient_id — wired at the X marker in pipeline.py.

# %%
report = skore.evaluate(
    learner,
    data={"data_dir": str(DATA_DIR)},
)
report

# %% [markdown]
# ## Persist

# %%
project.put("02_no_off_feature", report)
