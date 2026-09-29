# %% [markdown]
# # Experiment 01 — Baseline
#
# **Date:** 2025-07-14
# **Goal:** Establish a reference score using HistGradientBoostingRegressor
# with GroupKFold(patient_id, n_splits=5). No imputation, no feature
# engineering — the model handles NaN natively.
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
    name="parkinson-baseline",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data and learner

# %%
X, y = load_dataset()
learner = build_learner(data_dir_preview=DATA_DIR)

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
project.put("01_baseline", report)
