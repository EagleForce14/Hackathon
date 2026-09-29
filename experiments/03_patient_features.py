# %% [markdown]
# # Experiment 03 — Patient Features
#
# **Date:** 2025-07-14
# **Goal:** Add per-patient aggregate features (off_moy_patient, on_moy_patient,
# ledd_moy_patient, n_off_patient, n_visites, age_depuis_1re_visite,
# off_tendance_patient, pente_off_patient, duree_maladie) and remove patient_id
# from the feature set. GroupKFold(5) on patient_id, same HGBR as 01/02.
# **Hypothesis:** These features capture ~79% of between-patient variance.
# Expected: R² ≈ 0.93 / RMSE ≈ 4.4 (RMSE −40% vs baseline).
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
    name="parkinson-patient-features",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data and learner

# %%
X, y = load_dataset()
learner = build_learner(data_dir_preview=DATA_DIR, patient_features=True)

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
project.put("03_patient_features", report)
