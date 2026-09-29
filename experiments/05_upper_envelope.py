# %% [markdown]
# # Experiment 05 — Upper Envelope + Tuned HGBR + Patient Smoothing
#
# **Date:** 2025-07-14
# **Three sub-experiments, measured one by one:**
# - **05a** : envelope features only (off, off_estime, on_corrige upper envelopes)
# - **05b** : 05a + tuned HGBR (max_iter=1000, lr=0.03, l2=1.0)
# - **05c** : 05b + per-patient parabola smoothing of predictions
#
# **Hypothesis:** 59% of residual error² from 03+04 is a patient-level offset
# caused by the asymmetric downward noise on OFF measurements.  Upper envelopes
# correct this, HGBR tuning squeezes more capacity, and polynomial smoothing
# exploits the known smooth trajectory.
# **Expected:** RMSE ≈ 3.71 (05a) → 3.58 (05b) → 3.52 (05c).
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
    name="parkinson-upper-envelope",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data

# %%
X, y = load_dataset()

HGB_TUNED = dict(
    max_iter=1000,
    learning_rate=0.03,
    early_stopping=False,
    l2_regularization=1.0,
)

# %% [markdown]
# ## 05a — Envelope features only

# %%
learner_05a = build_learner(
    data_dir_preview=DATA_DIR,
    patient_features=True,
    drug_timing=True,
    envelope=True,
)
report_05a = skore.evaluate(learner_05a, data={"data_dir": str(DATA_DIR)})
report_05a

# %%
project.put("05a_envelope", report_05a)

# %% [markdown]
# ## 05b — Envelope + tuned HGBR

# %%
learner_05b = build_learner(
    data_dir_preview=DATA_DIR,
    patient_features=True,
    drug_timing=True,
    envelope=True,
    hgb_params=HGB_TUNED,
)
report_05b = skore.evaluate(learner_05b, data={"data_dir": str(DATA_DIR)})
report_05b

# %%
project.put("05b_tuned", report_05b)

# %% [markdown]
# ## 05c — Envelope + tuned HGBR + patient smoothing

# %%
learner_05c = build_learner(
    data_dir_preview=DATA_DIR,
    patient_features=True,
    drug_timing=True,
    envelope=True,
    hgb_params=HGB_TUNED,
    smooth=True,
)
report_05c = skore.evaluate(learner_05c, data={"data_dir": str(DATA_DIR)})
report_05c

# %%
project.put("05c_smoothed", report_05c)
