# %% [markdown]
# # Audit — 02_no_off_feature: HistGBR sans les colonnes off / time_since_intake_off
#
# Read-only review of the stored report: checks and metrics.

# %%
import skore

from parkinson.hub import load_skore_credentials

# %% [markdown]
# ## Open the project

# %%
cfg = load_skore_credentials()
skore.login(mode="hub")
project = skore.Project(
    name="parkinson-no-off",
    mode="hub",
    workspace=cfg["workspace"],
)
project

# %% [markdown]
# ## List the available reports

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL from put(): `.../cross-validations/42101`
# → id: `skore:report:cross-validation:42101`

# %%
REPORT_ID = "skore:report:cross-validation:42101"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
