# %% [markdown]
# # EDA: Parkinson's Motor Score Prediction
#
# Exploratory data analysis of the synthetic multi-cohort Parkinson's dataset,
# run before designing a model.
#
# - **Raw data** is read-only — loaded from `data/*.csv`. This file never
#   cleans or modifies the raw data.
# - **Outputs** go under `EDA_DIR` (`data/`): an `eda_<table>.html` report
#   per table, summarized in `eda.md`.
#
# Executed via:
#   python .bob/skills/audit-ml-pipeline/scripts/run_cells.py data/eda.py

# %%
import json
from pathlib import Path

import pandas as pd
import skrub

# EDA outputs always land here; the raw data is read from the same folder.
# The runner executes cells from the project root, so cwd() is the project root.
EDA_DIR = Path.cwd() / "data"
EDA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load the raw data
#
# X_train and y_train are joined on the Index column so that the target is
# visible during EDA. X_test is loaded separately for a missingness check.

# %%
X_train = pd.read_csv(EDA_DIR / "X_train.csv", index_col="Index")
y_train = pd.read_csv(EDA_DIR / "y_train.csv", index_col="Index")
X_test = pd.read_csv(EDA_DIR / "X_test.csv", index_col="Index")

# Join target into training frame for EDA
train = X_train.join(y_train)
train.shape

# %% [markdown]
# ## Table overview — training set
#
# Per-table report saved to `data/eda_train.html`, plus a compact per-column
# summary: dtype, fraction missing, and number of unique values.

# %%
report_train = skrub.TableReport(train, title="Parkinson train", verbose=0)
report_train.write_html(EDA_DIR / "eda_train.html")

summary = json.loads(report_train.json())
n_rows = summary.get("n_rows")
overview = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("nunique"),
    }
    for col in summary.get("columns", [])
]
{"n_rows": n_rows, "n_columns": len(overview), "columns": overview}

# %% [markdown]
# ## Table overview — test set
#
# Quick missingness check on the held-out set (different patients).

# %%
report_test = skrub.TableReport(X_test, title="Parkinson test", verbose=0)
report_test.write_html(EDA_DIR / "eda_test.html")

summary_test = json.loads(report_test.json())
n_rows_test = summary_test.get("n_rows")
overview_test = [
    {
        "column": col.get("name"),
        "dtype": col.get("dtype"),
        "null_pct": col.get("null_proportion"),
        "n_unique": col.get("nunique"),
    }
    for col in summary_test.get("columns", [])
]
{"n_rows_test": n_rows_test, "columns_test": overview_test}

# %% [markdown]
# ## Target
#
# Distribution of the true-OFF MDS-UPDRS score (regression target).
# Spread and skew inform the metric choice and whether a target transform is needed.

# %%
TARGET = "target"
next((col for col in summary.get("columns", []) if col.get("name") == TARGET), None)

# %% [markdown]
# ## Structure signals
#
# Datetime columns and high unique-ratio id / group-like columns.
# `patient_id` is expected to repeat within each patient's visits.

# %%
datetime_cols = [
    col.get("name")
    for col in summary.get("columns", [])
    if "date" in str(col.get("dtype", "")).lower()
]
unique_ratio = sorted(
    (
        {
            "column": col.get("name"),
            "unique_ratio": (col.get("nunique") or 0) / n_rows if n_rows else None,
        }
        for col in summary.get("columns", [])
    ),
    key=lambda r: (r["unique_ratio"] is not None, r["unique_ratio"]),
    reverse=True,
)
{"datetime_cols": datetime_cols, "top_unique_ratio": unique_ratio[:10]}

# %% [markdown]
# ## Associations
#
# Strongest pairwise associations in the training set. A near-perfect
# feature↔target link may indicate leakage.

# %%
skrub.column_associations(train).head(20)

# %% [markdown]
# ## Summary
#
# Findings and their modelling implications are written up in `data/eda.md`.
