# JOURNAL — Parkinson's Motor Score Prediction

## Status

- **Goal:** Predict the unbiased true OFF MDS-UPDRS motor score for each patient visit.
- **Task type:** Regression
- **Evaluation metric:** TBD (RMSE or MAE; competition uses a leaderboard score)
- **Workspace decisions:**
  - Tabular library: pandas
  - Environment manager: .venv (pip)
  - student prior: some-sklearn - recorded: 2025-07-14

## Data understanding (EDA)

- **Status:** done - 2025-07-14
- **Summary:** 44,590 training rows × 11 features + 1 target; 11,013 test rows (holdout by patient). Target (`true OFF MDS-UPDRS`) is near-normally distributed (mean 37.5, std 16.5). Strongest predictors are `off` (Pearson 0.886, but 42% missing) and `on` (0.688). `patient_id` repeats across visits — `GroupKFold` on `patient_id` is required to match the test-by-patient evaluation setup. `off` feature is a near-leakage risk.
- **Report:** [data/eda.md](../data/eda.md)

## History

| # | Stem | Intent | Status | Headline | Design note |
|---|---|---|---|---|---|
| 01 | baseline | HistGBR + GroupKFold(5) sur patient_id, toutes features | approved | n/a | [01_baseline.md](01_baseline.md) |
| 02 | no_off_feature | Même modèle sans `off` / `time_since_intake_off` — vrai plancher sans near-leakage | done | R² 0.648 ± 0.008 · RMSE 9.79 ± 0.26 · MAE 7.63 ± 0.20 | [02_no_off_feature.md](02_no_off_feature.md) |

## Backlog

| # | Item | Source |
|---|---|---|
