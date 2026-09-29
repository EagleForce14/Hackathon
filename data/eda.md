<!--
Exploratory data analysis summary for this workspace, written from
the data/eda.py run. Ground every claim in what the run actually
showed - do not invent facts. Keep "Modelling implications" as
candidate suggestions to weigh when designing the model, not final
decisions.
-->

# EDA: Parkinson's Motor Score Prediction

_Generated from [`data/eda.py`](eda.py) on 2025-07-14._

## Dataset at a glance

- **Tables:** 2 (train + test, same column schema)
- **Shape:** 44,590 × 12 (train, with target) · 11,013 × 11 (test, no target)
- **Split:** holdout is by patient — `X_test` patients do not appear in `X_train`
- **Target:** `target` — true OFF MDS-UPDRS motor score (continuous regression, range 0–109.5)
- **Rich reports:** [eda_train.html](eda_train.html) · [eda_test.html](eda_test.html)

## Per-column findings

| Column | Dtype | Missing (train) | Missing (test) | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0% | 0% | Group ID — repeated across visits |
| `cohort` | string | 0% | 0% | Cohort label (e.g. A) |
| `sexM` | int | 0% | 0% | Binary (0/1) |
| `gene` | string | **32.4%** | 32.0% | Genetic variant; >30% missing |
| `age_at_diagnosis` | float | 5.2% | 4.6% | Minor missingness |
| `age` | float | 0% | 0% | Age at visit; complete |
| `ledd` | float | **36.6%** | 38.6% | Levodopa equivalent daily dose; heavy missing |
| `time_since_intake_on` | float | **46.4%** | 47.8% | Hours since last ON intake; nearly half missing |
| `time_since_intake_off` | float | **78.8%** | 79.1% | Hours since last OFF intake; majority missing |
| `on` | float | 29.6% | 31.2% | Observed ON motor score |
| `off` | float | **42.4%** | 40.8% | Observed OFF motor score |
| `target` | float | 0% | — | True OFF score (train only; target to predict) |

Key observations:
- `time_since_intake_off` is missing in ~79% of rows — it is effectively a sparse feature.
- `off` is missing in ~42% of training rows, yet strongly correlated with the target. This makes it a valuable but unreliable feature.
- `age_at_diagnosis` and `age` are near-collinear (Pearson 0.94); one may be dropped or a derived `disease_duration = age - age_at_diagnosis` computed.
- No constant columns or obvious sentinel values detected.
- `n_unique` was not returned by `TableReport.json()` for most columns at skrub 0.10.1 (key `nunique` is `None`); unique counts below come from the target cell dict which carried full stats.

## Target

- **Task:** Regression
- **Mean:** 37.47 · **Std:** 16.50 · **Median:** 37.3
- **Range:** 0.0 – 109.5
- **IQR:** 25.6 – 49.3 (23.7 units)
- **Distribution:** Near-symmetric bell shape centred around 37, with a thin right tail reaching 109.5.
- **No missing values in train.**

The target is roughly normally distributed with mild right skew. No strong class imbalance / extreme skew issue. A target transform (log, quantile) is unlikely to be necessary but can be explored.

## Structure

- **No datetime columns detected** (skrub's dtype inference found no date dtypes). Temporal ordering is implicit in `age` (age at visit) and `time_since_intake_*`.
- **`patient_id` is a clear group-level ID**: each patient has multiple visits across the dataset. Because test patients are disjoint from train patients, rows for the same patient must not be split across train and validation folds — this is the key structural constraint.
- `cohort` is a low-cardinality group variable (the HTML report will show exact counts).
- There are no explicit timestamp columns, but `age` effectively orders visits for a given patient.

## Associations

Top pairwise associations (Cramér V / Pearson, from `skrub.column_associations`):

| Pair | Pearson | Cramér V | Interpretation |
|---|---|---|---|
| `age_at_diagnosis` ↔ `age` | **0.942** | 0.562 | Near-collinear; `age - age_at_diagnosis` gives disease duration |
| `off` ↔ `target` | **0.886** | 0.495 | Strong predictor, but 42% missing in train |
| `on` ↔ `off` | **0.867** | 0.345 | ON and OFF scores move together |
| `on` ↔ `target` | **0.688** | 0.403 | Useful predictor (less missing than `off`) |
| `ledd` ↔ `on` | 0.210 | 0.292 | Moderate: higher dose → better ON score |
| `age` ↔ `target` | 0.310 | 0.114 | Weak-moderate: age correlated with disease severity |

**Leakage flag:** `off` (Pearson 0.886 with `target`) is the unbiased observed OFF score, which is conceptually very close to `target` (the bias-corrected true OFF). Using `off` as a feature risks **information leakage** when it is available, as it approximates the ground truth. This must be explicitly considered in the modelling strategy.

## Modelling implications

- **`patient_id` repeats across rows** → use `GroupKFold` (or `GroupShuffleSplit`) on `patient_id` to prevent a patient's visits leaking between train and validation folds. Test-by-patient is the competition's evaluation setup.
- **No datetime column but temporal ordering via `age`** → if predicting future visits (forecasting), `TimeSeriesSplit` within each patient would be appropriate. For cross-sectional predictions, `GroupKFold` is sufficient.
- **High missingness on `off`, `ledd`, `time_since_intake_on/off`** → imputation (median/constant for numerics) or an estimator that handles NaNs natively (e.g. `HistGradientBoosting`) is needed.
- **`off` leakage risk** → consider two model variants: (a) using `off` as a feature (only valid when `off` is observed at inference), and (b) excluding `off` to generalise to visits where it is absent. The competition target is the true OFF, not the observed OFF.
- **`age_at_diagnosis` ↔ `age` collinearity** → either drop `age_at_diagnosis` or engineer `disease_duration`.
- **Target distribution is near-normal** → RMSE or MAE are natural metrics; no log-target transform required by default.
- **`gene`, `cohort`** are categorical with reasonable cardinality; skrub's `TableVectorizer` will handle them automatically.

## Open questions

1. **`off` feature at inference time:** Will `off` ever be observed at prediction time for the Kaggle test set? If not, it should be excluded from the feature set to avoid test-time leakage (the model would receive NaN for `off` in the test set, which is fine for imputer-equipped pipelines but misleads during CV if we train with `off` values present).
2. **Disease duration engineering:** Is `age - age_at_diagnosis` a better feature than both individual columns? Should `age_at_diagnosis` be dropped entirely?
3. **Cohort heterogeneity:** Are there systematic differences between cohorts A and B (e.g. different `ledd` distributions, different prevalence of `gene` variants)? The HTML report will surface this.
4. **Temporal modelling:** Should each patient's visit sequence be modelled as a time series (using lagged scores, cumulative LEDD, etc.), or is a cross-sectional model per visit sufficient for this benchmark?
