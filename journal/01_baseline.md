# 01_baseline

## Question / hypothesis

Un `HistGradientBoostingRegressor` avec `GroupKFold` sur `patient_id`
peut-il établir un score de référence solide pour prédire le score
MDS-UPDRS OFF vrai, sans aucun pré-traitement des données manquantes ?

## Motivation

- **Sourcing strategy:** bootstrap (baseline forcé)
- **Source(s):**
  - EDA `data/eda.md` : 44 590 lignes, missingness élevée (ledd 37%, off 42%, time_since_intake_off 79%), patients répétés → nécessite GroupKFold.
- **Why this matters:** Établir un floor de performance mesurable avant toute ingénierie de features. `HistGradientBoosting` gère nativement les NaN — zéro imputation requise pour démarrer.

## Method

- **Files touched:** `src/parkinson/pipeline.py`, `src/parkinson/evaluate.py`, `experiments/01_baseline.py`
- **Change versus baseline:** Premier modèle — `HistGradientBoostingRegressor` (sklearn), `TableVectorizer` de skrub pour encoder les colonnes catégorielles (`gene`, `cohort`). `GroupKFold(n_splits=5)` sur `patient_id`.
- **Cross-validation:** `GroupKFold(n_splits=5)` sur `patient_id` — les patients du test sont disjoint du train, la CV doit refléter ça.
- **Out of scope for this experiment:** Feature engineering (disease_duration), gestion explicite du leakage de `off`, target transform.

## Risks / things that could invalidate the result

- `off` (Pearson 0.89 avec la cible) est inclus comme feature. Si `off` est absent à l'inférence Kaggle, le score CV sera trop optimiste.
- `GroupKFold` non stratifié : la distribution du target peut varier entre groupes.

## Status

- **State:** approved
- **Approved by user on:** 2025-07-14
- **Headline result:** n/a
- **Implication for next iteration:** n/a
