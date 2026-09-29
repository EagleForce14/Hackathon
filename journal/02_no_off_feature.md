# 02_no_off_feature

## Question / hypothesis

Sans la colonne `off` (et ses dérivées `time_since_intake_off`), quelle est
la vraie performance plancher du modèle — celle qui reflète le cadre
d'inférence Kaggle où `off` peut être absent ?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - Expérience 01_baseline : R² = 0.790 ± 0.003. La colonne `off` est
    corrélée à 0.89 avec la cible (EDA) et 42 % manquante. Si elle n'est
    pas disponible à l'inférence Kaggle, le score baseline est trompeur.
- **Why this matters:** Connaître le plancher *sans* `off` est nécessaire
  avant tout feature engineering : si le delta est grand, reconstruire
  un proxy de `off` (agrégat patient, historique) devient la priorité.
  Si le delta est petit, `off` n'était pas le seul signal et d'autres
  features suffisent.

## Method

- **Files touched:** `src/parkinson/pipeline.py`
- **Change versus 01_baseline:** Supprimer `off` et `time_since_intake_off`
  du frame avant le mark_as_X. Tout le reste est identique : même
  `TableVectorizer` + `HistGradientBoostingRegressor(random_state=0)`,
  même `GroupKFold(n_splits=5)` sur `patient_id`.
- **Cross-validation:** `GroupKFold(n_splits=5)` sur `patient_id` —
  identique à la baseline, pour que les scores soient directement
  comparables.
- **Out of scope for this experiment:** Feature engineering
  (disease_duration, agrégats patient), imputation de `ledd`, tuning
  des hyperparamètres.

## Risks / things that could invalidate the result

- `on` (Pearson 0.688 avec la cible, EDA) reste inclus comme feature.
  Si `on` est aussi absent à l'inférence, ce score sera encore optimiste.
- `GroupKFold` non stratifié : distribution du target peut varier entre
  groupes — identique au risque de la baseline, ne biaise pas la
  comparaison.

## Status

- **State:** done
- **Approved by user on:** 2025-07-14
- **Headline result:** R² 0.648 ± 0.008 | RMSE 9.79 ± 0.26 | MAE 7.63 ± 0.20 (GroupKFold 5-fold)
- **Implication for next iteration:** Retirer `off` fait chuter le R² de 0.79 → 0.65 et le MAE de 5.9 → 7.6 (+1.7 pts). `off` portait ~14 pts de R² à lui seul. Pour l'exp 03, on repart de la baseline (avec `off`) et on ajoute du feature engineering (disease_duration, agrégats patient) pour gratter de la performance proprement.
