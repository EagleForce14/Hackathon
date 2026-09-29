# 04_drug_timing

## Question / hypothesis

Si l'on corrige le score `on` selon le délai depuis la prise
(`time_since_intake_on`) pour le ramener à l'échelle du OFF, peut-on
estimer le OFF des visites où il n'est pas mesuré (42 % des visites
n'ont que ON) et améliorer la prédiction au-delà de 03 ?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - EDA `data/eda_correlations.html` § 4 : le ratio `on / target`
    dépend fortement du délai (≈ 0,73 à < 0,5 h, plateau ≈ 0,45
    après 1,5 h), soit la courbe d'absorption de la lévodopa décrite
    dans `docs/CONTEXT.md`.
  - Ratio `on / off` sur les visites qui ont les deux (sans target) :

    | délai (h) | 0–0,25 | 0,25–0,5 | 0,5–0,75 | 0,75–1 | 1–1,25 | 1,25–1,5 | ≥ 1,5 |
    |---|---|---|---|---|---|---|---|
    | médiane on/off | 0,97 | 0,88 | 0,80 | 0,73 | 0,64 | 0,59 | ≈ 0,56 |

  - Prototype (scratchpad, HGBR, `GroupKFold(5)` sur `patient_id`) :
    03b R² 0,928 / RMSE 4,43, puis **03b + 04 R² 0,938 / RMSE 4,10**.
    `off_estime_tendance` a une corrélation de 0,934 avec la target, la
    plus forte des features testées.
- **Why this matters:** les visites « ON seul » sont les plus
  nombreuses. Sans correction, le modèle doit deviner le OFF à partir
  d'un ON dont l'échelle dépend de l'heure de l'examen.

## Method

- **Files touched:**
  - `src/parkinson/pharmaco.py` (nouveau) : transformer `DrugTimingFeatures`
  - `src/parkinson/pipeline.py` : option `drug_timing` dans `build_learner`
  - `experiments/04_drug_timing.py` (nouveau)
  - `tests/test_pharmaco.py` (nouveau), `tests/smoke/test_04_drug_timing.py` (nouveau)
- **Change versus 03_patient_features:** on ajoute un transformer
  sklearn **stateful** en tête du pipeline, avant `TableVectorizer`.
  Son `fit` apprend la courbe `on/off` par tranche de délai
  **sur le fold d'entraînement uniquement**. Son `transform` crée les
  features ci-dessous. Même modèle (HGBR `random_state=0`), même
  `GroupKFold(5)` sur `patient_id`.
- **Cross-validation:** `GroupKFold(n_splits=5)` sur `patient_id`,
  identique à 01–03.
- **Out of scope for this experiment:** réglage des hyperparamètres,
  courbe paramétrique (Bateman, etc.), changement de target
  (prédire `target - off`), correction de `off` par
  `time_since_intake_off` (l'EDA montre un effet quasi plat).

### Plan d'implémentation (pas à pas)

**Étape 0 : se synchroniser avec la 03.**
Partir de la branche de la 03 (`exp/03-patient`) et créer
`exp/04-drug-timing`. **À convenir avec la personne de la 03 :**
`patient_id` doit rester dans X jusqu'à l'intérieur du pipeline, car
le transformer 04 en a besoin pour les agrégats patient. Il est
retiré juste après, avec `skrub.DropCols(["patient_id"])`, avant
`TableVectorizer`. Si la 03 le retire avant `mark_as_X`, déplacer ce
retrait dans le pipeline.

**Étape 1 : `src/parkinson/pharmaco.py`.**
Classe `DrugTimingFeatures(TransformerMixin, BaseEstimator)`.

- Paramètres `__init__` : `bins=(0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 7)`,
  `plateau_from=1.5` (heures), `min_count=30`.
- `fit(X, y=None)` : **n'utilise jamais `y`.**
  1. Garder les lignes avec `on`, `off > 0` et `time_since_intake_on` non manquants.
  2. `ratio = on / off`, découper `time_since_intake_on` avec `pd.cut(bins)`,
     prendre la médiane par tranche. Si une tranche a moins de `min_count`
     lignes, lui donner la valeur du plateau.
  3. Stocker `self.curve_` (médianes par tranche) et `self.plateau_`
     (médiane des ratios pour un délai ≥ `plateau_from`).
- `transform(X)` : renvoie une **copie** de X avec les colonnes ajoutées :

  | Colonne | Calcul |
  |---|---|
  | `ratio_attendu` | valeur de `curve_` pour le délai de la visite, `plateau_` si le délai est manquant ou hors bornes |
  | `on_corrige` | `on / ratio_attendu` |
  | `off_estime` | `off` s'il est mesuré, sinon `on_corrige` |
  | `traite` | `1` si `ledd` ou `on` est non manquant |
  | `on_manquant`, `off_manquant` | indicateurs 0/1 |
  | `off_estime_moy_patient` | moyenne de `off_estime` par `patient_id` |
  | `off_estime_tendance` | par patient, droite `off_estime ~ age` (`np.polyfit` degré 1) sur les lignes non manquantes, évaluée à l'`age` de chaque visite. NaN s'il y a moins de 2 points ou un seul âge distinct |
  | `part_traite_patient` | moyenne de `traite` par patient |

- Pour la tendance, réutiliser `patient_trend(df, "off_estime")` de
  `src/parkinson/features.py` (créé par la 03). Ne pas le dupliquer.
- Garder l'ordre et l'index des lignes (`groupby(...).transform` ou
  réindexation explicite après `apply`).

**Étape 2 : `src/parkinson/pipeline.py`.**
Ajouter `drug_timing: bool = False` à `build_learner`. Construire le
pipeline sous la forme :

```python
steps = []
if patient_features:                      # ajouté par la 03
    steps.append(FunctionTransformer(add_patient_features))
if drug_timing:                           # ajouté par la 04
    steps.append(DrugTimingFeatures())
steps += [skrub.DropCols(["patient_id"]), skrub.TableVectorizer(),
          HistGradientBoostingRegressor(random_state=0)]
X.skb.apply(make_pipeline(*steps), y=y)
```

L'expérience 04 appelle `build_learner(..., patient_features=True, drug_timing=True)`.

La valeur par défaut `False` doit reproduire exactement la 03, pour
ne pas casser les expériences précédentes.

**Étape 3 : tests.**
- `tests/test_pharmaco.py` (tests unitaires, sur un petit DataFrame fabriqué à la main) :
  - `fit` fonctionne avec `y=None` ;
  - la courbe est globalement décroissante (1ʳᵉ tranche > plateau) ;
  - `transform` garde le même nombre de lignes et le même index ;
  - une ligne sans `on` donne `on_corrige = NaN` et `off_estime = off` ;
  - un délai manquant donne `ratio_attendu = plateau_` ;
  - un patient avec un seul `off_estime` donne `off_estime_tendance = NaN`.
- `tests/smoke/test_04_drug_timing.py` : copier le modèle de
  `tests/smoke/test_02_no_off_feature.py` avec `drug_timing=True`.

**Étape 4 : `experiments/04_drug_timing.py`.**
Copier `experiments/03_patient_features.py`, appeler
`build_learner(..., drug_timing=True)`, projet skore
`parkinson-drug-timing`, puis `project.put("04_drug_timing", report)`.

**Étape 5 : exécuter, lire, noter.**
- Lancer le script et relever R² et RMSE (moyenne ± écart-type sur les 5 folds).
- **Contrôle d'ablation** (dans `scratch/`, pas une nouvelle
  expérience) : retirer `off_estime_tendance` et regarder combien on
  perd, pour savoir si le gain vient surtout de cette feature.
- Remplir le bloc Status ci-dessous et la ligne 04 de `JOURNAL.md`.

**Définition de « terminé » :** tests verts, script exécuté, rapport
sur Skore Hub, Status rempli, résultat comparé à la 03 avec la même
CV.

## Risks / things that could invalidate the result

- **Fuite de fold :** si la courbe est calculée sur tout `X_train`
  avant la CV au lieu du `fit`, la validation voit une courbe
  apprise en partie sur ses propres lignes. Il n'y a pas de target en
  jeu, donc l'effet est faible, mais il faut garder le calcul dans `fit`.
- **Agrégats patient au moment de prédire :** `transform` doit
  recevoir **toutes les visites d'un patient ensemble** (tout
  `X_test` d'un coup). Si on prédit ligne par ligne, les moyennes et
  tendances deviennent fausses.
- **Référence de la courbe :** elle est apprise sur `on/off`
  (OFF mesuré, biaisé d'environ −8 points chez les traités), pas sur
  `on/target`. `on_corrige` est donc à l'échelle du OFF *mesuré*.
  C'est voulu (pas de target), mais le modèle doit encore apprendre
  le décalage vers le vrai OFF.
- **Doublon avec la 03 :** `off_estime_moy_patient` recouvre en
  partie `off_moy_patient`. Ce n'est pas grave pour HGBR, mais
  l'ablation dira ce qui apporte vraiment.
- Le gain du prototype (+0,010 de R²) est petit : vérifier qu'il
  dépasse l'écart-type entre folds (≈ 0,003).

## Status

- **State:** planned
- **Approved by user on:** n/a
- **Headline result:** n/a
- **Implication for next iteration:** n/a
