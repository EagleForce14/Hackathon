# 03_patient_features

## Question / hypothesis

Si l'on donne au modèle ce que l'on sait du patient sur l'ensemble de
ses visites (niveau moyen de OFF/ON, tendance du OFF dans le temps,
durée de maladie) et qu'on retire `patient_id` des features, la
prédiction s'améliore-t-elle nettement par rapport à la baseline ?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - EDA `data/eda_correlations.html` : **79 % de la variance de la
    target est entre patients**. `off_moy_patient` a une corrélation de
    0,84 avec la target et reste disponible quand la visite n'a pas de
    OFF (42 % des visites). La target progresse d'environ 2,5 points par
    an et par patient, et `duree_maladie` a une corrélation de 0,55.
  - 01_baseline : `patient_id` est encodé comme feature par
    `TableVectorizer` alors que les patients du test sont disjoints.
    Le retirer seul donne R² 0,791 → 0,797.
  - Prototype (scratchpad, HGBR, `GroupKFold(5)` sur `patient_id`) :

    | variante | R² | RMSE |
    |---|---|---|
    | 01 baseline | 0,791 | 7,54 |
    | 03a : sans `patient_id` + moyennes patient | 0,919 | 4,70 |
    | 03b : 03a + tendance OFF par patient | **0,928** | **4,43** |

  - 02_no_off_feature : `off` est présent dans `X_test` dans les mêmes
    proportions (41 % manquant contre 42 % dans le train). Ce n'est pas
    une fuite, on le garde.
- **Why this matters:** c'est la piste qui rapporte le plus (RMSE
  −40 %). Les expériences 04 et suivantes se construisent dessus.

## Method

- **Files touched:**
  - `src/parkinson/features.py` (nouveau) : fonctions pures de features patient
  - `src/parkinson/pipeline.py` : option `patient_features` dans `build_learner`
  - `experiments/03_patient_features.py` (nouveau)
  - `tests/test_features.py` (nouveau), `tests/smoke/test_03_patient_features.py` (nouveau)
- **Change versus 01_baseline:** deux changements, testés ensemble
  (03b du prototype) :
  1. `patient_id` n'est plus une feature. Il reste dans X jusqu'à
     l'intérieur du pipeline (pour les agrégats patient et pour la 04),
     puis est retiré avec `skrub.DropCols(["patient_id"])` avant
     `TableVectorizer`. Il sert toujours à `groups=` du `GroupKFold`.
  2. Ajout de features calculées **par patient à partir des colonnes
     de X uniquement** (jamais la target), en tête du pipeline.
  Même `HistGradientBoostingRegressor(random_state=0)`, même
  `GroupKFold(5)`.
- **Cross-validation:** `GroupKFold(n_splits=5)` sur `patient_id`,
  identique à 01 et 02.
- **Out of scope for this experiment:** correction de `on` par le
  délai de prise (→ 04), réglage des hyperparamètres, autres modèles,
  imputation.

### Plan d'implémentation (pas à pas)

**Étape 0 : branche.**
Créer `exp/03-patient` depuis `main`. **Cette expérience passe en
premier** : la 04 (et la personne qui s'en occupe) part de cette
branche. Fusionner dès que les tests sont verts.

**Étape 1 : `src/parkinson/features.py`.**
Deux fonctions pures (sans état, pas de `fit`) :

- `patient_trend(df, col, by="patient_id", x="age") -> pd.Series`
  - Pour chaque patient, ajuster une droite `col ~ age`
    (`np.polyfit` degré 1) sur les lignes où `col` n'est pas manquant.
  - Renvoyer la valeur de la droite à l'`age` de **chaque** visite du
    patient, y compris celles où `col` est manquant.
  - NaN s'il y a moins de 2 points ou un seul âge distinct.
  - Renvoyer une Series **alignée sur l'index de `df`**.
  - **Cette fonction sera réutilisée par la 04** (sur `off_estime`) :
    la garder générique (paramètre `col`).
- `add_patient_features(df) -> pd.DataFrame` : renvoie une **copie**
  de `df` avec les colonnes ajoutées, `patient_id` **conservé** :

  | Colonne | Calcul |
  |---|---|
  | `duree_maladie` | `age - age_at_diagnosis` |
  | `off_moy_patient` | moyenne de `off` par patient (`groupby.transform("mean")`) |
  | `on_moy_patient` | moyenne de `on` par patient |
  | `ledd_moy_patient` | moyenne de `ledd` par patient |
  | `n_off_patient` | nombre de `off` non manquants chez le patient |
  | `n_visites` | nombre de visites du patient |
  | `age_depuis_1re_visite` | `age - min(age)` du patient |
  | `off_tendance_patient` | `patient_trend(df, "off")` |
  | `pente_off_patient` | pente de cette droite (points/an), même règle de NaN |

**Étape 2 : `src/parkinson/pipeline.py`.**
Ajouter `patient_features: bool = False` à `build_learner`.

- Retirer `patient_id` du drop avant `mark_as_X` s'il y en a un :
  X doit contenir `patient_id`.
- Construire le pipeline sous la forme :

```python
from sklearn.preprocessing import FunctionTransformer
from parkinson.features import add_patient_features

steps = []
if patient_features:
    steps.append(FunctionTransformer(add_patient_features))
steps += [skrub.DropCols(["patient_id"]), skrub.TableVectorizer(),
          HistGradientBoostingRegressor(random_state=0)]
X.skb.apply(make_pipeline(*steps), y=y)
```

- La 04 insérera ensuite son transformer juste après celui-ci, avant
  `DropCols`.
- Note : avec `patient_features=False`, le `DropCols` change déjà le
  comportement de 01 (R² 0,797 au lieu de 0,791). Relancer 01 pour
  garder une référence à jour, ou ajouter `DropCols` seulement quand
  `patient_features=True`, au choix. Le noter dans le Status.

**Étape 3 : tests.**
- `tests/test_features.py` (petit DataFrame fabriqué à la main, 2–3 patients) :
  - `add_patient_features` garde le même nombre de lignes, le même
    index et la colonne `patient_id` ;
  - `off_moy_patient` vaut la même chose sur toutes les visites d'un
    patient, et elle est renseignée même sur une visite sans `off` ;
  - `patient_trend` sur un patient avec `off` = 10, 20 aux âges 50, 51
    donne 30 à l'âge 52 ;
  - un patient avec un seul `off` donne `off_tendance_patient = NaN` ;
  - **aucune dépendance à la target** : la fonction marche sur un
    frame sans colonne `target`.
- `tests/smoke/test_03_patient_features.py` : copier le modèle de
  `tests/smoke/test_02_no_off_feature.py` avec `patient_features=True`.

**Étape 4 : `experiments/03_patient_features.py`.**
Copier `experiments/01_baseline.py`, appeler
`build_learner(data_dir_preview=DATA_DIR, patient_features=True)`,
projet skore `parkinson-patient-features`, puis
`project.put("03_patient_features", report)`.

**Étape 5 : exécuter, lire, noter.**
- Lancer le script et relever R² et RMSE (moyenne ± écart-type sur les 5 folds).
- Comparer à 01 (avec la même CV). Mesure du prototype : R² 0,928,
  RMSE 4,43 ± 0,07. Un écart de plus de ≈ 0,1 signale une différence
  d'implémentation à rechercher.
- Remplir le bloc Status ci-dessous et ajouter la ligne 03 dans
  `JOURNAL.md`.
- Prévenir la personne de la 04 que la branche est fusionnée.

**Définition de « terminé » :** tests verts, script exécuté, rapport
sur Skore Hub, Status rempli, branche fusionnée dans `main`.

## Risks / things that could invalidate the result

- **Pas de fuite de target :** les agrégats n'utilisent que `off`,
  `on`, `ledd`, `age`. Si quelqu'un ajoute un jour une moyenne de la
  target par patient, le score CV deviendra faux. À refuser en revue.
- **Pas de fuite entre folds :** avec `GroupKFold`, un patient est
  entièrement d'un côté. Les agrégats d'un patient de validation ne
  voient donc que ses propres visites, exactement comme au test.
- **Usage des visites futures :** la moyenne et la tendance utilisent
  toutes les visites du patient, y compris les plus tardives. C'est
  légitime ici, car `X_test` contient toutes les visites de ses
  patients (≈ 7,9 par patient, comme le train). Ce ne serait pas
  valable en prédiction « en temps réel ».
- **Prédire sur tout `X_test` d'un coup :** si les visites d'un
  patient sont prédites séparément, les agrégats sont faux.
- **Patients avec peu de OFF :** `off_tendance_patient` peut être
  instable avec 2 points très proches. `n_off_patient` permet au
  modèle d'en tenir compte. À surveiller dans le rapport skore
  (erreurs par quantile de `n_off_patient` si besoin).

## Status

- **State:** planned
- **Approved by user on:** n/a
- **Headline result:** n/a
- **Implication for next iteration:** n/a
