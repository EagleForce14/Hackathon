# 05_upper_envelope

## Question / hypothesis

Le OFF mesuré sous-estime presque toujours le vrai OFF (bruit
asymétrique). Si l'on résume les mesures d'un patient par leur
**enveloppe haute** plutôt que par leur moyenne, puis qu'on règle HGBR
et qu'on lisse les prédictions par patient, le RMSE passe-t-il sous
3,6 ?

## Motivation

- **Sourcing strategy:** my-pick (analyse de la génération des données)
- **Source(s):**
  - **Le vrai OFF est une courbe lisse par patient** : une parabole en
    fonction de l'âge laisse un résidu de 0,36 point en RMSE (droite :
    1,68). La progression ralentit avec le temps.
  - **Le bruit sur OFF est asymétrique** (quantiles de `off - target`) :

    | groupe | 5 % | 25 % | médiane | 75 % | 95 % |
    |---|---|---|---|---|---|
    | non traités | −16,0 | −7,2 | −2,1 | +0,2 | +3,3 |
    | traités | −25,1 | −13,9 | −7,0 | −0,7 | +4,2 |

    La mesure dépasse rarement la vérité, mais tombe souvent bien en
    dessous. En gros, `off ≈ target × (1 − f) + petit bruit`, avec `f`
    aléatoire (médiane ≈ 0,15), **y compris chez les non traités**.
    C'est le biais de notation humaine de `docs/CONTEXT.md`.
  - **Le biais change d'une visite à l'autre** (ICC ≈ 0,2) et ne dépend
    presque pas de `ledd` ni de `time_since_intake_off`. La cohorte B est
    un peu moins biaisée (médiane de `f` 0,09–0,12 contre 0,15–0,17).
  - **ON suit la même logique** : `on / target` ≈ 0,45 au plateau, avec
    une forte dispersion (ICC ≈ 0,27).
  - **Décomposition de l'erreur de 03+04** (RMSE 4,12) : 59 % de
    l'erreur² est un décalage global du patient, car la moyenne des
    mesures, tirée vers le bas, place toute sa courbe trop bas.
  - Prototype (scratchpad, `GroupKFold(5)` sur `patient_id`,
    prédictions hors fold) :

    | étape | RMSE |
    |---|---|
    | 03 + 04 | 4,12 |
    | + enveloppe haute simple (off, off_estime) | 3,85 |
    | + multi-quantiles, parabole haute, enveloppe `on_corrige` | 3,71 |
    | + HGBR réglé | 3,58 |
    | + lissage final par patient | **3,52** |

    Même ordre de grandeur avec `GroupKFold(5, shuffle=True, random_state=1)`.
- **Why this matters:** d'autres groupes annoncent environ 3,6. C'est
  la première piste qui s'attaque à la cause principale de l'erreur
  restante, au lieu d'ajouter des features à la marge.

## Method

- **Files touched:**
  - `src/parkinson/envelope.py` (nouveau) : `patient_envelope_features`, transformer `UpperEnvelope`
  - `src/parkinson/smoothing.py` (nouveau) : méta-estimateur `PatientSmoothedRegressor`
  - `src/parkinson/pipeline.py` : options `envelope`, `hgb_params`, `smooth`
  - `experiments/05_upper_envelope.py` (nouveau)
  - `tests/test_envelope.py`, `tests/test_smoothing.py`, `tests/smoke/test_05_upper_envelope.py` (nouveaux)
- **Change versus 04_drug_timing:** trois changements, **mesurés un par
  un** (05a → 05b → 05c) pour savoir d'où vient le gain :
  - **05a, enveloppe haute :** un transformer sans état, après la 04,
    qui résume les mesures de chaque patient par leur haut plutôt que
    par leur moyenne.
  - **05b, HGBR réglé :** `max_iter=1000`, `learning_rate=0.03`,
    `early_stopping=False`, `l2_regularization=1.0`.
  - **05c, lissage :** une parabole par patient sur les prédictions
    (âge en abscisse), intégrée comme méta-estimateur pour que skore
    l'évalue.
- **Cross-validation:** `GroupKFold(n_splits=5)` sur `patient_id`,
  identique à 01–04.
- **Out of scope for this experiment:** modèle paramétrique complet
  (maximum de vraisemblance du bruit asymétrique), autres algorithmes
  (LightGBM…), moyenne de plusieurs graines, traitement spécifique par
  cohorte.

### Plan d'implémentation (pas à pas)

**Étape 0 : branche.**
Créer `exp/05-envelope` depuis la branche 04 fusionnée (ou `main` après
fusion de 03 et 04). La 05 réutilise `off_estime` et `on_corrige`,
produits par la 04.

**Étape 1 : `src/parkinson/envelope.py`.**

`patient_envelope_features(df, col, quantiles=(0.5, 0.75, 0.9))` renvoie
un DataFrame aligné sur `df.index`. Pour chaque patient, sur les lignes
où `col` n'est pas manquant (âges `a`, valeurs `v`) :

| Colonne | Calcul | Condition, sinon NaN |
|---|---|---|
| `{col}_max_patient` | `max(v)` | ≥ 1 mesure |
| `{col}_q50`, `_q75`, `_q90` | droite `base = polyfit(a, v, 1)`, résidus `r = v - base(a)`. Valeur = `base(age) + quantile(r, q)` à l'âge de chaque visite | ≥ 2 mesures, ≥ 2 âges distincts |
| `{col}_enveloppe` | partir de `base`, puis 2 fois : garder les points dont le résidu est ≥ à la médiane des résidus et réajuster la droite dessus. Évaluer à l'âge de chaque visite | idem, et ≥ 2 points gardés |
| `{col}_env2` | même principe avec une **parabole** (degré 2) | ≥ 6 mesures, ≥ 3 âges distincts parmi les points gardés |

- Les valeurs sont calculées pour **toutes** les visites du patient,
  y compris celles où `col` manque. C'est justement l'intérêt.
- Transformer `UpperEnvelope(cols=("off", "off_estime", "on_corrige"))` :
  `fit` ne fait rien, `transform` ajoute les colonnes de
  `patient_envelope_features` pour chaque `col`. Il ne touche pas à
  `patient_id`.
- Pour les ajustements de droite, réutiliser les helpers de
  `features.py` (03) si c'est possible.

**Étape 2 : `src/parkinson/smoothing.py`.**

`PatientSmoothedRegressor(estimator, deg=2, weight=1.0, group_col="patient_id", x_col="age")` :
- `fit(X, y)` : clone et entraîne `estimator` sur X complet.
  L'estimateur interne retire lui-même `patient_id` via `DropCols`.
- `predict(X)` : `p = estimator.predict(X)`. Puis, pour chaque patient
  ayant plus de `deg + 1` âges distincts, `lisse = polyval(polyfit(age, p, deg), age)`,
  et renvoyer `weight * lisse + (1 - weight) * p`. Les autres patients gardent `p`.
- **Ne jamais utiliser `y` dans `predict`.** Garder l'ordre des lignes.

**Étape 3 : `src/parkinson/pipeline.py`.**
Ajouter `envelope: bool = False`, `hgb_params: dict | None = None` et
`smooth: bool = False` à `build_learner` :

```python
steps = []
if patient_features:
    steps.append(FunctionTransformer(add_patient_features))
if drug_timing:
    steps.append(DrugTimingFeatures())
if envelope:                                  # 05a
    steps.append(UpperEnvelope())
steps += [skrub.DropCols(["patient_id"]), skrub.TableVectorizer(),
          HistGradientBoostingRegressor(random_state=0, **(hgb_params or {}))]  # 05b
model = make_pipeline(*steps)
if smooth:                                    # 05c
    model = PatientSmoothedRegressor(model)
X.skb.apply(model, y=y)
```

`envelope=True` suppose `drug_timing=True` (colonnes `off_estime`,
`on_corrige`). Lever une erreur claire sinon.

**Étape 4 : tests.**
- `tests/test_envelope.py` (petit DataFrame fabriqué à la main) :
  - un patient avec `off` = 40, 25, 41, 30, 39 à des âges proches donne
    `off_q90` > `off_q50` > moyenne, et `off_enveloppe` ≈ 40 ;
  - le nombre de lignes et l'index sont conservés ;
  - une visite sans `off` reçoit quand même les valeurs du patient ;
  - un patient avec une seule mesure a `off_max_patient` renseigné et le reste à NaN ;
  - la fonction marche sans colonne `target`.
- `tests/test_smoothing.py` :
  - avec un estimateur factice qui renvoie une parabole bruitée, la
    sortie lissée est plus proche de la parabole ;
  - un patient avec 2 visites garde ses prédictions telles quelles ;
  - l'ordre des lignes est conservé.
- `tests/smoke/test_05_upper_envelope.py` : sur le modèle des smoke
  tests existants, avec toutes les options activées.

**Étape 5 : `experiments/05_upper_envelope.py`.**
Trois évaluations skore dans le même script, projet
`parkinson-upper-envelope` :

| clé | options de `build_learner` |
|---|---|
| `05a_envelope` | `patient_features=True, drug_timing=True, envelope=True` |
| `05b_tuned` | idem + `hgb_params=dict(max_iter=1000, learning_rate=0.03, early_stopping=False, l2_regularization=1.0)` |
| `05c_smoothed` | idem 05b + `smooth=True` |

**Étape 6 : exécuter, lire, noter.**
- Relever le RMSE (moyenne ± écart-type) de 05a, 05b et 05c. On attend
  environ 3,71, 3,58 et 3,52.
- Remplir le Status et la ligne 05 de `JOURNAL.md`.
- Transmettre la configuration 05c à la personne chargée de la
  soumission Kaggle.

**Définition de « terminé » :** tests verts, les trois rapports sur
Skore Hub, Status rempli, configuration retenue communiquée pour la
soumission.

## Risks / things that could invalidate the result

- **Hyperparamètres choisis sur la même CV (05b) :** le 3,58 est
  légèrement optimiste. Viser plutôt 3,55–3,6. Ne pas multiplier les
  essais de réglage sur cette CV sans garder un fold à part.
- **Pas de fuite de target :** enveloppes et lissage n'utilisent que
  `off`, `on`, `ledd`, `age` et les prédictions. Refuser en revue toute
  statistique par patient calculée sur `target`.
- **Toutes les visites d'un patient ensemble :** enveloppes et lissage
  sont faux si l'on prédit patient par patient en morceaux, ou ligne par
  ligne. Prédire sur tout `X_test` d'un coup.
- **Patients avec peu de mesures** (4 visites, parfois 1 seul OFF) :
  l'enveloppe est fragile. `n_off_patient` (03) aide le modèle. À
  vérifier dans le rapport skore : erreur par nombre de visites.
- **Lissage et patients atypiques :** une parabole peut mal se
  comporter aux extrémités sur des patients à 4–5 visites. Si 05c
  dégrade ces patients, essayer `weight=0.5` ou `deg=1` pour eux.
- **Le modèle de bruit est une hypothèse empirique** (sous-notation
  multiplicative). Si le test était généré différemment (peu probable,
  les distributions train/test sont identiques), le gain ne se
  retrouverait pas sur Kaggle. La soumission le confirmera.

## Status

- **State:** planned
- **Approved by user on:** n/a
- **Headline result:** n/a
- **Implication for next iteration:** n/a
