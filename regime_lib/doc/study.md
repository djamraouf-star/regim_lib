# Module d'étude

**Dernière mise à jour** : 2026-10-06
**Module** : `regime_lib/study/`
**Statut** : implémenté ; validation logicielle sur données synthétiques

---

## 1. Objectif

Répondre à une seule question :

> Conditionnellement à une feature F, la cible X se comporte-t-elle
> différemment ?

Une feature peut être un **régime**, une **confidence**, ou un
**attribut de contexte** (`ctx_session`, `ctx_in_event`, etc.). Une
cible est une mesure du **futur** (rendement, volatilité, drawdown).

## 2. Architecture

Une étude est un pipeline :

| Module | Rôle |
|---|---|
| `targets.py` | Définir et calculer les cibles |
| `features.py` | Extraire les features d'un parquet de régimes |
| `split.py` | Découper train/test |
| `evaluation.py` | Calculer IC, hit rate, tests statistiques |
| `study.py` | Orchestrer les quatre étapes précédentes |
| `report.py` | Générer le rapport Markdown des études et le rapport descriptif des runs |

Chaque module est utilisable indépendamment.

## 3. Cibles

Convention de nommage : `<famille>_fwd_<K>`.

| Famille | Formule | Usage |
|---|---|---|
| `ret` | `log(close[t+K]/close[t])` | Direction |
| `vol` | écart-type des K log-returns de `t+1` à `t+K` | Volatilité future |
| `dd` | `min(close[t:t+K+1])/close[t] - 1` | Drawdown |
| `ru` | `max(close[t:t+K+1])/close[t] - 1` | Run-up |

**Lookahead** : par nature, les cibles utilisent le futur. Elles
servent de vérité terrain pour évaluer les features. Les K dernières
barres sont NaN.

## 4. Features

Le module extrait les features depuis un parquet long du CLI. Il
pivote pour avoir une colonne par `(méthode, feature)`.

**Catégories de features** :

| Type | Exemples |
|---|---|
| Catégorielles | `regime`, `ctx_session`, `ctx_event_type` |
| Numériques | `confidence`, `ctx_hour_ny`, `ctx_day_of_week` |
| Booléennes | `ctx_is_rollover`, `ctx_is_holiday`, `ctx_in_event` |

**Disponibilité** : le filtre `methodes_causales_only=True` conserve les méthodes
avec `availability="bar_close"`, selon les métadonnées exportées ou, pour les
anciens parquets, le contrat du registre. HMM, Price Action et MSS/CHOCH sont rétrospectifs et exclus. VPIN utilise
désormais les buckets antérieurs clos ; ses anciens parquets sans métadonnées
restent exclus, car ils peuvent contenir une fuite future. Une méthode inconnue sans déclaration est exclue.
Les nouvelles sorties portent `available_at` ; une date future/manquante est
refusée en mode causal. Une déclaration contradictoire avec le registre est
refusée. Cela ne transforme pas les méthodes rétrospectives en méthodes causales.

Les labels restent nominaux. `regime_dimension` et `regime_description`
précisent ce que mesure chaque détecteur ; deux labels RANGE de familles
volatilité/force de tendance n'ont pas la même signification. `regime_id` ne
constitue pas une échelle et ces métadonnées ne calibrent pas `confidence`.

## 5. Split

| Schéma | Usage |
|---|---|
| `in_sample` | Analyse exploratoire, pas de validation |
| `holdout` | Une coupure train / test |
| `walk_forward` | N fenêtres glissantes |

**Purge automatique** : les cibles du train qui atteignent le premier
timestamp de test sont retirées par `Study`. Un `embargo` facultatif ajoute
un espace entre les fenêtres avant cette purge.

## 6. Évaluation

### 6.1 Métriques globales

- **IC Spearman** : corrélation de rang pour les features continues ou explicitement ordonnées. Non applicable aux catégories nominales.
- **Hit rate directionnel** : % de bonnes directions (features
  explicitement directionnelles, sur une cible de rendement uniquement).

### 6.2 Batterie de 3 tests

Pour chaque modalité d'une feature, comparaison one-vs-rest :

| Test | Compare | Convention "succès" |
|---|---|---|
| **Welch** | Moyennes | — |
| **Mann-Whitney** | Distributions | — |
| **z de proportions** | Fréquences | dépend de la cible |

**Convention de succès du test z** :

| Cible | Succès |
|---|---|
| `ret_fwd_K` | cible > 0 |
| `vol_fwd_K` | cible > médiane globale |
| `dd_fwd_K` | cible > médiane globale (excursion moins négative) |
| `ru_fwd_K` | cible > médiane globale |

La médiane est calculée sur l'union des deux groupes (sinon les deux
groupes auraient mécaniquement 50 % de succès chacun).

### 6.3 Portée descriptive

Les tests historiques Welch/Mann-Whitney/z sont descriptifs. Le franchissement
simultané de trois seuils ne prouve pas la robustesse sous dépendance temporelle.
Utiliser l'inférence explicite de la section 7.8, sans sélectionner a posteriori
les périodes, paramètres ou tailles de blocs les plus favorables.

## 7. Utilisation

Depuis l'étape 2, les entrées doivent respecter les contrôles d'identité,
de fréquence et de support décrits dans le
[guide d'alignement et de validité](./alignement_validite.md).

### 7.1 Cas minimal

```python
from regime_lib.study import Study

etude = Study(
    regimes_path="results/regimes.parquet",
    ohlcv_path="data/EURUSD_Tick_...parquet",
    features=["regime"],
    targets=["ret_fwd_5", "vol_fwd_20"],
    methodes=["adx", "er_kaufman"],
    split="holdout",
    train_ratio=0.7,
    embargo=20,
)
etude.run()

print(etude.resultats)

detail = etude.detail_conditionnel(
    feature="adx__regime",
    cible="vol_fwd_20",
    fold=0,
)
print(detail)

tests = etude.tests_modalite_pour(
    feature="adx__regime",
    cible="vol_fwd_20",
)
print(tests)

etude.save("results/study/")
# Produit :
#   study_results.parquet         (métriques principales)
#   study_tests_modalite.parquet  (tests par modalité)
#   study_coverage.parquet        (couverture propre et commune)
#   study_common_support.parquet  (masque par timestamp et cible)
#   study_report.md               (rapport lisible)

etude = Study(
    ...,
    split="walk_forward",
    n_folds=5,
    embargo=20,
)
etude.run()
# etude.resultats contient une ligne par fold
```

### 7.2 Rapport descriptif d'un run

Le générateur historique est intégré à [report.py](../study/report.py).
Il décrit les régimes de toute méthode (statistiques, persistance, transitions et confidence),
tandis que `Study` évalue les features par rapport aux cibles futures.
Les labels d’origine sont conservés, notamment `STATE_N`, `TENDANCE` ou `RANGE`.
Les codes internes sont nominaux et ne définissent aucun ordre entre régimes.

```python
from regime_lib.study.report import rapport_complet

rapport = rapport_complet(
    "results_hmm",
    url_source="data/EURUSD_M1.parquet",
    method="hmm_gaussian",
    timeframe="H1",
)
```

```bash
python -m regime_lib.study.report --run-dir results_hmm \
    --url data/EURUSD_M1.parquet --method hmm_gaussian --timeframe H1
```

La commande `regime-report` reste disponible après réinstallation du paquet.
L'ancien chemin `regime_lib.utils.report` est supprimé : les imports et appels
avec `python -m` doivent utiliser `regime_lib.study.report`.

### 7.3 Contrôles communs des entrées

Les études et les rapports descriptifs utilisent [validation.py](../study/validation.py).
Les colonnes `asset` et `timeframe` sont obligatoires pour les régimes.
La sélection d'une configuration précède la vérification de l'identité ;
plusieurs `params_hash` sans sélection explicite sont refusés.
Les champs `source`, `price_side`, `adjustment` et `calendar`, lorsqu'ils sont
présents dans les régimes, doivent correspondre aux métadonnées des prix.

Les timestamps doivent être uniques et en UTC. Les régimes sont triés avant
validation ; chaque timestamp retenu doit exister dans les prix au timeframe
cible. Un trou des prix ne disparaît donc plus par simple intersection.
Les contrôles des trajectoires futures restent assurés par le calcul des cibles.

Les indicateurs `is_valid`, `is_partial`, `is_warmup`, les régimes inconnus
et les valeurs absentes déterminent un masque partagé. Les lignes invalides
sont masquées, sans compression de la chronologie. Les champs de validité
absents restent facultatifs ; les valeurs manquantes dans un indicateur présent
sont traitées comme invalides. Les features numériques non finies sont masquées.

La couverture expose `n_observations`, `n_valide`, `n_exclu` et les motifs
`n_invalide`, `n_partiel`, `n_chauffe`, `n_inconnu`, `n_absent`.
Ces motifs peuvent se chevaucher : leur somme n'est pas l'effectif exclu.
Le rapport descriptif indique aussi le support effectivement exploitable et
les observations sans features disponibles. Un support vide donne une mention
explicite d'absence d'estimation.

Le rapport accepte les arguments `asset`, `configuration`, `ohlcv_asset` et
`ohlcv_timeframe`, également disponibles en CLI sous les noms `--asset`,
`--configuration`, `--ohlcv-asset` et `--ohlcv-timeframe`.

### 7.4 Continuité des analyses descriptives

Une transition est comptée uniquement entre deux observations valides et
adjacentes sur la grille du timeframe. Les états inconnus, les observations
masquées et les trous temporels interrompent les épisodes. Les durées sont
exprimées en nombre de barres observées ; les épisodes aux frontières sont
tronqués, sans extrapolation de leur durée totale. Les fermetures de marché
créant un trou sur cette grille interrompent elles aussi les épisodes.

Le rapport affiche les effectifs de transition ainsi que les probabilités.
Une ligne sans transition sortante observée est non estimable (`NaN`).
Les appels directs à `persistance` et `matrice_transition` acceptent
`timeframe="H1"` ; une fréquence doit être déclarée sur un index irrégulier.
Sans index temporel, seules les positions successives définissent l'adjacence.

Les statistiques par état et les proportions excluent les états invalides.
Le rapport ne renomme pas automatiquement deux catégories en « calme » et
« agité » et ne déduit pas un biais directionnel d'un seuil fixe de rendement.
Les mois affichés respectent un seuil de jours observés, sans certification
qu'ils couvrent toutes les séances attendues.

### 7.5 Contrat des métriques

`Study(..., feature_types={"signal": "directional"})` déclare le type d'une
feature. Une clé complète (`adx__signal`) est prioritaire sur une clé courte
(`signal`). Les types reconnus sont `nominal`, `ordinal`, `continuous` et
`directional`. Les déclarations inconnues ou les noms de features absents
sont refusés.

Par défaut, les régimes, le contexte `ctx_*`, les booléens et les catégories
non ordonnées sont nominaux, même si leur représentation est numérique.
Les autres variables numériques sont continues. Les catégories pandas
`ordered=True` sont ordinales ; leur ordre explicite est conservé.
Un signal directionnel n'est jamais déduit automatiquement du dtype.
Une déclaration ordinale textuelle exige une catégorie pandas ordonnée
ou un encodage numérique fourni explicitement par l'appelant.

```python
from regime_lib.study.evaluation import hit_rate_directionnel

resultat = hit_rate_directionnel(
    signal, rendement,
    type_feature="directional", nom_cible="ret_fwd_5",
)
```

Le nom de la cible peut aussi venir de `rendement.name`. Sans déclaration
directionnelle et sans cible de rendement identifiable, le hit rate est
non applicable. C'est un changement du contrat des appels directs.
Le signe numérique prédit la direction ; zéro est une abstention. Pour un
booléen déclaré directionnel, True prédit une hausse et False une baisse.
Un rendement nul compte comme une prédiction incorrecte sur un signal actif.

Les valeurs absentes ou infinies sont exclues. Chaque métrique exige au
moins dix paires exploitables (après retrait des abstentions pour le hit rate).
Les résultats exportent `feature_type`, `ic_status_train/test`,
`hit_status_train/test` et `n_hit_train/test`, en plus des effectifs de paires
`n_train/test`. Les statuts distinguent `ok`, `constant`,
`insufficient_samples` et les motifs `not_applicable_*`. Une métrique non
applicable reste NaN ; son absence ne constitue pas une absence d'effet.

`vol_fwd_1` est désormais refusée : l'écart-type échantillonnal requiert au
moins deux rendements. `liste_cibles` omet cette combinaison ; les autres
familles conservent l'horizon 1.

### 7.6 Séparation temporelle et purge

En walk-forward, `train_ratio` représente train/(train+test), avant embargo
et purge, à l'arrondi d'une barre près. Les fenêtres de test ont la même taille
et ne se chevauchent pas. En mode ancré, le ratio fixe le premier train,
puis celui-ci s'étend. Les éventuelles dernières barres insuffisantes pour
une fenêtre complète ne sont pas utilisées. Les tailles impossibles, index
invalides et fenêtres vides sont refusés.

`Study.df_fin_cibles` conserve les dates de fin calculées sur la même grille
que les cibles, y compris avec `expected_index`. Pour chaque fold et cible,
seules les observations du train dont la fin est strictement antérieure au
début du test sont conservées. Les fins inconnues sont également retirées.
Un train entièrement purgé déclenche une erreur avant export. La purge
ne s'applique pas au mode exploratoire `in_sample`.

Les résultats exposent `n_purge_train` ; les dates de fin sont exportées dans
`study_target_ends.parquet`. Les effectifs `n_train` comptent les paires
valides après purge. Les fonctions de split seules ne connaissent pas les
horizons des cibles : la purge est assurée par l'orchestrateur `Study`.

Cette étape n'ajoute aucune transformation apprise ni sélection automatique
sur les données. Les déclarations de types et les ordres des catégories sont
fournis avant l'étude. Les tests statistiques descriptifs sur le test ne
constituent pas des seuils appris pour une prédiction. Toute future
transformation apprise devra être ajustée exclusivement sur le train purgé.

### 7.7 Évaluation de tous les folds

`resultats_tests` utilise désormais les clés `(fold, feature, cible)`.
L'accesseur `tests_modalite_pour(feature, cible, fold=0)` conserve le premier
fold par défaut ; préciser `fold` pour les autres périodes. Les détails
conditionnels sont toujours accessibles par `detail_conditionnel(..., fold=...)`.
Les exports des tests par modalité comportent une colonne `fold`.

Chaque comparaison modalité contre reste expose :

- `mean_diff` et `median_diff` : différence modalité moins reste, dans l'unité de la cible ;
- `cliffs_delta` : `2 * U / (n_a * n_b) - 1`, orienté modalité moins reste,
  avec égalités comptées pour moitié dans U ;
- `n_a`, `n_b`, `n_episodes_a`, `n_episodes_b` : effectifs et épisodes observés.

Les valeurs invalides et trous temporels interrompent les épisodes. Si un
calendrier `expected_index` est fourni, l'adjacence suit ce calendrier.
Les appels directs sur un index irrégulier doivent déclarer `timeframe`.
Les durées et nombres d'épisodes restent descriptifs, sans garantie d'indépendance.

Le statut `insufficient_samples` indique moins de deux observations dans un
groupe ; `insufficient_episodes` indique moins de deux épisodes dans un groupe.
Sinon le statut est `descriptive`, ce qui ne constitue pas une validation
statistique. Sans comparateur, l'effet est non estimable. Le delta de Cliff
n'est calculé que lorsque Mann-Whitney dispose d'au moins deux points par groupe.

Deux exports supplémentaires sont disponibles :

- `study_conditionnel.parquet` : statistiques de toutes les modalités, features,
  cibles et folds, y compris leur nombre d'épisodes ;
- `study_stability.parquet` : stabilité des écarts de moyenne entre folds,
  également accessible par `Study.stabilite_modalites`.

La stabilité indique les folds totaux, présents, estimables et suffisamment
représentés, les moyenne/écart-type/min/max des effets et leurs signes.
Les effets sont agrégés sans pondération ; une modalité absente n'est pas
remplacée par un effet nul. `sign_stability` est la fréquence du signe non nul
majoritaire parmi les effets estimables ; les zéros restent au dénominateur.
Un seul fold peut donner une stabilité de 1 sans établir une robustesse :
les nombres de folds doivent toujours être consultés.

Le rapport présente tous les folds et toutes les tables conditionnelles.
Les p-values restent descriptives et non corrigées pour la dépendance
temporelle ou les comparaisons multiples ; ces corrections relèvent de l'étape 7.

### 7.8 Inférence temporelle optionnelle

```python
from regime_lib.study import Study, InferenceConfig

etude = Study(
    regimes_path="results/regimes.parquet",
    ohlcv_path="data/prices.parquet",
    targets=["ret_fwd_5"],
    inference=InferenceConfig(
        block_size=20, n_resamples=1999, confidence_level=0.95,
        seed=42, min_group_size=10,
    ),
)
```

L'inférence est désactivée par défaut et requiert `tests_modalite=True`.
Elle porte sur **l'écart de moyenne modalité moins reste**, et non sur l'IC,
le hit rate, l'écart de médiane ou le delta de Cliff. Les intervalles sont
ponctuels, pas simultanés. La configuration est exportée dans
`study_inference.json`, et les tables des modalités conservent les paramètres.

Le bootstrap rééchantillonne conjointement feature et cible, par blocs mobiles
complets. Aucun bloc ne traverse une paire invalide ou un trou temporel ;
lorsqu'un calendrier explicite est fourni, il définit l'adjacence. Un segment
valide plus court qu'un bloc rend l'inférence non estimable : il n'est pas
retiré silencieusement. Les blocs mobiles peuvent sous-représenter les bords
des segments ; cette approximation nécessite des séries suffisamment longues.

`block_size` est déclaré avant l'analyse et doit être au moins égal à chaque
horizon cible. Ce minimum ne suffit pas nécessairement à capturer toute la
dépendance : il faut examiner la sensibilité aux tailles de blocs, sans choisir
celle qui rend les résultats significatifs. Le bootstrap suppose une dépendance
faible et une distribution suffisamment stable à l'intérieur de chaque fold.

Les garde-fous exigent quatre longueurs de bloc sur le support, au moins
`min_group_size` observations et deux épisodes par groupe, ainsi qu'au moins
90 % de réplications exploitables. Les statuts `insufficient_support`,
`insufficient_resamples` et `degenerate_bootstrap` laissent p-values et
intervalles à NaN. Ces seuils sont des garde-fous, pas des garanties de calibration.

Les intervalles `mean_diff_ci_low/high` utilisent la formule basic :
`[2*effet - quantile_sup, 2*effet - quantile_inf]`.
Le test bilatéral `block_p` compare `abs(effet_bootstrap - effet_observe)`
à `abs(effet_observe)` ; une correction Monte Carlo de +1 évite les p-values
nulles. Il s'agit d'une approximation bootstrap centrée, pas d'un test exact.
`bootstrap_valid` et `block_mc_se` indiquent la précision Monte Carlo.
La résolution minimale est `1/(bootstrap_valid+1)` : une grande famille de
tests nécessite souvent davantage de réplications pour distinguer de petites p-values.

`block_p_by` applique Benjamini–Yekutieli à **une famille globale** comprenant
toutes les lignes fold × feature × cible × modalité produites par cette étude.
Les lignes non estimables comptent comme p=1 dans la taille de famille, tout
en conservant NaN dans les exports. Les modalités absentes d'un fold ne créent
pas de ligne. `reject_by` reste indéterminé pour les tests non estimables.
Les recherches effectuées dans d'autres études ne sont pas incluses dans cette
famille : multiplier les exécutions puis sélectionner leurs résultats contourne
la correction. Celle-ci suppose des p-values valides et ne répare pas une
mauvaise approximation temporelle.

Les p-values historiques Welch, Mann-Whitney et z restent descriptives,
sans correction temporelle ni ajustement multiple. Les intervalles et p-values
bootstrap ont leurs propres colonnes ; aucune garantie de robustesse n'est déduite
du simple franchissement d'un seuil.

Références de méthode : [intervalles basic, arch](https://arch.readthedocs.io/en/latest/bootstrap/confidence-intervals.html)
et [correction BY, statsmodels](https://www.statsmodels.org/stable/_modules/statsmodels/stats/multitest.html).

### 7.9 Exports et reproductibilité

`run()` fige un manifeste de l'étude effectuée. `save()` exporte
`study_metadata.json` avec la configuration, les versions Python et dépendances,
les périodes, les bornes des folds avant purge, les exclusions, les empreintes
des données effectivement utilisées et des sources Python du paquet.
Le parquet de régimes est identifié par son chemin et son SHA-256 ; les prix
préparés, features, cibles et dates de fin ont leurs propres empreintes.
Les données injectées via `ohlcv_data` sont identifiées comme telles.
Les empreintes pandas permettent la comparaison à environnement identique ;
elles ne garantissent pas une identité binaire entre versions de pandas.

Le manifeste référence également les empreintes des fichiers exportés.
Deux exécutions identiques dans le même environnement, avec les mêmes paramètres
et graine, produisent les mêmes résultats et manifeste (sans date d'export variable).
Il faut conserver les entrées d'origine pour rejouer l'étude : le manifeste
n'est pas une copie des données. Une modification des paramètres, des données
évaluées ou du code après `run()` requiert une nouvelle exécution avant export.

L'export est entièrement préparé dans un répertoire temporaire avant publication.
Un échec de calcul du rapport ou de sérialisation laisse l'export précédent intact.
Les fichiers générés optionnels devenus inutiles sont supprimés ; les fichiers
personnels du répertoire sont conservés. Le manifeste est remplacé en dernier.
La publication de plusieurs fichiers n'est pas une transaction atomique globale :
une interruption disque pendant cette phase se détecte en vérifiant les empreintes
par rapport au manifeste. Ne pas faire écrire deux études simultanément au même endroit.

Un support valide vide produit un rapport explicite et une couverture nulle.
Les tables optionnelles sans lignes ne sont pas exportées. Un échec de `run()`
invalide les anciens résultats afin d'empêcher leur export accidentel.

Les rapports distinguent :

- **Description** : statistiques des régimes observés, sans validation prédictive ;
- **Exploration** : `in_sample`, avec mêmes observations pour train et test ;
- **Évaluation temporelle** : holdout ou walk-forward avec purge.

Le dernier cas ne certifie pas une validation indépendante finale : il faut
avoir réservé les périodes et figé les choix d'analyse avant de les consulter.
Aucun indicateur logiciel ne permet de garantir ce protocole à lui seul.

### 7.10 Installation et vérification des commandes

```bash
python -m pip install -e .
regime-report --help
python -m regime_lib.study.report --help
regime-lib --help
```

La réinstallation régénère le lanceur `regime-report` et remplace sa référence
historique à `regime_lib.utils.report`. SciPy est une dépendance directe du paquet.
Les scénarios de reproductibilité et de réécriture d'exports sont couverts par
[les tests dédiés](../../tests/test_study_reproducibility.py).

### 7.11 Sensibilité et confirmation indépendante

`regime_lib.study.inference.block_sensitivity(feature, target,
block_sizes=[20, 40, 80], horizon=5, config=InferenceConfig(20), timeframe="H1")`
retourne toutes les modalités pour chaque taille pré-spécifiée ; une correction
BY couvre la famille tailles × modalités du diagnostic. Aucun bloc « optimal »
n'est choisi. Les statuts non estimables sont conservés. Ce diagnostic ne
remplace pas la famille globale d'une étude complète.

Avant d'examiner une période finale, figer dans un protocole daté : méthodes,
configurations, features/modalités, cibles/horizons, calendrier, bornes temporelles,
critère principal, famille de tests, taille principale des blocs, tailles de
sensibilité et règle de décision. Réserver les données finales, purger les cibles
atteignant leur frontière, puis exécuter une seule analyse de confirmation.
Un changement après consultation transforme le résultat en exploration ; une
nouvelle confirmation exige une nouvelle période réservée. Le logiciel ne peut
pas certifier que l'utilisateur n'a jamais consulté les données finales.

Le manifeste inclut maintenant la version du protocole, l'environnement complet,
les ressources du paquet et l'empreinte explicite du support commun. Le manifeste
du CLI de détection inclut aussi les données préparées, les sorties et le parquet
exporté. Les chemins seuls ne servent jamais d'identifiants de contenu.
