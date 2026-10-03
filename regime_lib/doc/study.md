# Module d'étude

**Dernière mise à jour** : 2026-10-03
**Module** : `regime_lib/study/`
**Statut** : implémenté et validé

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
| `report.py` | Générer un rapport Markdown |

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

**Méthodes offline** : `hmm_gaussian`, `price_action`, `mss_choch` sont
exclues par défaut (`methodes_causales_only=True`) — leur lookahead
structurel biaise l'évaluation.

## 5. Split

| Schéma | Usage |
|---|---|
| `in_sample` | Analyse exploratoire, pas de validation |
| `holdout` | Une coupure train / test |
| `walk_forward` | N fenêtres glissantes |

**Embargo** : indispensable quand la cible utilise le futur. Met
`embargo >= K` (horizon de la cible) pour éviter que les dernières
cibles du train empiètent sur le test.

## 6. Évaluation

### 6.1 Métriques globales

- **IC Spearman** : corrélation de rang feature ↔ cible.
- **Hit rate directionnel** : % de bonnes directions (features
  directionnelles uniquement).

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
| `dd_fwd_K` | cible < médiane globale |
| `ru_fwd_K` | cible > médiane globale |

La médiane est calculée sur l'union des deux groupes (sinon les deux
groupes auraient mécaniquement 50 % de succès chacun).

### 6.3 Lecture croisée

| Observation | Interprétation |
|---|---|
| 3 p-values < 0.001 | Signal robuste |
| Welch seul significatif | Signal tiré par des outliers |
| Mann-Whitney seul significatif | Décalage de distribution sans changement de moyenne |
| Aucun significatif | Pas de signal |

## 7. Utilisation

### 7.1 Cas minimal

```python
from regime_lib.study import Study

etude = Study(
    regimes_path="results/regimes.parquet",
    ohlcv_path="actifs/EURUSD/EURUSD_Tick_...parquet",
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

etude.save("analyses/EURUSD/output/")
# Produit :
#   study_results.parquet         (métriques principales)
#   study_tests_modalite.parquet  (tests par modalité)
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

## 8. Criblage complet EURUSD et recherche sur les sessions

Le script [`screening_eurusd.py`](../../analyses/EURUSD/scripts/screening_eurusd.py)
exécute un criblage reproductible à partir du Parquet OHLCV ou tick :

```bash
python analyses/EURUSD/scripts/screening_eurusd.py
```

Par défaut, il traite `M5` et `H1`, utilise le profil `eurusd`, crible les
méthodes causales disponibles et évalue les cibles
`ret`, `vol`, `dd` et `ru` aux horizons de 1, 3, 6, 12 et 24 barres. Chaque
timeframe est lancé et évalué séparément : un horizon correspond toujours
au nombre de barres de ce timeframe. Les folds sont chronologiques
walk-forward et l'embargo vaut au moins le plus grand horizon.

Le script relance le CLI avec `--with-context`, puis produit :

- `method_screening/` : métriques train/test par méthode et régime ;
- `context_screening/` : criblage des sessions du profil, heures NY, jours
  de semaine, rollover, fériés et événements ;
- `session_regime_screening.parquet` : comparaison des régimes entre eux
  **au sein de chaque session**, uniquement sur les folds de test. Les
  lignes avec `INCONNU` et `HORS_SESSION` sont exclues de cette comparaison.

Les p-values du criblage croisé sont accompagnées de valeurs q corrigées
par Benjamini-Hochberg, séparément par fold, méthode et cible. Cette
correction limite les faux positifs dus aux comparaisons multiples ; elle
ne corrige pas la dépendance temporelle des rendements qui se chevauchent.
Les résultats restent exploratoires et doivent être confirmés sur une
période indépendante.

Options usuelles :

```bash
python analyses/EURUSD/scripts/screening_eurusd.py \
  --timeframes M15,H1 \
  --horizons 1,4,12 \
  --families ret,vol \
  --folds 6 \
  --output analyses/EURUSD/output/criblage_m15_h1
```

Par défaut, les résultats sont écrits dans un nouveau dossier horodaté
sous `analyses/EURUSD/output/`. Un dossier non vide est refusé pour éviter
d'écraser des sorties existantes ; `--overwrite` permet explicitement de
réutiliser un dossier.
