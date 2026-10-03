> ⚠ **Document archivé — décision révisée.**
>
> Le consensus multi-méthodes a été retiré du projet. La raison :
> les labels ne sont pas comparables entre méthodes (le RANGE d'ATR
> n'est pas le RANGE d'ADX). Un vote de labels matérialise une erreur
> ontologique. Pour agréger plusieurs méthodes, utiliser directement
> pandas (voir docstring de `core/output.py`).
>
> Ce document est conservé pour traçabilité historique uniquement.

# Consensus multi-méthodes : état, analyse et décisions

Ce document synthétise l'historique, les mesures empiriques et les décisions de conception concernant le consensus multi-méthodes dans `regime_lib`.

---

## 1. Décision : Retrait du pipeline standard

Le calcul d'une colonne `consensus` pré-calculée a été **retiré du pipeline CLI standard** le 27 septembre 2026.

La fonction `add_consensus` n'est plus présente dans
[`core/output.py`](../../core/output.py). Pour explorer les accords entre
méthodes, construire les pivots pandas illustrés plus bas.

### Justifications du retrait

1. **Redondance** : Les états de consensus se reconstruisent instantanément à partir des colonnes `timestamp`, `method` et `regime` du parquet long via un pivot pandas. Figer une colonne dans le parquet était redondant.
2. **Rigidité** : Figer 4 états arbitraires empêchait les filtrages conditionnels fins dont les stratégies de trading ont besoin (ex. filtrer par seuil de confiance, par sous-ensemble de méthodes, ou par famille sémantique).
3. **Hétérogénéité sémantique** : Les labels `RANGE` et `CHOP` n'ont pas la même signification d'une méthode à l'autre, ce qu'un consensus naïf a tendance à masquer.

---

## 2. Hétérogénéité sémantique entre méthodes

### Définitions des labels partagés

Bien que les méthodes partagent les libellés `RANGE` et `CHOP`, elles mesurent des réalités de marché fondamentalement différentes :

| Méthode | Signification de `RANGE` | Signification de `CHOP` |
|---|---|---|
| `atr_volatility` | Volatilité **faible** ($\text{ATR} \le q_{33}$) | Volatilité **intermédiaire** |
| `er_kaufman` | Efficience **intermédiaire** ($q_{40} < \text{ER} < q_{75}$) | Efficience **faible / bruit** ($\text{ER} \le q_{40}$) |
| `adx` | Absence de tendance ($\text{ADX} < 20$) | Zone de transition ($20 \le \text{ADX} < 25$) |
| `price_action` | Sommets ET creux **quasi-plats** (4 pivots alignés) | Configuration **mixte** (ambiguïté des pivots) |

Le `RANGE` de `price_action` est extrêmement restrictif (il exige l'alignement strict de 4 pivots dans la tolérance ATR), alors que son `CHOP` est très permissif.

### Mesure empirique (EURUSD H1, 14 mois / 6 566 barres)

Proportions de labels observées sur le même historique :

| Méthode | CHOP | INCONNU | RANGE | TEND_B | TEND_H |
|---|---|---|---|---|---|
| `atr_volatility` | 31.9 % | 3.8 % | **36.0 %** | 14.1 % | 14.2 % |
| `er_kaufman` | 36.9 % | 5.7 % | **33.5 %** | 10.8 % | 13.1 % |
| `price_action` | **49.4 %** | 0.6 % | **4.4 %** | 24.5 % | 21.1 % |

`price_action` produit **10× moins de RANGE** et **1.5× plus de CHOP** qu'ATR ou ER.

### Effet sur le consensus naïf

| Type de consensus | Causal (ATR + ER) | Mixte (+ Price Action) |
|---|---|---|
| `aligne` | **34.4 %** | **11.5 %** |
| `divergent` | **34.6 %** | **62.2 %** |
| `partiel` | 25.3 % | 20.7 % |
| `insuffisant` | 5.6 % | 5.6 % |

Inclure `price_action` (méthode offline) fait mécaniquement chuter l'accord strict de 34 % à 11 % et bondir la divergence à 62 %. Ce n'est pas une anomalie de code, mais la signature de la divergence conceptuelle entre price action et indicateurs causaux.

---

## 3. Analyse des divergences causales & Proposition v2

L'analyse approfondie des 2 272 barres où ATR et ER divergent sur EURUSD H1 a mis en lumière une structure très nette :

1. **Aucune contradiction directe** : Un régime `TENDANCE_HAUSSIERE` sur l'ATR ne coïncide **jamais** avec une `TENDANCE_BAISSIERE` sur l'ER, et vice-versa.
2. **Deux régimes de divergence distincts** :
   - `ATR directionnel + ER neutre` (1 275 cas, 56 %) : **Tendance impulsive** (mouvements violents avec pullbacks prononcés).
   - `ATR neutre + ER directionnel` (997 cas, 44 %) : **Grind** (tendance régulière, faible volatilité, setup très favorable au suivi de tendance).

### Proposition v2 (5 états)

Une nomenclature plus expressive a été modélisée pour l'analyse des signaux :

| État | Condition | Interprétation |
|---|---|---|
| `aligne` | Même label sur toutes les méthodes | Signal fort et confirmé |
| `partiel` | Labels distincts mais même famille neutre (`RANGE` vs `CHOP`) | Marché sans direction confirmé |
| `impulsif` | ATR en tendance, ER neutre | Tendance nerveuse |
| `grind` | ATR neutre, ER en tendance | Tendance lente et propre |
| `contradictoire` | Directions opposées haussière vs baissière | Incohérence (garde-fou) |
| `insuffisant` | Au moins un `INCONNU` ou moins de 2 méthodes | Période de chauffe / incomplète |

L'introduction de la méthode **ADX** sert d'arbitre naturel pour départager ces configurations.

---

## 4. Recommandations pour l'agrégation en aval

L'utilisateur effectue directement ses croisements en quelques lignes avec pandas sur `regimes.parquet` :

```python
import pandas as pd

# 1. Chargement et pivot
df = pd.read_parquet("results_multi/regimes.parquet")
h1 = df[df["timeframe"] == "H1"]
pivot_regime = h1.pivot(index="timestamp", columns="method", values="regime")
pivot_conf = h1.pivot(index="timestamp", columns="method", values="confidence")

# 2. Filtrer les barres où ATR et ER confirment tous deux une hausse
tendance_haussiere = pivot_regime[
    (pivot_regime["atr_volatility"] == "TENDANCE_HAUSSIERE") &
    (pivot_regime["er_kaufman"] == "TENDANCE_HAUSSIERE")
]

# 3. Filtrer par accord strict avec seuil de confiance minimal
accord_strict = pivot_regime[
    pivot_regime.eq(pivot_regime.iloc[:, 0], axis=0).all(axis=1) &
    (pivot_conf.min(axis=1) >= 0.7)
]
```
