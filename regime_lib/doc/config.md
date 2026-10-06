# Configuration par profils

**Dernière mise à jour** : 2026-10-05
**Module** : `regime_lib/config/`
**Source de vérité** : `regime_lib/config/profiles/default.yaml`

---

## 1. Objectif

Rendre la bibliothèque **portable entre actifs** sans toucher au code.
Changer d'actif, de timeframe ou de calibration revient à **changer de
profil**, pas à modifier les méthodes.

Chaque actif a ses propres paramètres (fenêtres, seuils, quantiles) qui
dépendent de sa volatilité et de sa microstructure. Le profil les
regroupe dans un fichier YAML lisible et versionnable.

## 2. Structure d'un profil

Un profil est un fichier YAML avec des métadonnées (`nom`, `type`,
`description`) et des sections de configuration :

```yaml
nom: eurusd                          # identifiant court (obligatoire)
type: forex_majeur                   # catégorie libre (documentaire)
description: "EURUSD — 2025-2026"   # texte libre

donnees:
  source: barres                     # 'barres' ou 'tick'
  timeframes: [M1, M5, M15, H1]

methodes:
  <nom_methode>:
    <paramètre>: <valeur>
    timeframes:
      <TF>: {<paramètre>: <valeur>}

segmentation:
  modele: normal
  pen: 50.0
  min_size: 20
```

- `nom` identifie le profil. `type` et `description` sont des métadonnées
  descriptives.
- `donnees.source` accepte `barres` ou `tick`. `donnees.timeframes` décrit
  les timeframes associés au profil.
- `methodes` contient une entrée par identifiant de méthode. Les paramètres
  variant selon le timeframe sont groupés sous `timeframes`.
- `segmentation` décrit les paramètres du module de segmentation.

Les noms de méthode doivent correspondre exactement aux identifiants
enregistrés, par exemple `shannon` ou `pca_axis`. Chaque détecteur déclare un `PARAM_SCHEMA` : clés, types stricts, bornes,
autorisation par timeframe et contraintes entre paramètres. Le chargement
valide toutes les méthodes et tous les timeframes, même non sélectionnés.
Les clés inconnues, doublons YAML, booléens utilisés comme entiers et valeurs
non finies sont rejetés avec un diagnostic. Les signatures Python gardent
leurs arguments nommés ; elles ne définissent plus de valeurs de secours.

## 3. Profils embarqués et personnalisés

Les profils embarqués sont stockés dans
`regime_lib/config/profiles/`. Lister les noms disponibles :

```bash
regime-lib --list-profiles
```

Charger un profil embarqué par nom :

```bash
regime-lib \
  --url data/EURUSD_M1.parquet \
  --timeframe H1 \
  --methods adx,atr_volatility \
  --profile eurusd \
  --output results_eurusd
```

Un profil personnalisé peut être placé ailleurs et chargé par chemin :

```bash
regime-lib \
  --url data/GBPUSD_M1.parquet \
  --timeframe H1 \
  --methods adx \
  --profile config/profiles/gbpusd.yaml \
  --output results_gbpusd
```

Utiliser l'extension `.yaml` ou `.yml` pour un chemin de profil explicite. La commande
`--list-profiles` liste uniquement les profils embarqués.

## 4. Un seul héritage pour les paramètres de méthodes

[default.yaml](../config/profiles/default.yaml) contient toutes les valeurs
par défaut. Les profils d'actifs ne décrivent que leurs différences.
Le chargeur garde ces deux couches séparées : `profil.methodes` contient
uniquement les surcharges utilisateur, et `profil.method_defaults` les défauts.
Le résolveur commun au CLI et à Python les applique au timeframe demandé.

L'ordre, du moins prioritaire au plus prioritaire, est :

1. défaut général YAML ;
2. défaut YAML pour le timeframe ;
3. surcharge générale du profil ;
4. surcharge du profil pour le timeframe ;
5. argument explicite Python ou `--method-params`.

Ainsi `shannon.fenetre: 100` dans un profil remplace aussi la valeur M1 de
200 héritée des défauts. Une surcharge M1 dans ce même profil peut ensuite
préciser une autre valeur. Les paramètres marqués `per_timeframe=False`
dans leur schéma sont interdits sous `timeframes`.

Les sections générales (`donnees`, `context`, `segmentation`) conservent
leur fusion récursive historique : dictionnaires fusionnés, listes et
scalaires remplacés. Cette fusion n'est jamais appliquée à `methodes`.
Les chemins des ressources de contexte gardent leur convention existante,
décrite dans [le guide du contexte](./context.md).

Un chemin utilisateur nommé `default.yaml` est traité comme tout autre
profil utilisateur. Seul le nom court `default` ou le chemin du fichier
embarqué désigne le profil de référence.

### Valeurs absentes et calcul automatique

Une clé absente hérite de la couche précédente. Dans les profils YAML,
`null` est accepté uniquement pour `min_periods` (ATR/Kaufman) et
`min_periods_c` (Minkowski) : le schéma calcule alors la moitié entière de
la fenêtre **effective**, avec un minimum de 1. Les défauts utilisent cette
règle, sans recopier les demi-fenêtres dans chaque timeframe.
Une valeur numérique explicite de chauffe reste fixe ; elle doit être
inférieure ou égale à la fenêtre.

Pour compatibilité, un argument Python `None` ou JSON `null` dans
`--method-params` signifie « argument absent ». Pour réactiver une chauffe
automatique après une valeur fixe du profil, mettre `null` dans le profil.
Les arguments CLI sont scalaires ; utiliser un profil pour les surcharges
par timeframe. Les clés YAML dupliquées et les fusions YAML `<<` sont
refusées : l'héritage appartient au résolveur de profils.

## 5. API Python, inspection et migration

```python
from regime_lib.config import load_profile
from regime_lib.methods.trend.adx import ADXDetector

profil = load_profile("eurusd")
detecteur = ADXDetector(
    timeframe="H1", profile=profil.for_method("adx"), n_adx=20,
)
print(detecteur.params)         # valeurs effectives, timeframe inclus
print(detecteur.param_sources)  # provenance de chaque paramètre de méthode
```

L'instanciation directe `ADXDetector(timeframe="H1")` utilise le même
résolveur et les mêmes défauts. Un dictionnaire `profile={...}` reste
accepté comme couche de surcharges brute.

Chaque profil chargé capture les défauts et les surcharges à cet instant.
Les appels `profil.for_method(...)` fournissent des copies isolées. Une
modification du fichier n'affecte pas les instances ou profils existants :
appeler de nouveau `load_profile(...)` pour recharger. Une nouvelle
instanciation directe sans profil lit les défauts actuels ; aucun cache
global ne fige le YAML. Un fichier manquant, corrompu ou incomplet produit
une erreur, même si un argument explicite aurait fourni la valeur absente.

Afficher la configuration sans charger de données :

```bash
regime-lib --show-config --profile eurusd --timeframe M1,H1 \
  --methods adx,shannon --method-params '{"adx": {"n_adx": 20}}'
```

La sortie JSON expose, par timeframe et méthode, `params`, `sources` et
`params_hash`. Ces informations sont aussi exportées sous `effective_config`
dans `run_meta.json`. Le hash de méthode porte sur les valeurs effectives,
indépendamment de leur provenance ; le hash du profil couvre les deux couches.

**Migration :** remplacer les accès aux sous-profils supposés fusionnés
(`profil.methodes["adx"]`) par `profil.for_method("adx")` lors de la
construction. Pour lire les valeurs finales, consulter `detecteur.params`.
`load_profile("default").methodes` est désormais vide puisqu'il ne porte
aucune surcharge. Les constantes de secours des détecteurs et leur ancienne
méthode `_resolve` ont été supprimées. Les anciens hashes ne sont pas
comparables aux nouveaux hashes effectifs.

Le profil EURUSD conserve sa surcharge générale Shannon `fenetre: 100` :
elle s'applique désormais à tous les timeframes. Pour conserver une fenêtre
M1/M5 de 200, l'indiquer explicitement sous `timeframes` dans ce profil.

## 6. Paramètres des méthodes

Cette section décrit les clés reconnues sous `methodes` dans
[`profiles/default.yaml`](../config/profiles/default.yaml). Les valeurs
ci-dessous sont les calibrations du profil par défaut, pas des valeurs
universelles pour tous les actifs. Les valeurs codées en secours par les
constructeurs peuvent différer lorsque le profil n'est pas fourni.

Un paramètre indiqué « par timeframe » peut être placé soit directement
sous la méthode (valeur commune à tous les timeframes), soit dans
`timeframes.<TF>` (valeur spécifique). La valeur spécifique au timeframe
est prioritaire sur la valeur commune. Les paramètres non présents dans
un profil personnalisé continuent d'être hérités de `default.yaml`.

### `atr_volatility`

Les seuils de volatilité sont des quantiles glissants de l'ATR précédent ;
la direction est évaluée séparément.

| Paramètre | Défaut | Portée et rôle |
|---|---:|---|
| `n_direction` | `20` | Retard en barres pour déterminer le signe du mouvement `close[t] - close[t-n_direction]`. |
| `q_low` | `0.33` | Quantile inférieur de l'ATR : en dessous, le régime est `RANGE`. |
| `q_high` | `0.67` | Quantile supérieur de l'ATR : au-dessus, le régime peut être une tendance directionnelle. |
| `n_atr` | Par timeframe, voir ci-dessous | Période de calcul de l'ATR. |
| `fenetre` | Par timeframe, voir ci-dessous | Taille de l'historique glissant servant à calculer les seuils de quantile de l'ATR. |
| `min_periods` | Par timeframe, voir ci-dessous | Nombre minimal d'observations valides dans la fenêtre avant de calculer les seuils. |

Valeurs `n_atr` / `fenetre` / `min_periods` de `default.yaml` :

| Timeframe | `n_atr` | `fenetre` | `min_periods` |
|---|---:|---:|---:|
| M1 | 20 | 30 000 | 15 000 |
| M5 | 20 | 6 000 | 3 000 |
| M15 | 14 | 2 000 | 1 000 |
| M30 | 14 | 1 000 | 500 |
| H1 | 14 | 500 | 250 |
| H4 | 14 | 125 | 62 |
| D1 | 14 | 21 | 10 |
| W1 | 14 | 12 | 6 |

### `er_kaufman`

L'ER mesure l'efficience du mouvement : proche de 1, le prix progresse de
façon directionnelle ; proche de 0, il est davantage bruité. Les bornes de
régime sont des quantiles glissants de l'ER précédent.

| Paramètre | Défaut | Portée et rôle |
|---|---:|---|
| `q_chop` | `0.40` | Quantile inférieur de l'ER, borne du régime `CHOP`. |
| `q_tendance` | `0.60` | Quantile supérieur de l'ER, à partir duquel le mouvement peut être classé en tendance. |
| `n_er` | `20` | Par timeframe : période de calcul de l'Efficiency Ratio de Kaufman. |
| `fenetre` | Par timeframe, voir ci-dessous | Taille de l'historique glissant des quantiles de l'ER. |
| `min_periods` | Par timeframe, voir ci-dessous | Nombre minimal d'observations valides avant le calcul des quantiles. |

Valeurs `fenetre` / `min_periods` de `default.yaml` :

| Timeframe | `fenetre` | `min_periods` |
|---|---:|---:|
| M1 | 8 000 | 4 000 |
| M5 | 3 000 | 1 500 |
| M15 | 2 000 | 1 000 |
| M30 | 1 500 | 750 |
| H1 | 700 | 350 |
| H4 | 200 | 100 |
| D1 | 40 | 20 |
| W1 | 20 | 10 |

### `adx`

L'ADX mesure la force d'une tendance ; la direction haussière ou baissière
est déterminée à partir des indicateurs directionnels.

| Paramètre | Défaut | Rôle |
|---|---:|---|
| `n_adx` | `14` | Période de l'ADX de Wilder. |
| `seuil_range` | `20.0` | En dessous de cette valeur, le marché est classé `RANGE` (absence de tendance). |
| `seuil_tendance` | `25.0` | À partir de cette valeur, l'ADX indique une tendance ; entre les seuils, le régime est `CHOP`. La direction vient des indicateurs directionnels. |

### `hmm_gaussian`

Le HMM gaussien est une méthode hors ligne qui utilise du lookahead
structurel. Son exécution nécessite l'autorisation explicite du lookahead.

| Paramètre | Défaut | Rôle |
|---|---:|---|
| `n_states` | `2` | Nombre d'états latents à estimer ; détermine aussi les labels `STATE_0`, `STATE_1`, etc. |
| `n_iter` | `100` | Nombre maximal d'itérations de l'algorithme d'ajustement. |
| `seed` | `42` | Graine entière pour rendre l'initialisation reproductible. |

### `price_action`

Méthode descriptive hors ligne : une fractale n'est confirmée qu'après
observation des barres suivantes. Elle nécessite donc l'autorisation
explicite du lookahead et n'est pas destinée à produire un signal temps réel.

| Paramètre | Défaut | Rôle |
|---|---:|---|
| `n_fractale` | `5` | Nombre de barres de chaque côté nécessaires pour confirmer un sommet ou un creux fractal. |
| `k_atr` | `0.5` | Multiplicateur de l'ATR définissant la tolérance de proximité entre pivots considérés quasi-plats. |
| `n_atr_tol` | `14` | Période de l'ATR utilisé pour calculer cette tolérance. |

### `mss_choch`

Méthode de structure de marché hors ligne : la confirmation des fractales
emploie des barres futures. L'autorisation explicite du lookahead est
requise.

| Paramètre | Défaut | Portée et rôle |
|---|---:|---|
| `n_fractale` | `5` | Par timeframe : nombre de barres de chaque côté pour confirmer un swing high ou swing low. Une valeur plus grande donne moins de pivots, généralement plus sélectifs. |

### `shannon`

| Paramètre | Défaut | Portée et rôle |
|---|---:|---|
| `fenetre` | M1 `200`, M5 `200`, M15 `150`, H1 `100`, H4 `50`, D1 `30` | Nombre de rendements de la fenêtre glissante utilisée pour estimer l'entropie. |
| `n_bins` | `10` | Nombre de classes utilisées pour discrétiser les rendements dans l'histogramme. |
| `seuil_ordonne` | `0.65` | En dessous de ce seuil d'entropie normalisée, le régime est `ORDONNE`. |
| `seuil_desordre` | `0.78` | Au-dessus de ce seuil, le régime est `DESORDRE` ; entre les deux, `NEUTRE`. |

### `pca_axis`

| Paramètre | Défaut | Portée et rôle |
|---|---:|---|
| `fenetre` | M1 `200`, M5 `200`, M15 `150`, H1 `100`, H4 `50`, D1 `30` | Nombre d'observations passées de la trajectoire utilisées par l'ACP glissante. |
| `seuil_direction` | `0.3` | Valeur absolue minimale de la direction signée pour classer `HAUSSIER` ou `BAISSIER`. |
| `seuil_eigen_ratio` | `0.65` | Concentration minimale de variance sur la première composante principale pour accepter une classification directionnelle. |

### `minkowski_causal`

| Paramètre | Défaut | Portée et rôle |
|---|---:|---|
| `fenetre_c` | `500` | Fenêtre glissante utilisée pour estimer la vitesse limite du marché `c_mkt`. |
| `min_periods_c` | `250` (`fenetre_c // 2`) | Nombre minimal d'observations valides pour calculer `c_mkt`. |
| `q_c` | `0.90` | Quantile des rendements absolus passés utilisé pour estimer `c_mkt`. |
| `seuil_lightlike` | `0.01` | Tolérance relative pour classer un intervalle `LIGHTLIKE` près de la frontière, selon `|Δs²| < seuil_lightlike · c_mkt²`. |

### `ofi`

Cette méthode requiert les colonnes `bid_volume` et `ask_volume`, produites
par l'agrégation des ticks. Le volume Dukascopy représente une profondeur
observée, et non un volume de transactions ; voir [le guide Volume](./volume.md).

| Paramètre | Défaut | Portée et rôle |
|---|---:|---|
| `fenetre` | `20` (D1 `10`) | Par timeframe : longueur de la fenêtre glissante pour calculer l'imbalance. |
| `seuil_pression` | `0.1` | Seuil absolu de l'OFI pour distinguer `PRESSION_ACHETEUSE` / `PRESSION_VENDEUSE` d'`EQUILIBRE`. |

### `divergence_pv`

Cette méthode requiert `close` et `volume`. Elle compare les nouveaux
extrêmes de prix au volume observé ; ses résultats sont une hypothèse à
évaluer sur les données, pas une règle universelle.

| Paramètre | Défaut | Portée et rôle |
|---|---:|---|
| `fenetre_extreme` | `20` | Par timeframe : nombre de barres utilisé pour détecter un nouveau plus haut ou plus bas. |
| `fenetre_median_vol` | `50` (D1 `30`) | Par timeframe : fenêtre de calcul de la médiane glissante du volume de référence. |
| `ratio_volume_fort` | `1.5` | Le volume est considéré fort lorsqu'il dépasse ce multiple de sa médiane glissante. |

### `segmentation`

Ces paramètres sont à la racine du profil, sous `segmentation`, et non
sous `methodes`. Ils configurent la segmentation PELT décrite dans le
module de projection.

| Paramètre | Défaut | Rôle |
|---|---:|---|
| `modele` | `normal` | Modèle de coût de ruptures fourni à PELT (`l2`, `normal` ou `rbf`). |
| `pen` | `50.0` dans `default.yaml` | Pénalité de complexité : une valeur plus élevée produit généralement moins de segments. Doit être strictement positive. |
| `min_size` | `20` | Nombre minimal de barres dans un segment. |

Les fenêtres et périodes sont exprimées en **nombre de barres**, pas en
durée calendaire. Ainsi une fenêtre de `20` vaut 20 barres du timeframe
choisi.

## 7. Limites et traçabilité

Les options `donnees.source` et `donnees.timeframes` documentent le profil,
mais ne remplacent pas le fichier source ou l'option CLI `--timeframe`.
L'exécution CLI nécessite `--url`, `--timeframe`, `--methods` et
`--output`. Le mode `--show-config` exige seulement `--timeframe` et `--methods`.

L'agrégation tick vers M1 est actuellement détectée automatiquement par le
chargeur et utilise le Bid par défaut. `donnees.tick_aggregation`, bien que
représentable dans le schéma, ne pilote pas encore l'algorithme
d'agrégation. Voir le [contrat d'entrée/sortie](./output_format.md) pour le
détail des champs tick enrichis et des conventions de frontière, et
[le guide du contexte temporel](./context.md) pour la configuration `context`.

Le nom, le type, le chemin du profil et son hash sont consignés dans
`run_meta.json` sous `profile_name`, `profile_type`, `profile_path` et
`profile_hash`. Les paramètres passés via `--method-params` sont aussi
tracés séparément.
