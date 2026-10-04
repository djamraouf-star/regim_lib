# regime_lib

Bibliothèque Python de détection de régimes de marché à partir de données
OHLCV. Chaque méthode produit une sortie tabulaire commune, utilisable
indépendamment ou combinée en aval avec pandas.

**Copyright (C) 2026 Djamal Raouf**  
Ce projet est distribué sous la licence **GNU GPL version 3 ou toute version
ultérieure** (SPDX : `GPL-3.0-or-later`). Voir [LICENSE](./LICENSE).

## Usage et limites

`regime_lib` caractérise les mouvements des indices financiers et compare les
familles d'indicateurs en projetant leurs représentations dans l'espace du prix.
Les cinq dimensions de comparaison sont l'amplitude, la régularité du trajet,
la durée, la fréquence d'apparition et l'excursion opposée. L'API ne constitue
ni une stratégie de trading ni un moteur de backtest.

Le [contrat commun de projection et de mesure v1.0.0](./regime_lib/doc/protocole_projection_mesure.md)
fixe la direction du développement, les conventions et les exemples numériques
de référence. Il décrit les garanties à implémenter, pas une conformité déjà
acquise du code actuel.

## Fonctionnalités

- Détecteurs organisés par familles et enregistrés dans un registre commun.
- Profils YAML pour configurer les paramètres par méthode et timeframe.
- Traitement de plusieurs timeframes dans un même run.
- Protection contre le lookahead, désactivé par défaut et activable
  explicitement pour les méthodes offline.
- Exports Parquet et métadonnées JSON pour faciliter la traçabilité.

## Installation

Python 3.10 ou supérieur est requis.

Depuis la racine du projet :

```bash
python -m pip install -e .
```

Pour installer aussi les dépendances de test :

```bash
python -m pip install -e ".[dev]"
```

## Données d'entrée

Le pipeline accepte des barres OHLCV ou des cotations tick dans un fichier
Parquet. Les noms de colonnes sont normalisés (casse et espaces).

- **Barres OHLCV** : colonnes `open`, `high`, `low`, `close`, `volume`, avec
  un `DatetimeIndex` ou une colonne de timestamp (`timestamp`, `datetime`,
  `date` ou `time`).
- **Ticks de cotation** : timestamp et colonnes `Bid Price`, `Ask Price`,
  `Bid Volume`, `Ask Volume` (alias et variations de casse/espacement
  reconnus). Les horodatages sans fuseau sont interprétés en UTC et les ticks
  sont triés de façon stable. L'OHLC utilise par défaut le Bid ; les options
  Python `prix="ask"` et `prix="mid"` sélectionnent respectivement l'Ask ou
  le milieu du spread. Le volume des barres est la somme des tailles Bid et
  Ask affichées ; ce n'est pas un volume de transactions.

Les barres tick incluent aussi `spread_mean`, `spread_max`, `spread_close`,
`ask_close`, `tick_count` et `is_partial`. Une barre M1 étiquetée `T` couvre
`[T - 1 min, T)` : un tick exactement sur la frontière `T` appartient à la
barre suivante. La première et la dernière barre d'une agrégation tick sont
marquées partielles. Le resampling conserve ces marqueurs et marque les agrégats
dont le nombre de barres source est insuffisant, y compris au début de la série.
Il publie `source_count`, `expected_count` et leur rapport `coverage` ; un agrégat
peut avoir une couverture de 100 % tout en contenant une barre source partielle.
Une dernière barre complète n'est pas marquée partielle artificiellement.

Les ticks sont agrégés en barres M1 avant le resampling vers le ou les
timeframes demandés. Le chargement d'un Parquet local peut être traité par
lots par l'API Python (`load_parquet(..., batch_size=...)`). Si `--asset`
n'est pas fourni, le symbole est inféré du nom de fichier.

Timeframes pris en charge : `M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`,
`W1`.

Les conventions d'alignement, les nouveaux paramètres de sélection et les
exports de couverture sont décrits dans le [guide de l'étape 2](./regime_lib/doc/alignement_validite.md).

## Utilisation rapide

Exécuter des méthodes causales sur un timeframe :

```bash
regime-lib \
  --url data/EURUSD_M1.parquet \
  --timeframe H1 \
  --methods atr_volatility,er_kaufman,adx \
  --output results
```

`python -m regime_lib` peut être utilisé à la place de `regime-lib`.
Le run écrit `regimes.parquet` au format long et un sidecar
`run_meta.json` dans le répertoire de sortie.

Les méthodes `price_action`, `mss_choch` et `hmm_gaussian` nécessitent un
lookahead structurel et sont destinées à l'analyse offline. Leur activation
doit être explicite, par exemple :

```bash
regime-lib \
  --url data/EURUSD_M1.parquet \
  --timeframe H1 \
  --methods hmm_gaussian \
  --allow-lookahead-override hmm_gaussian:true \
  --output results_hmm
```

Voir les profils embarqués avec :

```bash
regime-lib --list-profiles
```

Un profil peut être choisi avec `--profile`, et les paramètres d'une
méthode peuvent être surchargés avec `--method-params` (JSON). Utiliser
`regime-lib --help` pour la liste complète des options.

## Méthodes disponibles

- `atr_volatility`
- `er_kaufman`
- `adx`
- `price_action` (offline, lookahead requis)
- `mss_choch` (offline, lookahead requis)
- `hmm_gaussian` (offline, lookahead requis)
- `minkowski_causal`

## Documentation

- [Décisions de conception](./regime_lib/doc/design.md)
- [Format de sortie](./regime_lib/doc/output_format.md)
- [Guide d'ajout d'une méthode](./regime_lib/doc/adding_method.md)
- [Configuration par profils](./regime_lib/doc/config.md)
- [Choix de conception de PCA axis](./regime_lib/doc/pca_axis.md)
- [Méthode MSS / CHOCH](./regime_lib/doc/mss_choch.md)
- [Module d'étude](./regime_lib/doc/study.md)
- [Référence CLI et rapport](./regime_lib/doc/cli.md)
- [Notes de recherche](./regime_lib/doc/research/index.md)
- [Documentation complète](./regime_lib/doc/index.md)

## Licence

Copyright (C) 2026 Djamal Raouf. Ce programme est un logiciel
libre : vous pouvez le redistribuer et/ou le modifier selon les termes de
la GNU General Public License telle que publiée par la Free Software
Foundation, version 3 ou toute version ultérieure, selon votre choix. Le texte
intégral est disponible dans
[LICENSE](./LICENSE).
