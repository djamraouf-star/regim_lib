# Format de Sortie Uniforme

Ce document spécifie le format des données en sortie des détecteurs de régime et du pipeline d'export de `regime_lib`.

---

## 1. Contrat d'entrée des données

Toute méthode reçoit un DataFrame OHLCV validé selon les critères suivants :
- **Colonnes obligatoires** : `open`, `high`, `low`, `close`, `volume` (en minuscules).
- **Index** : `DatetimeIndex` nommé `timestamp`, trié par ordre strictement croissant, sans doublons.
- **Convention de resampling** : `label='right', closed='right'` (chaque barre temporelle est horodatée par son instant de clôture).
- **Barres incomplètes** : La colonne booléenne `is_partial` indique si une barre temporelle n'était pas entièrement couverte par les données sources (typiquement la dernière barre de la série).

Le chargeur accepte aussi les Parquets de cotations tick contenant un
timestamp (`Timestamp`, `Datetime`, `Date` ou `Time`) et les colonnes
`bidPrice`, `askPrice`, `bidVolume`, `askVolume` (alias normalisés sans
tenir compte de la casse, des espaces et des séparateurs usuels). Les
horodatages sans fuseau sont interprétés en UTC et les ticks sont triés de
façon stable. Les OHLC M1 utilisent le Bid par défaut ; l'API permet aussi
de choisir l'Ask ou le mid. Le volume agrégé est la somme de `bidVolume` et
`askVolume`, qui sont des tailles de cotation et non un volume de
transactions.

Les barres tick comportent en plus `spread_mean`, `spread_max`,
`spread_close`, `ask_close`, `tick_count` et `is_partial`. Une barre
étiquetée T couvre `[T - période, T)` : un tick situé exactement sur T
appartient à la barre qui commence à T. La première et la dernière barre
produites à partir des ticks sont marquées partielles. Le resampling
OHLCV vers un timeframe supérieur utilise `label='right', closed='right'`,
agrège les seules colonnes OHLCV et ajoute `is_partial` pour la dernière
barre potentiellement incomplète.

---

## 2. Schéma tabulaire uniforme

Le DataFrame produit par [`to_uniform`](../core/output.py) et enregistré
dans le fichier Parquet respecte un ordre de colonnes strict.

### Colonnes obligatoires (en préfixe, dans cet ordre)

| Colonne | Type | Description |
|---|---|---|
| `timestamp` | Datetime | Instant de clôture de la barre (matérialisé en colonne via `reset_index()`). |
| `regime` | `str` | Libellé textuel du régime détecté (ex. `TENDANCE_HAUSSIERE`, `RANGE`, `STATE_0`). |
| `regime_id` | `int` | Identifiant numérique stable et bijectif, dérivé du `REGIME_MAP` de la méthode. |
| `confidence` | `float` | Niveau de confiance dans l'intervalle $[0.0, 1.0]$. `NaN` strictement interdit. Vaut $0.0$ si `regime == "INCONNU"`. |
| `method` | `str` | Nom court snake_case de la méthode (ex. `atr_volatility`, `hmm_gaussian`). |
| `timeframe` | `str` | Timeframe analysé (`M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`, `W1`). |
| `asset` | `str` | Symbole de l'actif financier (ex. `EURUSD`). |

### Colonnes optionnelles autorisées

| Colonne | Type | Description |
|---|---|---|
| `is_partial` | `bool` | `True` si la barre est incomplète ou non clôturée (`regime` alors forcé à `INCONNU` et `confidence` à $0.0$). |
| `params_hash` | `str` | Hash SHA256 court (8 caractères hexadécimaux) des hyperparamètres utilisés pour ce run. |

---

## 3. Formats d'export du pipeline CLI

Chaque exécution via le CLI [`regime-lib`](./cli.md) génère deux fichiers dans le répertoire de sortie :

### 1. `regimes.parquet` (Format Long)
- Toutes les détections de toutes les méthodes et de tous les timeframes demandés sont empilées verticalement (`pd.concat(..., ignore_index=True)`).
- L'index de sortie est un `RangeIndex`.
- Le découpage et les analyses par méthode, timeframe ou période s'effectuent par simple filtrage ou pivot pandas.

### 2. `run_meta.json` (Sidecar de traçabilité)
Fichier JSON garantissant la reproductibilité complète du run :

```json
{
  "url": "data/EURUSD_M1.parquet",
  "source_type": "barres",
  "timeframes": ["M1", "H1"],
  "asset": "EURUSD",
  "methods": ["atr_volatility", "adx"],
  "allow_lookahead": false,
  "allow_lookahead_override": {},
  "method_params": {},
  "methodes_avec_lookahead": [],
  "run_descriptif": false,
  "run_at": "2026-09-28T20:30:00.000000+00:00",
  "n_rows_input": 384212,
  "n_rows_par_timeframe": {
    "M1": 384212,
    "H1": 6567
  },
  "n_lignes_total": 781558
}
```
