# Référence de la Ligne de Commande (CLI)

`regime_lib` fournit deux points d'entrée en ligne de commande :
1. **`regime-lib`** (ou `python -m regime_lib`) : exécute le pipeline de chargement, calcul des régimes et export des résultats.
2. **`regime-report`** (ou `python -m regime_lib.study.report`) : génère un rapport statistique complet d'analyse d'un run.

---

## 1. Commande principale : `regime-lib`

### Exemples d'utilisation

**Exécution simple (mono-timeframe causale) :**
```bash
python -m regime_lib \
    --url "data/EURUSD_M1.parquet" \
    --timeframe H1 \
    --methods atr_volatility,er_kaufman,adx \
    --output results_causal
```

**Exécution multi-timeframes avec méthode offline :**
```bash
python -m regime_lib \
    --url "data/EURUSD_M1.parquet" \
    --timeframe M1,M5,M15,H1 \
    --methods atr_volatility,er_kaufman,hmm_gaussian \
    --allow-lookahead-override hmm_gaussian:true \
    --method-params '{"hmm_gaussian": {"n_states": 2, "n_iter": 100}}' \
    --output results_multi
```

**Détection MSS / CHOCH (offline, lookahead requis) :**
```bash
python -m regime_lib \
    --url "data/EURUSD_M1.parquet" \
    --timeframe H1 \
    --methods mss_choch \
    --allow-lookahead-override mss_choch:true \
    --output results_mss_choch
```

**Exécution avec un profil de configuration :**
```bash
python -m regime_lib \
    --url "data/EURUSD_M1.parquet" \
    --timeframe H1 \
    --methods adx \
    --profile eurusd \
    --output results_eurusd
```

**Lister les profils embarqués disponibles :**
```bash
python -m regime_lib --list-profiles
```

**Lister les méthodes actuellement enregistrées :**
```bash
python -m regime_lib --list-methods
```
La liste provient du registre chargé par les modules de méthodes ; elle
constitue l'inventaire à jour, sans liste parallèle à maintenir dans le README.

**Exécuter depuis un Parquet de cotations tick :**
```bash
python -m regime_lib \
    --url "data/XAUUSD_ticks.parquet" \
    --timeframe M1,H1 \
    --methods atr_volatility,adx \
    --output results_ticks
```

Le chargeur détecte les Parquets tick contenant un timestamp et un prix
Bid ou Ask, puis exige les champs `bidPrice`, `askPrice`, `bidVolume` et
`askVolume` (les alias de casse et d'espacement usuels sont normalisés).
Les timestamps sans fuseau sont interprétés en UTC. L'OHLC M1 utilise le
Bid par défaut via le pipeline CLI ; le volume additionne les tailles de
cotation Bid et Ask et ne représente pas un volume de transactions. Les
barres tick contiennent également les statistiques de spread, `ask_close`,
`tick_count` et `is_partial`. Une barre étiquetée T couvre `[T - période, T)`,
et un tick à la frontière T est affecté à la barre suivante. La première et
la dernière barre M1 sont marquées partielles.

L'API Python permet de sélectionner `prix="bid"`, `"ask"` ou `"mid"` avec
`load_parquet` ou `agreger_tick`. La lecture par lots est disponible pour les
fichiers locaux via `load_parquet(..., batch_size=...)`, mais tous les lots
sont conservés puis concaténés avant agrégation. `batch_size` ne borne donc
pas la mémoire totale. Dimensionner sur la taille décompressée complète et
réserver une marge pour les copies intermédiaires et le résultat agrégé. Le
CLI ne propose pas actuellement ces paramètres.

### Options disponibles

| Option | Requis | Défaut | Description |
|---|---|---|---|
| `--url` | Oui* | — | Chemin local ou URL distante vers un Parquet OHLCV ou un flux de cotations tick. |
| `--timeframe` | Oui* | — | Timeframe(s) cible(s). Valeur unique (`H1`) ou liste séparée par des virgules (`M1,M5,M15,H1`). |
| `--methods` | Oui* | — | Liste de méthodes séparées par des virgules, dont `atr_volatility`, `er_kaufman`, `adx`, `price_action`, `mss_choch` et `hmm_gaussian`. |
| `--asset` | Non | Inféré | Symbole de l'actif. Si omis, il est déduit du nom de fichier (ex. `EURUSD_M1.parquet` $\to$ `EURUSD`). |
| `--show-config` | Non | `false` | Affiche les valeurs effectives, leur provenance et leur hash, sans données ni export ; nécessite méthodes et timeframes. |
| `--profile` | Non | `default` | Nom court d'un profil embarqué (`default`, `eurusd`, ...) ou chemin vers un fichier YAML. Le résolveur applique défauts et surcharges selon le [contrat de configuration](./config.md). |
| `--list-profiles` | Non | `false` | Liste les profils embarqués disponibles puis quitte. Aucun argument de pipeline requis dans ce mode. |
| `--list-methods` | Non | `false` | Liste les méthodes enregistrées puis quitte. Aucun argument de pipeline requis dans ce mode. |
| `--allow-lookahead` | Non | `false` | Autorise globalement le lookahead pour toutes les méthodes. |
| `--allow-lookahead-override` | Non | `{}` | Autorisation ciblée par méthode, ex. `hmm_gaussian:true,price_action:true,mss_choch:true`. |
| `--method-params` | Non | `{}` | Surcharges d'hyperparamètres au format JSON, ex. `'{"atr_volatility": {"n_atr": 20}}'`. |
| `--output` | Oui* | — | Répertoire de destination des fichiers exportés. |

\* En mode exécution normale. Avec `--list-profiles` ou `--list-methods`, les arguments `--url`, `--timeframe`, `--methods` et `--output` ne sont pas requis.

### Profils de configuration

`--profile` accepte :
- un **nom court** résolu dans `regime_lib/config/profiles/`, ex. `--profile eurusd` ;
- un **chemin YAML explicite**, ex. `--profile config/profils/gbpusd.yaml`.

Les paramètres de méthodes conservent séparément défauts et surcharges jusqu'à la résolution par timeframe. Les sections générales gardent leur fusion profonde. Voir [la configuration et sa migration](./config.md). Pendant le run, le profil chargé est tracé sur la sortie d'erreur avec son hash, puis exporté dans `run_meta.json` via les champs `profile_name`, `profile_type`, `profile_hash` et `profile_path`.

Voir le [guide des profils](./config.md) pour le schéma YAML complet,
l'ordre de priorité des paramètres et des exemples de profils personnalisés.

### Fichiers produits

Dans le répertoire spécifié par `--output` :
- **`regimes.parquet`** : table longue empilant l'ensemble des détections pour tous les timeframes et méthodes demandés.
- **`run_meta.json`** : sidecar JSON traçant la configuration complète du run, la date d'exécution UTC, la liste des timeframes, le nombre de lignes, les informations du profil utilisé et l'éventuel caractère descriptif (`run_descriptif: true`).

---

## 2. Générateur de rapport : `regime-report`

Ce rapport descriptif appartient au module `study` et conserve les labels
des régimes de toutes les méthodes. Pour l'évaluation prédictive des
features, voir le [module d'étude](./study.md).

### Exemples d'utilisation

**Rapport sur un timeframe précis :**
```bash
python -m regime_lib.study.report \
    --run-dir results_multi \
    --url "data/EURUSD_M1.parquet" \
    --method hmm_gaussian \
    --timeframe H1 \
    --output doc/rapport_hmm_h1.txt
```

### Options

| Option | Requis | Défaut | Description |
|---|---|---|---|
| `--run-dir` | Oui | — | Répertoire contenant `regimes.parquet`. |
| `--url` | Oui | — | Parquet source OHLCV (utilisé pour reconstruire les features). |
| `--method` | Non | Auto | Méthode à analyser (automatique si une seule méthode est présente dans le run). |
| `--timeframe` | Non | Tous | Timeframe cible. Si omis et que le run en contient plusieurs, un rapport est produit pour chacun. |
| `--output` | Non | stdout | Fichier texte de sortie (affiche sur la sortie standard par défaut). |

Les rapports acceptent aussi `--asset` (sélection des régimes),
`--configuration` (sélection du `params_hash`), `--ohlcv-asset` et
`--ohlcv-timeframe` (identité explicite des prix source).
Les incompatibilités d'identité et les configurations ambiguës sont refusées.
Voir les [contrôles communs des entrées](./study.md#73-contrôles-communs-des-entrées).
