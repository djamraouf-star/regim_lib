# Référence de la Ligne de Commande (CLI)

`regime_lib` fournit deux points d'entrée en ligne de commande :
1. **`regime-lib`** (ou `python -m regime_lib`) : exécute le pipeline de chargement, calcul des régimes et export des résultats.
2. **`regime-report`** (ou `python -m regime_lib.utils.report`) : génère un rapport statistique complet d'analyse d'un run.

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
`load_parquet` ou `agreger_tick`. Le chargement par lots est disponible pour
les fichiers locaux via `load_parquet(..., batch_size=...)` ; le CLI ne
propose pas actuellement ces paramètres.

### Options disponibles

| Option | Requis | Défaut | Description |
|---|---|---|---|
| `--url` | Oui* | — | Chemin local ou URL distante vers un Parquet OHLCV ou un flux de cotations tick. |
| `--timeframe` | Oui* | — | Timeframe(s) cible(s). Valeur unique (`H1`) ou liste séparée par des virgules (`M1,M5,M15,H1`). |
| `--methods` | Oui* | — | Liste de méthodes séparées par des virgules, dont `atr_volatility`, `er_kaufman`, `adx`, `price_action`, `mss_choch` et `hmm_gaussian`. |
| `--asset` | Non | Inféré | Symbole de l'actif. Si omis, il est déduit du nom de fichier (ex. `EURUSD_M1.parquet` $\to$ `EURUSD`). |
| `--profile` | Non | `default` | Nom court d'un profil embarqué (`default`, `eurusd`, ...) ou chemin vers un fichier YAML. Le profil est fusionné avec `default.yaml`; les surcharges `--method-params` restent prioritaires. |
| `--list-profiles` | Non | `false` | Liste les profils embarqués disponibles puis quitte. Aucun argument de pipeline requis dans ce mode. |
| `--allow-lookahead` | Non | `false` | Autorise globalement le lookahead pour toutes les méthodes. |
| `--allow-lookahead-override` | Non | `{}` | Autorisation ciblée par méthode, ex. `hmm_gaussian:true,price_action:true,mss_choch:true`. |
| `--method-params` | Non | `{}` | Surcharges d'hyperparamètres au format JSON, ex. `'{"atr_volatility": {"n_atr": 20}}'`. |
| `--output` | Oui* | — | Répertoire de destination des fichiers exportés. |

\* En mode exécution normale. Avec `--list-profiles`, les arguments `--url`, `--timeframe`, `--methods` et `--output` ne sont pas requis.

### Profils de configuration

`--profile` accepte :
- un **nom court** résolu dans `regime_lib/config/profiles/`, ex. `--profile eurusd` ;
- un **chemin YAML explicite**, ex. `--profile config/profils/gbpusd.yaml`.

Tout profil autre que `default` est **fusionné profondément** avec `regime_lib/config/profiles/default.yaml` : il ne redéfinit que les valeurs qu'il change. Pendant le run, le profil chargé est tracé sur la sortie d'erreur avec son hash, puis exporté dans `run_meta.json` via les champs `profile_name`, `profile_type`, `profile_hash` et `profile_path`.

Voir le [guide des profils](./config.md) pour le schéma YAML complet,
l'ordre de priorité des paramètres et des exemples de profils personnalisés.

### Fichiers produits

Dans le répertoire spécifié par `--output` :
- **`regimes.parquet`** : table longue empilant l'ensemble des détections pour tous les timeframes et méthodes demandés.
- **`run_meta.json`** : sidecar JSON traçant la configuration complète du run, la date d'exécution UTC, la liste des timeframes, le nombre de lignes, les informations du profil utilisé et l'éventuel caractère descriptif (`run_descriptif: true`).

---

## 2. Générateur de rapport : `regime-report`

### Exemples d'utilisation

**Rapport sur un timeframe précis :**
```bash
python -m regime_lib.utils.report \
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
