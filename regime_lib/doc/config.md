# Configuration par profils

**Dernière mise à jour** : 2026-10-02
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
`description`) et trois sections de configuration :

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
enregistrés, par exemple `shannon` ou `pca_axis`. Les paramètres de méthode
sont principalement vérifiés par le constructeur de la méthode au moment de
son utilisation.

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

Utiliser l'extension `.yaml` pour un chemin de profil explicite. La commande
`--list-profiles` liste uniquement les profils embarqués.

## 4. Héritage et fusion

Tout profil autre que `default` est fusionné récursivement avec
`default.yaml`. Le profil ne redéfinit que les valeurs qu'il modifie :

- les dictionnaires sont fusionnés récursivement ;
- les scalaires et les listes du profil remplacent les valeurs par défaut ;
- les paramètres et timeframes omis restent hérités du profil par défaut.

Par exemple, un profil qui ne définit que
`methodes.adx.seuil_range: 22.0` conserve les autres paramètres ADX et ceux
des autres méthodes. Le profil `default` charge directement
`default.yaml`, sans fusion avec lui-même.

## 5. Priorité de résolution

Lorsqu'un détecteur résout un paramètre, l'ordre de priorité est :

1. paramètre explicite du constructeur ;
2. valeur correspondante dans le profil passé au détecteur ;
3. valeur de `default.yaml` ;
4. valeur de secours codée dans la méthode.

Dans le CLI, `--method-params` transmet les paramètres comme valeurs
explicites et prend donc priorité sur le profil :

```bash
regime-lib \
  --url data/EURUSD_M1.parquet \
  --timeframe H1 \
  --methods adx \
  --profile eurusd \
  --method-params '{"adx": {"n_adx": 20}}' \
  --output results_adx20
```

Pour les paramètres dépendant du timeframe, la méthode cherche d'abord la
valeur dans `methodes.<méthode>.timeframes.<timeframe>`, puis dans les
paramètres généraux de cette méthode.

## 6. Limites et traçabilité

Les options `donnees.source` et `donnees.timeframes` documentent le profil,
mais ne remplacent pas le fichier source ou l'option CLI `--timeframe`.
L'exécution CLI nécessite toujours `--url`, `--timeframe`, `--methods` et
`--output`.

L'agrégation tick vers M1 est actuellement détectée automatiquement par le
chargeur et utilise le Bid par défaut. `donnees.tick_aggregation`, bien que
représentable dans le schéma, ne pilote pas encore l'algorithme
d'agrégation. Voir le contrat d'entrée/sortie pour le détail des champs tick
enrichis et des conventions de frontière.

Le nom, le type, le chemin du profil et son hash sont consignés dans
`run_meta.json` sous `profile_name`, `profile_type`, `profile_path` et
`profile_hash`. Les paramètres passés via `--method-params` sont aussi
tracés séparément.
