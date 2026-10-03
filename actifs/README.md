# Actifs — données brutes

Ce dossier contient les **données brutes** des actifs analysés avec
`regime_lib`. Chaque actif a son propre sous-dossier.

## Convention de rangement


### Nommage des actifs

**Majuscules** (`EURUSD`, `XAUUSD`, `DXY`) — différencie visuellement
des profils YAML en minuscules (`eurusd.yaml`).

### Nommage des fichiers

Format `<ACTIF>_<type>_<période>.parquet` :

- `EURUSD_Tick_20250623_20261001.parquet`
- `XAUUSD_H1_20250623_20261001.parquet`
- `DXY_Tick_20250623_20261001.parquet`

Période au format `YYYYMMDD_YYYYMMDD` (début_fin).

## Versionnement

Ce dossier **n'est pas versionné** dans git (fichiers trop volumineux).

Le fichier `.gitignore` à la racine doit contenir :

