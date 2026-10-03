# Actifs — données locales

Les jeux de données ne sont pas inclus dans le dépôt. Placez ici vos données
locales, dans un sous-dossier par actif ; les fichiers Parquet sont ignorés
par Git.

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

Les fichiers Parquet sous ce dossier ne sont pas versionnés car ils peuvent
être volumineux. Les noms d'actifs (majuscules) restent distincts des noms de
profils YAML (minuscules).
