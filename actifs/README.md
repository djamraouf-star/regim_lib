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

## Télécharger les ticks

Depuis la racine du projet, installer `pip install -e '.[download]'`, puis :

```bash
python download_dukascopy.py --instrument EURUSD --start 2024-01-01 --end 2025-01-01
```

Le [script](../download_dukascopy.py) appelle `dukascopy_python.fetch` en mode
tick et écrit un Parquet sous `actifs/EURUSD/`. Les dates sont en UTC : début
inclus, fin exclue. `--output` permet de choisir un autre fichier.
Les prix et volumes Bid/Ask sont conservés tels que fournis par la bibliothèque,
ainsi que les ticks partageant le même horodatage. Les volumes de la bibliothèque
sont exprimés en millions ; aucune conversion supplémentaire n'est appliquée.
La lecture est découpée par journées ; le fichier final n'est publié qu'en cas
de succès et un fichier existant n'est jamais écrasé. Les journées sans ticks sont signalées dans les
logs ; leur absence ne prouve pas à elle seule que le marché était fermé.

### Cache et reprise

Chaque lot non vide est conservé dans `.cache/dukascopy/` après validation,
avec une écriture atomique. Ce dossier est ignoré par Git. Après une erreur ou
une interruption, relancer la même commande réutilise les lots terminés et
télécharge ceux qui manquent. Le cache reste disponible après un succès.
`--cache-dir /chemin/du/cache` permet de choisir un autre emplacement.

Les lots sont découpés à minuit UTC. La clé distingue l'instrument, les bornes
exactes, le mode tick et la version de la bibliothèque. Une journée partielle
est donc distincte d'une journée complète. À chaque lecture, les métadonnées
et la cohérence des ticks sont vérifiées ; un cache illisible ou invalide est
retéléchargé. Les réponses vides et les périodes dont la fin est encore dans
le futur ne sont pas mises en cache. Les contrôles ne garantissent pas la
complétude des cotations fournies par le serveur.

API utilisée : [dukascopy-python](https://pypi.org/project/dukascopy-python/).

## Versionnement des données

Les fichiers Parquet sous ce dossier ne sont pas versionnés car ils peuvent
être volumineux. Les noms d'actifs (majuscules) restent distincts des noms de
profils YAML (minuscules).
