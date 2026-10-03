# TODO — regime_lib

Ce document reflète l'état réel du projet au 2026-10-01.
Il sert de feuille de route active, sans garder les anciens éléments
qui ne correspondent plus à l'architecture actuelle.

## État actuel du projet

### Architecture active

- [x] Architecture à méthodes pluggées avec `RegimeDetector`.
- [x] Registry centralisé dans `regime_lib.core.registry`.
- [x] Sortie uniforme standardisée via `core/output.py`.
- [x] Gestion des profils YAML avec héritage implicite depuis `default.yaml`.
- [x] CLI orientée pipeline multi-timeframe + sidecar JSON de métadonnées.
- [x] Traitement des `lookahead` avec garde-fou explicite par méthode.
- [x] Reproductibilité via `params_hash` / hash de profil.
- [x] Chargement de Parquets de cotations tick, agrégation M1 en UTC,
  choix du côté de prix, statistiques de spread, tailles affichées et
  marquage des barres partielles.

### Ce qui est vrai aujourd'hui

- Le design actuel est une bibliothèque de détection de régimes by-method,
  pas une architecture de consensus multi-méthodes forcée.
- Le coût de maintenance dépend davantage de la qualité du contrat des
  méthodes que de la quantité d’outils ad hoc.
- Les décisions de conception doivent rester explicites et testées ; la
  TODO ne doit pas servir de journal technique “inactif”.

## Priorités immédiates

### 1) Nettoyage de la base de code

- [ ] Supprimer les artefacts résiduels et les fichiers de backup non utiles.
- [ ] Retirer les commentaires de legacy / blocs `# à venir` dans les modules actifs.
- [ ] Garder les références historiques uniquement dans les dossiers d’archive.

### 2) Stabiliser la couche de validation

- [ ] Ajouter les tests CLI de base (`_parse_methods`, `_parse_timeframes`, `_parse_lookahead_override`, `_parse_method_params`, `_str2bool`).
- [ ] Vérifier le comportement réel de `run(args)` sur un parquet temporaire.
- [ ] Valider les exports multi-timeframe et la présence des sidecars JSON.
- [ ] Vérifier les cas d'erreur de configuration et de lookahead.

### 3) Renforcer la robustesse des méthodes

- [ ] Ajouter des tests de robustesse sur les valeurs extrêmes OHLCV.
- [ ] Vérifier la normalisation des sorties sur les séries partielles.
- [ ] Contrôler les NaN et les bornes de `confidence` dans toutes les méthodes.

### 4) Documentation utile et maintenue

- [x] Préciser dans le README que le projet soutient la recherche exploratoire et n'est pas destiné au trading en production.
- [ ] Conserver la documentation technique `doc/` comme source de vérité du design, pas comme lieu de tâches mortes.
- [ ] Garder les fichiers d’archive uniquement pour l’historique, pas pour la roadmap active.

## Backlog de court terme

- [ ] Ajouter une utilité de format large (wide format) si cela devient une vraie nécessité de produit.
- [ ] Évaluer un éventuel mode d’analyse plus avancé, seulement si les besoins métier le justifient.
- [x] Exposer le nombre de ticks, les tailles Bid/Ask agrégées et les
  statistiques de spread dans les barres tick.
- [ ] Mettre en place un minimum de CI / validation locale sur les dépendances de dev.

## Backlog de moyen terme

- [ ] Tester la cohérence sur données réelles.
- [ ] Valider la qualité des signaux sur plusieurs actifs.
- [ ] Comparer les méthodes sur un même panel de données pour mesurer les écarts sémantiques.

## Backlog de long terme

- [ ] Visualisation / backtest / extension multi-actifs.
- [ ] Analyse des régimes sur séries longues et multipériodes.
- [ ] Évaluer les hypothèses de calibration et les limites méthodologiques pour les usages de recherche exploratoire.

## Décisions actives (source de vérité)

- ✅ Lookahead interdit par défaut (`allow_lookahead=False`).
- ✅ Seuils causals / quantiles avec décalage explicite quand requis.
- ✅ Sortie standardisée avec colonnes obligatoires en préfixe.
- ✅ Registry découplé dans `core/registry.py`.
- ✅ Profile YAML comme source de configuration par défaut.
- ✅ `params_hash` dans les sorties pour traçabilité et reproductibilité.
- ✅ Les méthodes ne doivent pas être comparées comme si elles partageaient le même sens sémantique de label.

## Archive / historique

Les éléments suivants ne doivent plus être considérés comme des
priorités actives :

- anciens plans de consensus multi-méthodes sur plusieurs labels
- notes de développement qui ne reflètent plus la direction actuelle
- fichiers parasites ou non utiles au projet fonctionnel

## Nettoyage effectué / en cours

- [x] Supprimer les fichiers de backup inutiles.
- [x] Nettoyer les imports et commentaires obsolètes dans les modules actifs.
- [x] Clarifier la TODO comme document de suivi actif plutôt que journal de développement désynchronisé.
