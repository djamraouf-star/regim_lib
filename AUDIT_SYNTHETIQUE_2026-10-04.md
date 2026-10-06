# Audit synthétique de regime_lib

**Date :** 4 octobre 2026. **Révision auditée :** [1f73f32](https://github.com/djamraouf-star/regim_lib/commit/1f73f32ba2b6888d634d25b7f0ab373d14c97c08).

**Suivi mis à jour :** 6 octobre 2026. Cette livraison complète la révision
[2293bda](https://github.com/djamraouf-star/regim_lib/commit/2293bda) avec les
corrections décrites ci-dessous, versionnées dans le même commit que ce suivi.
Les étapes 1 et 2 avaient été validées sur la révision
[67c47ad](https://github.com/djamraouf-star/regim_lib/commit/67c47ad).

## Suivi actuel — les 35 constats

**Ce tableau fait foi pour les statuts actuels.** Les sections datées du
5 octobre et les constats initiaux ci-dessous sont conservés comme historique.
Les huit étapes livrées du chantier `study` ne correspondent pas aux six étapes
de l'audit global : elles ont avancé les étapes globales 4 à 6, sans terminer
la causalité des détecteurs (étape 3).

**Bilan : 23 corrigés dans le périmètre indiqué, 1 partiel, 11 ouverts.**
Ce décompte n'est pas un pourcentage de travail : les constats se recoupent et
leurs efforts restants diffèrent. « Corrigé » ne vaut pas validation empirique.

**Validation locale :** 548 tests réussis avec exclusion de
`tests/test_screening_eurusd.py` (I24 toujours ouvert), dont 21 nouveaux tests
numériques, temporels et de traçabilité dans
[test_audit_completion.py](./tests/test_audit_completion.py). Un avertissement
joblib sur les cœurs physiques subsiste. La validation d'installation autonome
est détaillée avec I26. Aucun résultat empirique n'est déduit de ces tests.

**Changements observables :** VPIN utilise désormais les seuls buckets antérieurs
clos, avec un retard d'un bucket. Cette correction apparue pendant le chantier
a été conservée et vérifiée sur préfixes/prolongements/futurs perturbés.
Les nouveaux exports sont disponibles à clôture ; les anciens parquets VPIN sans
métadonnées restent exclus du mode causal. Le succès z sur dd signifie une
excursion moins négative. Les parquets portent disponibilité et sémantique.

### Critiques

| Réf. | Statut | Correction livrée / preuve et reste à faire |
|---|---|---|
| C1 | Corrigé | Moyenne glissante décalée d’un bucket : seules les valeurs des buckets antérieurs clos sont publiées. Huit préfixes, prolongements et futurs perturbés vérifiés, avec référence numérique dans [les tests](./tests/test_audit_completion.py). La fidélité du proxy et les buckets non fractionnés restent I12. |
| C2 | Corrigé | Identité, agrégation au timeframe des régimes et alignement strict : [tests du contrat](./tests/test_comparison_contract.py), [entrées study](./tests/test_study_input_contract.py). Horizons en barres uniquement ; durées explicites non implémentées. |
| C3 | Corrigé | Sélection actif/timeframe/configuration et rejet des doublons : [validation](./regime_lib/study/validation.py), [tests](./tests/test_study_input_contract.py). Comparaison simultanée de deux configurations d'une même méthode hors interface actuelle. |
| C4 | Corrigé | Types explicites, Spearman non applicable au nominal et rapport adapté : [tests des métriques](./tests/test_study_metric_contract.py), [régressions](./tests/test_study_audit_regressions.py). Ne clôture pas les cinq mesures communes. |
| C5 | Partiel | Bootstrap par blocs et correction BY globale optionnels : [inférence](./regime_lib/study/inference.py), [tests](./tests/test_study_inference.py). Diagnostic block_sensitivity ajouté, avec toutes les tailles pré-spécifiées et correction de leur famille. Écarts de moyenne seulement ; adéquation/calibration à vérifier sur les données retenues. Le protocole de confirmation est documenté mais la confirmation empirique nécessite une période réservée non consultée : C5 reste partiel. |

### Importants

| Réf. | Statut | Correction livrée / preuve et reste à faire |
|---|---|---|
| I1 | Corrigé | Métadonnées availability/available_at/revises_history dans les sorties ; filtre causal piloté par ces métadonnées et repli sur le registre pour les anciens parquets. Méthodes inconnues non déclarées exclues, publications futures et contradictions refusées. Modes des rapports distincts. Les variantes causales pivots/HMM restent I9/I10 ; C1 traité avec garde-fou pour les anciens parquets VPIN. |
| I2 | Corrigé | Ratio walk-forward appliqué, folds vides refusés et purge selon la fin des cibles : [tests](./tests/test_study_temporal_purge.py). Purge assurée par Study, pas par les fonctions de découpage seules. |
| I3 | Corrigé | Partialité propagée et trajectoires invalides exclues : [tests](./tests/test_comparison_contract.py). Lire ensemble coverage à l'étage courant et is_partial hérité de la source. |
| I4 | Corrigé | [session_grid](./regime_lib/core/temporal.py) génère des séances locales converties en UTC, y compris nuits et DST ; heures ambiguës/inexistantes refusées. Calendrier partagé avec resample et Study. Jours fériés explicites ; grilles D1/W1 toujours ancrées UTC, sans prétendre représenter une clôture journalière locale variable. |
| I5 | Corrigé | Validation OHLCV partagée, y compris appels directs : [validation](./regime_lib/utils/validation.py), [tests](./tests/test_comparison_contract.py). |
| I6 | Corrigé | Alias timestamp/datetime/date/time mutualisés : [chargeur](./regime_lib/core/data_loader.py), [tests](./tests/test_comparison_contract.py). |
| I7 | Corrigé | Les 13 détecteurs déclarent dimension, description et échelle nominale ; métadonnées conservées en sortie et dans les features. Labels natifs conservés ; aucune équivalence interméthodes ni direction de position implicite. Les scores confidence restent I8. |
| I8 | Ouvert | Scores confidence toujours hétérogènes. Déclarer leur type/sens ; aucune calibration commune hors échantillon livrée. |
| I9 | Ouvert | Pivots rétrospectifs conservés. Distinguer observation, confirmation et révision ; variante causale à événements et tests au futur à livrer. |
| I10 | Ouvert | HMM rétrospectif conservé. Apprentissage sur train, filtrage avant, caractérisation des états et diagnostics à livrer. |
| I11 | Ouvert | Machine à états de confirmation MSS/CHOCH ou renommage explicite de la variante simplifiée à réaliser. |
| I12 | Ouvert | Clarifier données/formules des proxys OFI/VPIN et renommer, ou implémenter les indicateurs correspondants ; distinct de la fuite C1. |
| I13 | Ouvert | Amorçage ADX à vérifier sur référence numérique indépendante ; corriger ou documenter la variante de Wilder. |
| I14 | Corrigé | NaN OHLC rejetés à l'entrée publique : [validation](./regime_lib/utils/validation.py), [tests](./tests/test_comparison_contract.py). Le cumul interne Kaufman ne reprend pas après un NaN injecté hors contrat public. |
| I15 | Ouvert | dd/ru toujours sur close dans [targets](./regime_lib/study/targets.py). Restent nommage excursion/drawdown, MFE/MAE high/low long/short et temps jusqu'aux extrêmes. |
| I16 | Corrigé | vol_fwd_1 refusé ; le succès z sur dd utilise maintenant cible > médiane, donc une excursion moins négative. Référence numérique et documentation corrigées. |
| I17 | Corrigé | Hit rate réservé aux features directionnelles déclarées et aux rendements : [tests](./tests/test_study_metric_contract.py). |
| I18 | Corrigé | Inconnus, partialité et chauffe déclarée exclus ; support commun et couverture exportés : [tests](./tests/test_comparison_contract.py). Tous les détecteurs ne déclarent pas séparément leur chauffe. |
| I19 | Corrigé | Tous les folds, tailles d'effet, effectifs, épisodes, stabilité et exports : [tests](./tests/test_study_all_folds.py). Stabilité descriptive, pas confirmation indépendante. |
| I20 | Ouvert | Baseline constante et métriques pénalisant les partitions dégénérées à ajouter à la comparaison PELT. |
| I21 | Corrigé | Index stricts, cas vides et segments invalides contrôlés : [tests](./tests/test_comparison_contract.py). Ne garantit pas la pertinence de la pureté (I20). |
| I22 | Corrigé | Manifeste partagé détection/étude : paramètres effectifs, provenance, version du protocole, versions des dépendances, empreintes du code/ressources, données préparées, sorties et artefacts. Support commun explicitement empreinté ; mutation après run refusée à l’export. Rejouer exige de conserver les données originales. |
| I23 | Corrigé | Résolveur et schémas uniques pour les 13 détecteurs : [tests](./tests/test_config_profiles.py). Refonte des sections donnees/context/segmentation hors périmètre. |
| I24 | Ouvert | [Test screening](./tests/test_screening_eurusd.py) dépend encore du module analyses absent. Restaurer/déplacer la fonctionnalité ou retirer ce test après décision de périmètre ; obtenir une collecte complète. |
| I25 | Corrigé | Invariance temporelle VPIN, contre-exemples de remplacement des pivots et normalisation globale HMM ; référence numérique et invariance au futur de Volume Profile ; référence de projection et tests calendrier/métriques/manifestes dans [les nouveaux tests](./tests/test_audit_completion.py). Ces contre-exemples exposent les défauts encore ouverts I9/I10 ; les futures mesures de l’étape 4 devront apporter leurs propres références. |
| I26 | Corrigé | SciPy et ruptures directs, ressources YAML/CSV déclarées, lanceur regime-report réparé. Wheel construite et installée avec ses dépendances dans un environnement isolé ; pip check, identité des sources installées, instanciation des 13 détecteurs, profils YAML, CSV, calcul PELT et aides des deux commandes vérifiés hors import du dépôt. |

### Mineurs

| Réf. | Statut | Correction livrée / preuve et reste à faire |
|---|---|---|
| M1 | Corrigé | Rapport multi-labels, transitions adjacentes et épisodes interrompus par inconnus/trous : [tests](./tests/test_study_descriptive.py). |
| M2 | Ouvert | Agréger progressivement ou documenter que le chargeur accumule les lots avant concaténation. |
| M3 | Corrigé | Exports dupliqués et auto-import retirés de methods ; masques VPIN/Volume Profile délégués au calendrier commun ; arguments high/low et variable de pivots inutilisés retirés du classificateur MSS/CHOCH. Ancien rapport dupliqué déjà supprimé. |
| M4 | Ouvert | Guides enrichis, mais versions/inventaire à centraliser. Les passages anciens de study sur la robustesse statistique ont été réconciliés. |

### Ordre de reprise et critères de clôture

Huit des neuf constats partiels ont reçu leur complément logiciel ; **C5 reste
partiel sur la confirmation empirique**. Le diagnostic et le protocole sont
livrés dans le [guide study](./regime_lib/doc/study.md), mais aucun code ne peut
certifier qu'une période n'a jamais été consultée. Il faut fixer les données,
les périodes et les critères avant l'analyse finale.

1. **I24 — collecte complète.** Résoudre le module screening absent, puis lancer
   la suite sans exclusion. I26 est traité séparément par la wheel autonome.
2. **C1 livré — surveiller la migration.** Régénérer les anciens parquets VPIN
   pour bénéficier du calcul causal ; la fidélité économique du proxy reste I12.
3. **I9/I10 — variantes causales pivots/HMM.** Les métadonnées I1 sont livrées ;
   les algorithmes disponibles à date et leurs tests restent à réaliser.
4. **Étape globale 4 — mesures communes.** Terminer I8/I15/I20 et les cinq
   mesures, avec des références numériques ; I7/I16 sont désormais traités.
5. **C5 — confirmation.** Réserver une période indépendante et figer les choix
   avant consultation. Les tests synthétiques ne constituent pas cette preuve.

I11–I13 restent un chantier distinct de fidélité des indicateurs. M2/M4 restent
ouverts. Les constats historiques ci-dessous ne sont pas les statuts actuels.

## Historique — chantier YAML du 5 octobre, désormais livré dans 2293bda

### Défaut architectural identifié

L'héritage des paramètres était implémenté deux fois : fusion profonde des
méthodes dans [loader.py](./regime_lib/config/loader.py) pour le CLI, puis
cascade dans [base.py](./regime_lib/core/base.py) pour les instances Python.
Dans le CLI, la lecture des défauts par la cascade était redondante pour les
clés déjà fusionnées. En Python direct, elle restait déterminante.

Cette architecture présentait cinq défauts liés à I23 :

1. Deux chemins de résolution pour une même règle métier, avec une priorité
   ambiguë entre surcharge générale et défaut par timeframe.
2. Chargement silencieux des défauts : une exception pouvait déclencher les
   constantes de secours au lieu de signaler un YAML invalide.
3. Cache global conservant les anciennes valeurs après modification du fichier.
4. Tables et constantes dupliquées entre les détecteurs et le YAML, susceptibles
   de diverger.
5. Paramètres de méthodes sans validation stricte : une faute comme `n_atrr`
   pouvait être ignorée pendant que `n_atr` était hérité des défauts.

### Décision et architecture livrée

Les **13 détecteurs** déclarent un `PARAM_SCHEMA` unique : clés autorisées,
types, bornes, possibilité de surcharge par timeframe et contraintes croisées.
Aucune liste `PARAM_KEYS` indépendante n'est entretenue. Les signatures Python
conservent leurs arguments nommés ; un test vérifie leur concordance avec les
schémas. Les valeurs par défaut appartiennent exclusivement au
[YAML de référence](./regime_lib/config/profiles/default.yaml).

| Composant | Responsabilité après refonte |
|---|---|
| [parameters.py](./regime_lib/config/parameters.py) | Schémas déclaratifs, contraintes, résolution commune, valeurs effectives et provenance. |
| [loader.py](./regime_lib/config/loader.py) et [schema.py](./regime_lib/config/schema.py) | Lecture, validation des profils et conservation séparée des défauts et surcharges de méthodes. |
| [base.py](./regime_lib/core/base.py) et [détecteurs](./regime_lib/methods/) | Consommation des paramètres résolus ; suppression de l'ancienne cascade, du cache et des constantes de secours. |
| [cli.py](./regime_lib/cli.py) | Construction par le même résolveur, inspection sans données et export des paramètres effectifs. |

Priorité, du moins prioritaire au plus prioritaire :

**Défaut général → défaut du timeframe → surcharge générale du profil →
surcharge du timeframe du profil → argument explicite Python/CLI.**

La fusion profonde ne concerne plus `methodes`. Les sections générales
`donnees`, `context` et `segmentation` conservent leur comportement historique.
Leur refonte complète ne fait pas partie de cette livraison.

Les clés inconnues, valeurs non finies, types invalides, timeframes inconnus,
contraintes incompatibles et clés YAML dupliquées sont rejetés. Les fusions
YAML `<<` sont également refusées. Le chargement vérifie toutes les méthodes
et tous les timeframes du profil, y compris ceux non sélectionnés pour un run.
Un fichier par défaut absent, corrompu ou incomplet n'active aucun secours codé.

Un profil chargé capture les valeurs à cet instant ; `for_method(...)` fournit
une copie isolée de ses couches. Recharger avec `load_profile(...)` prend en
compte les changements du fichier. Les instances déjà créées restent stables.
Une nouvelle instanciation directe sans profil lit les défauts actuels.

### Migration et changements observables

- Construire les détecteurs avec `profile=profil.for_method("adx")` plutôt
  qu'avec un sous-dictionnaire supposé fusionné. `profil.methodes` contient
  uniquement les surcharges ; il est vide pour le profil `default`.
- Lire les valeurs finales dans `detecteur.params` et leur provenance dans
  `detecteur.param_sources`. Un dictionnaire brut reste accepté comme surcharge.
- Les demi-fenêtres de chauffe sont calculées à partir de la fenêtre effective
  lorsque le profil contient `null` pour `min_periods` ou `min_periods_c`.
  Une chauffe numérique reste fixe et doit respecter les bornes de sa fenêtre.
- Un argument Python `None` ou un JSON `null` dans `--method-params` signifie
  « argument absent ». Réactiver une chauffe automatique se fait dans le profil.
- La surcharge générale Shannon EURUSD `fenetre: 100` s'applique désormais à
  tous les timeframes. Pour conserver M1/M5 à 200, les préciser dans le profil.
- Les hashes portent maintenant sur les paramètres effectifs, timeframe inclus ;
  les anciens hashes ne sont pas comparables aux nouveaux. La provenance ne
  change pas le hash d'une méthode à valeurs effectives identiques.
- Un chemin utilisateur nommé `default.yaml` est effectivement chargé. HMM et
  `n_bins` Shannon suivent désormais le résolveur commun.

Inspection sans chargement de données :

```bash
regime-lib --show-config --profile eurusd --timeframe M1,H1 --methods adx,shannon
```

Le JSON expose `params`, `sources` et `params_hash` par méthode/timeframe.
Ces éléments sont également exportés sous `effective_config` dans le fichier
`run_meta.json` de chaque run. Voir le [guide de migration](./regime_lib/doc/config.md)
et le [guide d'ajout d'une méthode](./regime_lib/doc/adding_method.md).

### Validation exécutée et limites

```bash
.venv/bin/python -m pytest --ignore=tests/test_screening_eurusd.py -o addopts='' -q
.venv/bin/python -m pytest tests/test_config_profiles.py tests/test_config_cli.py -o addopts='' -q
```

**Résultats : 429 tests réussis sur la suite avec exclusion ; 48 tests ciblés
réussis** dans [test_config_profiles.py](./tests/test_config_profiles.py) et
[test_config_cli.py](./tests/test_config_cli.py). Ces 48 cas remplacent les deux
anciens tests de profils, soit 46 cas supplémentaires. Ils couvrent les
priorités, la validation, le rechargement, l'équivalence CLI/Python des 13
détecteurs et la cohérence entre paramètres exportés et hashes du parquet.
Les tests ciblés ont été relancés après la simplification finale du YAML.
Les liens locaux des documents modifiés et `git diff --check` ont été vérifiés.

L'avertissement joblib sur les cœurs physiques reste présent. Le
[test de screening](./tests/test_screening_eurusd.py) reste exclu pour son
import absent : **I24 reste ouvert**, la suite complète n'est pas validée.
Les ressources YAML/CSV sont déclarées dans [pyproject.toml](./pyproject.toml),
mais aucune construction/installation autonome de wheel n'a été validée ; les
outils de construction manquent dans l'environnement virtuel utilisé. Les
dépendances directes SciPy/ruptures restent à traiter : **I26 reste ouvert**.

Cette livraison ne démontre ni causalité ni validité statistique des méthodes.
Les étapes 3 à 5 restent à réaliser. Pour I22, les versions du code et des
dépendances ainsi que les empreintes des données et du support restent ouvertes.

## Historique — portée et conclusion au 5 octobre

L'audit initial était statique et concernait la révision 1f73f32 ; sa suite de
tests n'avait pas été exécutée. Les tableaux de constats ci-dessous conservent
cet état initial pour la traçabilité. **Le tableau du 6 octobre en tête du document fait foi
pour les statuts actuels.** Cette mise à jour ne constitue pas un nouvel audit
exhaustif des méthodes.

L'objectif est de comparer les familles d'indicateurs en projetant leurs
représentations dans l'espace du prix, selon cinq dimensions : **amplitude,
régularité du trajet, durée, fréquence d'apparition et excursion opposée**.
L'API n'est ni une stratégie de trading ni un moteur de backtest. Les labels
natifs ne constituent pas une échelle commune de comparaison ; PELT reste une
référence descriptive optionnelle.

Les étapes 1 et 2 ont établi le contrat commun puis renforcé les contrôles de
série, l'alignement temporel et la validité du support de comparaison. Les
classements ne sont pas encore fiables : la fuite future VPIN (C1), les métriques
ordinales sur catégories (C4) et l'inférence statistique (C5) restent à corriger.
Les données de marché ne sont pas incluses dans le dépôt et aucune performance
empirique ni supériorité d'une méthode n'est établie.

## Historique — suivi au 5 octobre, étapes 1 et 2 et chantier YAML

**Étape 1 livrée :** [contrat v1.0.0](./regime_lib/doc/protocole_projection_mesure.md),
avec unités de projection, cinq mesures, conventions temporelles et exemples
numériques. Une convention documentée n'est pas considérée comme implémentée.

**Étape 2 livrée :** [guide d'alignement et de validité](./regime_lib/doc/alignement_validite.md),
contrôles et propagation du support jusqu'aux projections et études.

| Réf. | Statut au 5 octobre 2026 | Correction réalisée / limite restante |
|---|---|---|
| C2 | Corrigé pour les horizons en barres | L'étude vérifie l'actif et le timeframe, agrège les prix source plus fins avant de calculer les cibles et refuse les timestamps de régimes absents des prix. Les horizons en durée explicite restent à implémenter. |
| C3 | Corrigé pour la sélection d'une série/configuration | Sélection actif/timeframe et `params_hash` par méthode ; refus des doublons et mélanges. Comparer simultanément deux configurations d'une même méthode dans un pivot reste hors de l'interface actuelle. |
| I3 | Corrigé pour les marqueurs disponibles | Partialité conservée au resampling, dans les appels directs aux détecteurs et à l'export ; fenêtres invalides exclues des cibles. `coverage` mesure la présence des barres à l'étage courant et se lit avec `is_partial`, qui conserve l'invalidité source. |
| I4 | Partiellement corrigé | Conventions source déclarables, refus des conversions impossibles, grille UTC centralisée et W1 ancré au lundi pour agrégation et masques. Les séances peuvent être fournies par `expected_index` ; pas de génération automatique de bornes locales/DST. Les anciennes entrées sans métadonnées gardent les défauts documentés close/UTC. |
| I5 | Corrigé pour les entrées des détecteurs et chargeurs | Contrôles OHLC numériques, finis, positifs et cohérents ; index UTC unique ; volumes disponibles finis et non négatifs. Validation partagée par tous les détecteurs, y compris les appels directs. |
| I6 | Corrigé | Les quatre alias timestamp/datetime/date/time utilisent la même préparation tick en lecture complète et par lots. |
| I14 | Corrigé à l'entrée publique | Les NaN OHLC sont rejetés avant le calcul de Kaufman. Le cumul interne n'a pas été transformé en calcul capable de reprendre après un NaN. |
| I18 | Corrigé pour le support valide | INCONNU, absences, chauffe déclarée et partialité sont exclus ; couverture propre et masque commun par cible exportés. Les compteurs INCONNU et chauffe restent distincts, tous les détecteurs ne déclarant pas une chauffe séparée. |
| I21 | Corrigé pour l'alignement et les cas invalides | Index stricts, segments manquants/dupliqués ou disjoints refusés, sorties vides prises en charge ; segments traversant des trous attendus ou des barres invalides signalés. |
| I25 | Partiellement corrigé | 37 tests ciblés ajoutés sur identité, calendrier, projection, validité et support commun ; 48 cas couvrent désormais la configuration et son intégration CLI. Les tests spécifiques de causalité VPIN/pivots/HMM et les références numériques des nouvelles mesures restent à ajouter. |
| C1, I1, I9, I10 | À réaliser — étape 3 | Le contrat décrit les modes rétrospectif et disponible à date. Clôture des buckets VPIN, métadonnées de disponibilité, événements de pivots et filtrage avant HMM ne sont pas encore implémentés. |
| C4, I7, I8, I15–I17, I20 | À réaliser — étape 4 | Les mesures communes, la sémantique des scores et la suppression des classements arbitraires restent à implémenter. L'accord utilise désormais un support commun, mais une pureté parfaite d'un détecteur constant reste possible. |
| C5, I2, I19 | À réaliser — étape 5 | Dépendance temporelle, comparaisons multiples, purge et exports statistiques complets restent ouverts. |
| I22 | Partiellement corrigé — étape 6 | Paramètres effectifs, provenance et hashes livrés. Versions du code/dépendances et empreintes des données/support restent ouvertes. |
| I23 | Corrigé pour les paramètres de méthodes | Schémas déclaratifs et résolveur CLI/Python uniques ; priorités, HMM, Shannon et chemin utilisateur `default.yaml` corrigés. Migration documentée. |
| I24 | À réaliser — étape 6 | La collecte complète reste bloquée par le module de screening absent. |
| I26 | Partiellement traité, reste ouvert — étape 6 | Ressources YAML/CSV déclarées ; dépendances directes et installation autonome restent à vérifier/corriger. |
| I11–I13, M1–M4 | Non traités dans ces livraisons | Fidélité des indicateurs et maintenance à traiter séparément selon le périmètre retenu. |

Les nouveaux exports transportent les métadonnées de source disponibles. Un
ancien parquet qui ne déclare pas sa source, son côté de prix, ses ajustements
ou son calendrier ne permet pas de certifier ces dimensions par rétro-inférence.
Le hash utilisé pour sélectionner une configuration reste soumis à I22.

### Validation historique des étapes 1 et 2 — 4 octobre 2026

Sur la révision 67c47ad, avec Python 3.14 et un environnement local contenant
les dépendances du projet ainsi que SciPy et ruptures installés explicitement :

```bash
.venv/bin/python -m pytest --ignore=tests/test_screening_eurusd.py -o addopts='' -q
```

**Résultat : 383 tests réussis, dont 37 nouveaux tests ciblés ; un avertissement
joblib sur la détection des cœurs physiques.** Compilation Python, liens des
nouveaux documents et `git diff --check` vérifiés.

La collecte du [test de screening](./tests/test_screening_eurusd.py) a également
été tentée : elle échoue sur le module absent
`analyses.EURUSD.scripts.screening_eurusd`. **I24 reste ouvert ; il ne s'agit donc
pas d'une validation de la suite entière.** L'installation explicite de SciPy
et ruptures pour les tests ne résout pas I26 et ne valide pas une wheel autonome.

Les [tests du contrat](./tests/test_comparison_contract.py) vérifient notamment
le passage de prix M5 à des cibles H1, le rejet d'index décalés de même longueur,
la propagation des barres partielles et les supports communs vides. Ils ne
prouvent ni la causalité des méthodes encore concernées par l'étape 3, ni la
validité des classements statistiques encore concernés par les étapes 4 et 5.

## Constats initiaux — révision 1f73f32

Les problèmes et recommandations ci-dessous sont conservés à titre historique.
Pour le périmètre et la terminologie de développement actuels, utiliser le
contrat et le plan de livraison en fin de document.

## Architecture et garanties existantes

Le flux principal suit [data_loader.py](./regime_lib/core/data_loader.py) → [cli.py](./regime_lib/cli.py) → [détecteurs](./regime_lib/methods/) → [output.py](./regime_lib/core/output.py). Le module [study](./regime_lib/study/) recharge les régimes, calcule indépendamment les cibles futures, puis applique découpages et statistiques. La branche [projection](./regime_lib/projection/) fournit une référence descriptive offline par PELT.

Points solides : registre extensible, seuils historiques décalés pour ATR/Kaufman, autorisation explicite des trois détecteurs offline connus et séparation entre détection et calcul des cibles.

## CRITIQUE

| Réf. | Fichier / fonction | Problème et conséquence | Correction recommandée |
|---|---|---|---|
| C1 | [vpin.py](./regime_lib/methods/volume/vpin.py), `VpinDetector.fit_predict` | La valeur finale d'un bucket est attribuée à ses premières observations : fuite future malgré `requires_lookahead=False`. Masquer le dernier bucket ne corrige pas les buckets historiques. | Publier les valeurs seulement à la clôture des buckets ; tester l'invariance des sorties passées par prolongement et perturbation du futur. |
| C2 | [study.py](./regime_lib/study/study.py), `_charger_cibles`, `_evaluer` | Alignement par timestamp sans vérifier le timeframe : des régimes H1 peuvent être évalués sur cinq minutes au lieu de cinq heures. | Imposer actif, timeframe et unité d'horizon ; calculer sur les mêmes barres ou une durée explicitement définie. |
| C3 | [features.py](./regime_lib/study/features.py), `extraire_features` | Le pivot ignore actif, timeframe et configuration. Le format multi-timeframes du CLI peut produire doublons, ambiguïtés ou erreurs. | Sélectionner une série unique ou conserver une clé composée ; vérifier l'unicité. |
| C4 | [evaluation.py](./regime_lib/study/evaluation.py), `ic_spearman` ; [report.py](./regime_lib/study/report.py), `generer_rapport` | Codes ordinaux arbitraires appliqués aux labels, puis sélection du meilleur IC. Renommer les catégories peut changer le classement. | Réserver Spearman aux variables ordonnées ; comparer distributions conditionnelles et tailles d'effet. |
| C5 | [evaluation.py](./regime_lib/study/evaluation.py), `batterie_tests`, `par_modalite_tests` | Cibles chevauchantes, régimes persistants et comparaisons multiples ne sont pas traités par l'inférence de l'étude. Les seuils de significativité affichés peuvent être trompeurs. | Inférence temporelle adaptée, correction multiple et période finale de confirmation indépendante. |

## IMPORTANT — données et causalité

| Réf. | Fichier / fonction | Problème et correction recommandée |
|---|---|---|
| I1 | [study.py](./regime_lib/study/study.py), `_evaluer` ; [features.py](./regime_lib/study/features.py), `extraire_features` | Découper des labels offline déjà calculés ne les rend pas hors échantillon. La liste d'exclusion est codée en dur. Séparer parcours descriptif/prédictif et porter causalité/disponibilité dans les métadonnées. |
| I2 | [split.py](./regime_lib/study/split.py), `split_holdout`, `split_walk_forward` | Embargo nul par défaut ; `train_ratio` validé mais ignoré dans le walk-forward. Purger selon la fin des cibles, appliquer le ratio et contrôler les folds vides. |
| I3 | [data_loader.py](./regime_lib/core/data_loader.py), `resample` ; [cli.py](./regime_lib/cli.py), `_run_one_timeframe` | Les marqueurs sources `is_partial` sont perdus puis recréés incomplètement ; ils ne sont pas réintroduits dans les sorties des détecteurs. Propager partialité/couverture jusqu'aux cibles et à l'évaluation. |
| I4 | [data_loader.py](./regime_lib/core/data_loader.py), `load_parquet`, `resample` ; [utils.py](./regime_lib/core/utils.py), `unaligned_mask` | Convention ouverture/clôture implicite, fréquence source non contrôlée et ancrage W1 incohérent entre masque et resampling. Déclarer les conventions, refuser les conversions impossibles et centraliser les bornes calendaires. |
| I5 | [validation.py](./regime_lib/utils/validation.py), `validate_ohlcv` ; [data_loader.py](./regime_lib/core/data_loader.py), `_preparer_tick` | Contrôles insuffisants des valeurs finies, prix positifs, volumes et relations OHLC. Instaurer un contrat d'entrée commun, y compris pour les appels directs aux détecteurs. |
| I6 | [data_loader.py](./regime_lib/core/data_loader.py), `_preparer_tick`, `_charger_tick_par_lots` | Les alias temporels tick sont traités différemment selon le mode de chargement. Mutualiser leur reconnaissance et conversion. |

## IMPORTANT — détecteurs et labels

| Réf. | Fichier / fonction | Problème et correction recommandée |
|---|---|---|
| I7 | [ATR](./regime_lib/methods/volatility/atr_volatility.py), [Kaufman](./regime_lib/methods/trend/er_kaufman.py), [ADX](./regime_lib/methods/trend/adx.py), `fit_predict` | Les mêmes labels décrivent volatilité, efficience ou force directionnelle différentes. Conserver les labels natifs et distinguer direction, volatilité, efficience, structure et événement. |
| I8 | [HMM](./regime_lib/methods/statistical/hmm_gaussian.py), `_confidence_from_posteriors` ; [MSS](./regime_lib/methods/price/mss_choch.py) et [Divergence PV](./regime_lib/methods/volume/divergence_pv.py), `fit_predict` | `confidence` mélange probabilités transformées, distances et constantes. Exposer le type de score ; ne pas employer de seuil universel sans calibration hors échantillon. |
| I9 | [utils.py](./regime_lib/core/utils.py), `build_alternating_pivots` ; [price_action.py](./regime_lib/methods/price/price_action.py), `_classify_from_pivots` | Un pivot ultérieur plus extrême peut remplacer un pivot confirmé : les révisions dépassent une simple latence fractale. Distinguer pivot, confirmation et révision ; tester les remplacements et créer une variante causale à événements si nécessaire. |
| I10 | [hmm_gaussian.py](./regime_lib/methods/statistical/hmm_gaussian.py), `_entrainer_hmm` | Normalisation, apprentissage, décodage et postérieurs globaux utilisent le futur ; numéros d'états non stables entre fits. Pour une variante prédictive : apprentissage train uniquement, filtrage avant, caractérisation des états sur train et diagnostics de convergence. |
| I11 | [mss_choch.py](./regime_lib/methods/price/mss_choch.py), `_classify_mss_choch` | Aucune mémoire du CHOCH préalable ni séquence explicite de confirmation. Implémenter une machine à états ou renommer en classificateur simplifié de cassures. |
| I12 | [ofi.py](./regime_lib/methods/volume/ofi.py), [vpin.py](./regime_lib/methods/volume/vpin.py), `fit_predict` | Les tailles de cotations ne sont pas des volumes exécutés ; signe OFI économiquement ambigu ; buckets VPIN non fractionnés et somme des déséquilibres absolus. Renommer les proxys ou employer les données/formules correspondant aux indicateurs annoncés. |
| I13 | [adx.py](./regime_lib/methods/trend/adx.py), `_wilder_smooth`, `_compute_adx` | Amorçage EWM différent du Wilder de référence. Corriger ou documenter la variante ; vérifier sur des valeurs de référence indépendantes. |
| I14 | [er_kaufman.py](./regime_lib/methods/trend/er_kaufman.py), `_efficiency_ratio` | Un NaN contamine toute la suite du cumul. Rejeter les entrées invalides ou employer un calcul glissant permettant la reprise. |

## IMPORTANT — potentiel futur et comparaison

| Réf. | Fichier / fonction | Problème et correction recommandée |
|---|---|---|
| I15 | [targets.py](./regime_lib/study/targets.py), `calculer_cible` | Run-up/drawdown sur clôtures seulement ; `dd` est une excursion depuis l'entrée, pas un drawdown sommet-creux. Préciser les noms et ajouter MFE/MAE long/short sur high/low, avec prix d'entrée et temps jusqu'aux extrêmes. |
| I16 | [targets.py](./regime_lib/study/targets.py), `calculer_cible` ; [evaluation.py](./regime_lib/study/evaluation.py), `_convention_succes` | `vol_fwd_1` entièrement NaN avec l'écart-type échantillonnal ; pertes plus profondes comptées comme succès de drawdown malgré le commentaire inverse. Définir K=1 et le sens économique du succès. |
| I17 | [evaluation.py](./regime_lib/study/evaluation.py), `hit_rate_directionnel` | Confiance, heures et booléens sont convertis implicitement en directions de position. Restreindre la métrique aux signaux directionnels déclarés. |
| I18 | [features.py](./regime_lib/study/features.py), `extraire_features` ; [study.py](./regime_lib/study/study.py), `_evaluer` | `INCONNU` participe aux statistiques ; les chauffes différentes rendent les échantillons difficilement comparables. Exclure les observations invalides et publier couverture propre et période commune valide. |
| I19 | [study.py](./regime_lib/study/study.py), `_evaluer`, `save` ; [report.py](./regime_lib/study/report.py), `generer_rapport` | Tests par modalité limités au fold 0 et détails conditionnels incomplets à l'export. Exporter tous les folds/régimes/horizons et leur stabilité. |
| I20 | [agreement.py](./regime_lib/projection/agreement.py), `purete_par_segment`, `comparer_methodes` | Un détecteur constant obtient une pureté parfaite. Ajouter cette baseline et des métriques pénalisant les partitions dégénérées ; garder PELT comme référence descriptive. |
| I21 | [agreement.py](./regime_lib/projection/agreement.py), `purete_par_segment` ; [features.py](./regime_lib/projection/features.py), `caracteriser_segments` ; [projection.py](./regime_lib/projection/projection.py), `projeter_barres` | Association par position sans vérifier les index : des séries décalées de même taille sont acceptées. Imposer un alignement strict et traiter les cas vides/manquants. |

## IMPORTANT — reproductibilité et tests

| Réf. | Fichier / fonction | Problème et correction recommandée |
|---|---|---|
| I22 | [base.py](./regime_lib/core/base.py), `__init__`, `_resolve` ; [cli.py](./regime_lib/cli.py), `_run_one_timeframe` | Hash des arguments initiaux, souvent `None`, plutôt que des paramètres effectifs. Sérialiser les paramètres résolus, version du code, dépendances et empreinte des données. |
| I23 | [loader.py](./regime_lib/config/loader.py), `load_profile`, `_deep_merge` ; constructeurs [HMM](./regime_lib/methods/statistical/hmm_gaussian.py) et [Shannon](./regime_lib/methods/entropy/shannon.py) | Chemin utilisateur nommé `default.yaml` ignoré ; cascade HMM distincte ; `n_bins` Shannon par timeframe non résolu comme tel ; surcharges globales dominées par valeurs par timeframe héritées. Unifier la cascade et tester ses priorités. |
| I24 | [test_screening_eurusd.py](./tests/test_screening_eurusd.py), import de module | Module `analyses.EURUSD.scripts.screening_eurusd` absent du dépôt audité : collecte autonome bloquée. Versionner le module ou déplacer ses fonctions dans la bibliothèque. |
| I25 | [tests](./tests/), couverture | Aucun test dédié VPIN, Volume Profile ou projection ; certaines assertions se limitent aux formes et bornes. Ajouter contre-exemples temporels et résultats numériques attendus indépendants. |
| I26 | [pyproject.toml](./pyproject.toml) ; imports [segmentation.py](./regime_lib/projection/segmentation.py) et [evaluation.py](./regime_lib/study/evaluation.py) | `ruptures` absent des dépendances et SciPy seulement transitif ; ressources YAML/CSV non explicitement déclarées. Déclarer dépendances/extras et ressources. Le risque d'omission des ressources reste à confirmer par construction et installation d'une wheel propre. |

## MINEUR

- **M1 — Rapport spécialisé présenté comme générique.** [study/report.py](./regime_lib/study/report.py), `extraire_etats_int`, `rapport_complet`, `matrice_transition` : seuls les labels HMM sont reconnus ; supprimer les inconnus crée aussi des transitions artificielles. Généraliser ou restreindre l'API et compter uniquement les transitions adjacentes.
- **M2 — Mémoire du chargement par lots.** [data_loader.py](./regime_lib/core/data_loader.py), `_charger_tick_par_lots` : tous les lots restent en mémoire avant concaténation. Agréger progressivement ou documenter cette limite.
- **M3 — Duplication et éléments inutilisés.** [methods/__init__.py](./regime_lib/methods/__init__.py), masques de [VPIN](./regime_lib/methods/volume/vpin.py)/[Volume Profile](./regime_lib/methods/volume/volume_profile.py), `_classify_mss_choch` dans [mss_choch.py](./regime_lib/methods/price/mss_choch.py) : exports/masques dupliqués et variables/arguments inutilisés. Centraliser et simplifier. Aucune dépendance déclarée entièrement sans usage n'a été démontrée.
- **M4 — Versions et documentation.** [pyproject.toml](./pyproject.toml), [__init__.py](./regime_lib/__init__.py), [methods/__init__.py](./regime_lib/methods/__init__.py), [volume.md](./regime_lib/doc/volume.md) : versions divergentes et méthodes actives décrites comme futures. Centraliser version et inventaire.

## Interprétation des autres détecteurs

À la lecture du code, ATR et Kaufman utilisent des seuils historiques décalés. ADX, Shannon, PCA, Minkowski et Divergence PV ne présentent pas la fuite future identifiée dans VPIN, sous réserve d'entrées correctement horodatées.

Shannon décrit un histogramme de rendements, pas leur ordre temporel. PCA utilise une fenêtre strictement passée, contrairement aux détecteurs intégrant la clôture courante. Minkowski classe la taille relative du rendement ; son vocabulaire physique ne démontre aucune causalité économique. Volume Profile reste une approximation OHLC du volume par prix.

Les excursions futures maximales décrivent une opportunité rétrospective ; elles ne constituent pas à elles seules un gain réalisable par une règle d'entrée/sortie.

## Historique — plan de livraison au 5 octobre

1. **Étape 1 — Livrée : contrat commun.** Identité, unités d'observation,
   conventions, cinq mesures et exemples de référence documentés.
2. **Étape 2 — Livrée dans les limites ci-dessus : données et support.**
   Alignement, contrôles OHLCV, partialité, couverture propre et support commun.
   Les calendriers de séances doivent être fournis explicitement ; les grilles
   prises en charge sont UTC.
3. **Étape 3 — Prochaine : disponibilité temporelle (C1, I1, I9, I10).**
   Publier VPIN à la clôture des buckets ; porter disponibilité et révisions dans
   les métadonnées ; distinguer pivots observés/confirmés/révisés ; conserver le
   HMM rétrospectif et ajouter un mode appris sur une période de référence puis
   filtré en avant. Critère : le futur ne modifie pas les sorties déjà publiées
   en mode disponible à date, à apprentissage historique inchangé.
4. **Étape 4 — À réaliser : projection et cinq mesures communes.** Comparer les
   distributions dans l'espace du prix, expliciter orientation, unités et horizons,
   supprimer l'effet du codage arbitraire des labels et les conversions implicites
   en directions. Critère : une projection identique reçoit les mêmes mesures,
   quel que soit le nom des catégories ou de la méthode.
5. **Étape 5 — À réaliser : comparaison statistique.** Traiter chevauchements,
   dépendance temporelle, comparaisons multiples et frontières des périodes ;
   exporter toutes les périodes/modalités/horizons et leur stabilité. Confirmation
   indépendante si une sélection est présentée comme stable.
6. **Étape 6 — Partiellement livrée par anticipation : reproductibilité.**
   Configuration unifiée, paramètres effectifs, provenance et hashes livrés dans
   le chantier YAML. Restent : versions du protocole/code/dépendances, empreintes
   des données et du support, réparation de la collecte complète et vérification
   des dépendances/ressources dans une installation autonome.

**Ordre de reprise recommandé :** terminer la collecte et l'installation
(I24/I26) pour disposer d'une base de validation complète, puis reprendre les
étapes 3, 4 et 5. La numérotation historique est conservée ; le chantier YAML
constitue une livraison anticipée d'une partie de l'étape 6.

Les corrections de fidélité des indicateurs I11–I13 restent distinctes du contrat
commun. Les sujets de maintenance M1–M4 restent secondaires sauf s'ils bloquent
une livraison. Aucun résultat de ces étapes n'introduit une règle de trading,
un backtest ou un score global imposé.
