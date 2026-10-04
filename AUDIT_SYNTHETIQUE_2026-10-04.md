# Audit synthétique de regime_lib

**Date :** 4 octobre 2026. **Révision auditée :** [1f73f32](https://github.com/djamraouf-star/regim_lib/commit/1f73f32ba2b6888d634d25b7f0ab373d14c97c08).

## Portée et conclusion

Audit statique du code, de l'architecture et des tests versionnés. La suite de tests n'a pas été exécutée et les données de marché ne sont pas incluses dans le dépôt. Aucune performance empirique n'est établie. Les constats concernent la révision ci-dessus et doivent être revérifiés après correction.

L'objectif est de comparer les détecteurs et d'identifier les mouvements et horizons offrant le meilleur potentiel brut de gain. L'architecture constitue une base exploitable, mais les classements ne sont pas encore fiables : fuite future dans VPIN, identité temporelle insuffisamment contrôlée et métriques inadaptées aux catégories.

Ce document synthétise l'audit ; il ne constitue pas une correction du code.

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

- **M1 — Rapport spécialisé présenté comme générique.** [utils/report.py](./regime_lib/utils/report.py), `extraire_etats_int`, `rapport_complet`, `matrice_transition` : seuls les labels HMM sont reconnus ; supprimer les inconnus crée aussi des transitions artificielles. Généraliser ou restreindre l'API et compter uniquement les transitions adjacentes.
- **M2 — Mémoire du chargement par lots.** [data_loader.py](./regime_lib/core/data_loader.py), `_charger_tick_par_lots` : tous les lots restent en mémoire avant concaténation. Agréger progressivement ou documenter cette limite.
- **M3 — Duplication et éléments inutilisés.** [methods/__init__.py](./regime_lib/methods/__init__.py), masques de [VPIN](./regime_lib/methods/volume/vpin.py)/[Volume Profile](./regime_lib/methods/volume/volume_profile.py), `_classify_mss_choch` dans [mss_choch.py](./regime_lib/methods/price/mss_choch.py) : exports/masques dupliqués et variables/arguments inutilisés. Centraliser et simplifier. Aucune dépendance déclarée entièrement sans usage n'a été démontrée.
- **M4 — Versions et documentation.** [pyproject.toml](./pyproject.toml), [__init__.py](./regime_lib/__init__.py), [methods/__init__.py](./regime_lib/methods/__init__.py), [volume.md](./regime_lib/doc/volume.md) : versions divergentes et méthodes actives décrites comme futures. Centraliser version et inventaire.

## Interprétation des autres détecteurs

À la lecture du code, ATR et Kaufman utilisent des seuils historiques décalés. ADX, Shannon, PCA, Minkowski et Divergence PV ne présentent pas la fuite future identifiée dans VPIN, sous réserve d'entrées correctement horodatées.

Shannon décrit un histogramme de rendements, pas leur ordre temporel. PCA utilise une fenêtre strictement passée, contrairement aux détecteurs intégrant la clôture courante. Minkowski classe la taille relative du rendement ; son vocabulaire physique ne démontre aucune causalité économique. Volume Profile reste une approximation OHLC du volume par prix.

Les excursions futures maximales décrivent une opportunité rétrospective ; elles ne constituent pas à elles seules un gain réalisable par une règle d'entrée/sortie.

## Plan de correction priorisé

1. **P0 — Données et causalité :** corriger VPIN, imposer les clés de série et les conventions temporelles, propager validité/partialité. Critère : aucune sortie causale passée ne change lorsqu'on ajoute du futur.
2. **P0 — Évaluation :** supprimer les métriques ordinales arbitraires, restreindre le hit rate, purger selon les cibles et exclure les observations invalides. Critère : renommer les catégories ne change pas les conclusions.
3. **P1 — Potentiel brut :** définir entrée, unités et horizons ; ajouter rendement terminal, MFE/MAE long/short, quantiles et temps jusqu'aux extrêmes. Comparer des horizons en barres et des durées communes.
4. **P1 — Validation :** séparer exploration, sélection et confirmation finale ; traiter dépendance temporelle/comparaisons multiples ; exporter tous les folds.
5. **P1 — Reproductibilité :** réparer l'import absent, hacher paramètres effectifs et données, vérifier installation et ressources embarquées.
6. **P2 — Méthodes et maintenance :** préciser labels/scores, corriger Wilder/MSS, envisager des variantes causales distinctes pour Price Action/HMM, compléter les tests et nettoyer les duplications.
