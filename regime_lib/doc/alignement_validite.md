# Alignement et validité — étape 2

Cette livraison applique la partie données/support du
[contrat de projection et de mesure](./protocole_projection_mesure.md).
Les formules des cinq mesures et les corrections de disponibilité temporelle
C1/I1/I9/I10 appartiennent aux étapes suivantes. Les statistiques historiques
de l'étude ne sont pas encore certifiées par le contrat complet.

## Chargement et grilles

Les appels directs aux détecteurs exigent des OHLCV numériques cohérents,
des prix finis et positifs, un index UTC ordonné et unique. La colonne `volume`
peut contenir NaN pour une méthode qui ne l'utilise pas ; les méthodes dépendant
du volume exigent leurs colonnes de volume disponibles. Les barres partielles
sont conservées et signalées jusqu'à l'export.

Le chargeur normalise les noms, trie les observations et convertit les timestamps
en UTC. Pour les barres sans métadonnées, la convention par défaut déclarée par
l'API reste la clôture et le fuseau des timestamps naïfs reste UTC. Pour une autre
convention, la fournir explicitement :

```python
from regime_lib.core.data_loader import load_parquet, resample

prix, actif = load_parquet(
    "INDEX_H1.parquet", asset="INDEX", timeframe="H1",
    timestamp_convention="open", source_timezone="Europe/Paris",
)
prix_h4 = resample(prix, "H4")
```

La conversion ouverture/clôture ci-dessus utilise une durée fixe H1. Les grilles
prises en charge dans cette livraison sont UTC : D1 à minuit, W1 le lundi à minuit.
Le resampling et les masques des détecteurs emploient la même origine W1.
Les bornes de séances locales ou D1/W1 variables lors d'un changement d'heure
ne sont pas générées automatiquement. Les données hors grille sont refusées.

Une série régulière peut fournir sa fréquence par inférence. Pour une série
trouée ou trop courte, déclarer `df.attrs["timeframe"]` ou `source_timeframe`
dans `resample`. L'inférence décrit la cadence observée, elle ne peut pas
retrouver une fréquence nominale si toutes les barres intermédiaires manquent.
Les agrégations vers une fréquence plus fine ou non divisible sont refusées.

Le CLI expose `--source-timeframe`, `--timestamp-convention` et
`--source-timezone` pour les barres. Les ticks utilisent leur convention propre
et sont chargés en M1. Les quatre alias timestamp/datetime/date/time suivent le
même chemin de validation en lecture complète et par lots.

## Calendrier et données absentes

Par défaut, la grille attendue est continue UTC : aucun week-end ou arrêt de
séance n'est supposé. Pour distinguer fermeture prévue et donnée absente,
fournir un `expected_index` UTC de clôtures attendues, issu d'un calendrier de
séances, à `resample`, `calculer_cible` ou `caracteriser_segments`.
Cette grille doit couvrir les bornes complètes des agrégats étudiés, pas être
construite en supprimant les observations manquantes du prix.

Les clôtures attendues doivent appartenir à la grille UTC du timeframe.
Les fermetures exclues du calendrier ne constituent pas des trous ; toute
clôture attendue manquante invalide les trajets qui la traversent. Aucun prix
n'est interpolé. Sans calendrier fourni, le traitement reste conservateur.

`coverage` décrit la proportion de barres présentes à l'étage d'agrégation
courant. `is_partial` conserve en plus toute invalidité des barres sources :
les deux informations doivent être lues ensemble. L'agrégation ne transforme
pas une barre source partielle en donnée valide.

## Une série et une configuration par méthode dans une étude

```python
from regime_lib.study import Study

prix.attrs.update(asset="INDEX", timeframe="H1")
etude = Study(
    regimes_path="regimes.parquet",
    ohlcv_path="INDEX_H1.parquet",
    ohlcv_data=prix,
    asset="INDEX", timeframe="H1",
    methodes=["adx", "er_kaufman"],
    targets=["ret_fwd_5"],
)
etude.run()
etude.save("resultats")
```

`asset` et `timeframe` sélectionnent la série dans le parquet long. L'actif des
OHLCV est vérifié via leurs métadonnées ou `ohlcv_asset`. `ohlcv_timeframe`
déclare leur fréquence source. Une fréquence source plus fine est agrégée au
timeframe sélectionné avant le calcul des cibles : cinq barres H1 ne deviennent
jamais cinq barres M5. Avec un calendrier `expected_index` fourni à `Study`,
les OHLCV doivent être déjà agrégés au timeframe sélectionné.

Plusieurs `params_hash` pour une même méthode sont refusés, même sur des dates
disjointes. `configurations={"adx": "hash_choisi"}` permet de sélectionner une
configuration. Pour comparer deux configurations d'une même méthode, utiliser
des études séparées sur un support commun explicitement conservé ; le pivot
actuel ne produit pas deux colonnes pour une même méthode.

Source, côté de prix, ajustement et calendrier sont également vérifiés lorsqu'ils
figurent dans les sorties. Les nouveaux exports CLI transportent les métadonnées
disponibles. Un ancien parquet dépourvu de ces champs ne permet pas de certifier
ces dimensions : les renseigner pour une comparaison entièrement traçable.

Les duplications, mélanges d'identité, dates de régime absentes des prix et
index décalés en projection produisent des erreurs explicites. Les dates OHLCV
supplémentaires sont autorisées pour calculer la fin des horizons.

## Support commun et couverture

INCONNU, valeurs manquantes, chauffe explicitement signalée et barres partielles
sont exclues. Une valeur numérique non finie est également exclue. Le masque
d'une cible couvre tout son trajet, pas seulement ses extrémités.

L'étude calcule le support commun à toutes les features sélectionnées, par cible,
sans comprimer la grille avant le découpage temporel. Les résultats de cette
livraison portent sur ce support commun. Les comptes sur le support propre sont
publiés pour mesurer la couverture, sans constituer un second classement.

- `study_coverage.parquet` : effectifs propres/communs, taux de couverture et
  comptes d'exclusion (motifs pouvant se chevaucher).
- `study_common_support.parquet` : masque par timestamp et cible.
- `study_report.md` : section de couverture ; mention explicite si le support
  commun est entièrement vide, avec estimations indisponibles.

Les méthodes ne signalent pas encore toutes une chauffe distincte : le compteur
INCONNU reste séparé et ne présume pas que toute indisponibilité est une chauffe.

En projection, un segment contenant une barre invalide ou un trou attendu porte
`is_valid=False` et ses mesures sont indisponibles. Un identifiant réutilisé sur
des épisodes disjoints est rejeté. Les accords entre séries de labels excluent
les inconnus sur un support commun ; un tableau sans observation n'obtient pas
une qualité artificielle de zéro.

## Vérifications

Les [tests du contrat](../../tests/test_comparison_contract.py) couvrent les
ambiguïtés d'identité, les alias tick, les grilles et calendriers, la propagation
de partialité, les trous dans les cibles, l'alignement des projections et les
exports de support. Les tests préexistants de perturbation du futur conservent
désormais des OHLC cohérents après modification des prix.

## Calendriers locaux et changements d'heure

`regime_lib.core.temporal.session_grid` génère un `expected_index` UTC à partir
de dates locales, d'un fuseau IANA et des heures d'ouverture/clôture. Exemple :

```python
from regime_lib.core.temporal import session_grid
schedule = session_grid(
    "2026-03-01", "2026-03-31", timezone="America/New_York",
    opens="09:30", closes="16:00", timeframe="M30",
    holidays=["2026-03-17"],
)
```

Les jours de semaine valent lundi=0 à dimanche=6 (défaut lundi–vendredi).
Les dates sont celles d'ouverture ; une clôture <= à l'ouverture est le lendemain.
Les bornes locales suivent le DST ; les heures ambiguës/inexistantes sont refusées.
L'intervalle est ouvert à l'ouverture, fermé à la clôture. Fournir les jours fériés
et fermetures exceptionnelles : aucun calendrier de bourse n'est inventé.
Les bornes doivent être alignées sur la grille UTC du timeframe ; choisir une
fréquence plus fine sinon. La grille s'utilise dans `resample(expected_index=...)`
pour les barres source et dans `Study(expected_index=...)` pour les barres étudiées.
Une agrégation D1/W1 reste ancrée UTC ; ce helper ne crée pas de bougies journalières
à clôture locale variable.
