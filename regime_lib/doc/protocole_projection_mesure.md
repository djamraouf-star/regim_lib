# Contrat commun de projection et de mesure

Version : **1.0.0** — 4 octobre 2026.

Statut : spécification de référence pour le développement. Les garanties
décrites ici ne sont pas encore toutes implémentées. Ce document fixe les
conventions de l'étape 1 ; il ne certifie pas la conformité du code existant.

## Finalité et périmètre

L'API caractérise les mouvements des indices financiers à partir de différentes
familles d'indicateurs. Elle compare leurs représentations en les projetant
dans l'espace du prix selon cinq dimensions : amplitude, régularité du trajet,
durée, fréquence d'apparition et excursion opposée.

Les labels restent natifs à chaque méthode. Leurs noms, numéros et scores ne
constituent pas une échelle commune. Le protocole ne définit ni stratégie de
trading, ni backtest, ni gain réalisable, ni score global pondéré. PELT est une
référence descriptive optionnelle, pas une vérité objective à reproduire.

## 1. Identité, temps et validité

Une série est identifiée par l'actif, la source, la nature du prix (indice,
bid, ask, mid, etc.), les ajustements appliqués, le timeframe et le calendrier
de séances versionné. Deux sources ou conventions de prix ne sont pas fusionnées
implicitement. Une exécution ajoute la méthode, ses paramètres effectifs et sa
version d'apprentissage éventuelle.

Les timestamps sont uniques, strictement croissants et exprimés en UTC. Le
fuseau d'une source naïve doit être déclaré. Les barres portent leur instant de
clôture et couvrent [ouverture, clôture). Les bornes journalières et hebdomadaires
sont dérivées du calendrier déclaré, pas d'une durée UTC supposée constante.
Une conversion de convention conserve les bornes originales dans la traçabilité.

L'alignement est une jointure contrôlée sur l'identité et le timestamp, jamais
une association par position. Un doublon, une identité ambiguë, un timeframe
incompatible ou un décalage d'index provoque une erreur explicite.

Les prix doivent être finis, strictement positifs et respecter les relations
OHLC. Les volumes utilisés doivent être finis et non négatifs, avec leur nature
déclarée (transactions ou tailles de cotations). Une absence de volume n'exclut
pas une méthode qui n'en utilise pas.

Les observations partielles, invalides, en chauffe ou sans représentation
valide (dont INCONNU) sont signalées séparément. Aucun remplissage implicite des
prix, des labels ou des scores n'est autorisé. Un trou pendant une séance attendue
invalide toute fenêtre qui le traverse. Une fermeture prévue par le calendrier
n'est pas un trou ; le déplacement entre clôtures de séances reste dans le trajet.

## 2. Disponibilité de la représentation

Chaque sortie porte les informations conceptuelles suivantes (noms de schéma
à implémenter) :

| Champ | Signification |
|---|---|
| `observed_at` | Date de l'observation ou de la structure décrite |
| `available_at` | Première date à laquelle cette version du résultat est calculable |
| `information_end` | Dernière date des données utilisées, apprentissage compris |
| `temporal_mode` | `retrospective` ou `as_of` (information disponible à la date considérée) |
| `revision_id`, `supersedes` | Identité de version et éventuelle version remplacée |
| `fit_id` | Identité de l'apprentissage, si applicable |
| `score_kind` | Nature du score : probabilité, distance, constante, etc. |

En mode `as_of`, `information_end <= available_at` et la projection à t ne
consomme que les versions disponibles au plus tard à t. Les sorties déjà
publiées restent immuables ; les révisions sont de nouveaux événements.
Une latence est enregistrée, jamais effacée par un rattachement rétroactif.

Applications obligatoires aux corrections de l'audit :

- **C1 (VPIN)** : valeur publiée à la clôture du bucket seulement. Avant le
  premier bucket complet, sortie indisponible. Entre clôtures, une dernière
  valeur publiée peut être conservée avec sa date et son âge explicites ;
  cette règle appartient à la configuration. Aucun bucket incomplet ne reçoit
  sa valeur finale anticipée.
- **I1** : les métadonnées remplacent les exclusions par nom de méthode.
  Découper une sortie rétrospective ne change pas son mode temporel.
- **I9 (pivots)** : date du pivot, confirmation et révision sont distinctes.
  La version rétrospective finale et l'historique des événements disponibles
  au fil du temps constituent deux représentations distinctes.
- **I10 (HMM)** : l'apprentissage/décodage global est rétrospectif. Le mode
  `as_of` apprend normalisation et paramètres sur une période antérieure,
  puis applique un filtrage avant à paramètres fixes jusqu'au prochain
  apprentissage déclaré. Chaque apprentissage conserve ses diagnostics.
  Les identifiants d'états sont locaux au `fit_id` ; ils ne sont pas assimilés
  entre apprentissages par leur numéro.

Les résultats rétrospectifs et `as_of` sont présentés séparément. Les mesures
de trajet peuvent utiliser des prix ultérieurs : cette disponibilité tardive
de la mesure ne doit pas être confondue avec celle de la représentation.

## 3. Deux unités d'observation complémentaires

### Fenêtres communes

Pour chaque ancrage t éligible et horizon K >= 1, le trajet contient les
clôtures de t à t+K incluses : K intervalles et K+1 prix. Toutes les méthodes
utilisent les mêmes ancrages, prix et horizons. Leur représentation à t sert
à conditionner les distributions des mesures ; aucun résultat futur ne sert
à choisir la catégorie en mode `as_of`.

Un horizon est déclaré soit en nombre de barres, soit en durée écoulée. Pour
une durée, la borne finale doit exister exactement ; pas d'arrondi silencieux.
Un K identique sur H1 et M5 n'est pas une même durée. Les comparaisons entre
timeframes sont séparées ou effectuées sur une grille de prix commune déclarée.

### Épisodes des représentations

Un épisode catégoriel est une suite maximale de barres attendues consécutives
portant la même catégorie native valide. Son trajet va de la première à la
dernière clôture incluse. Une transition n'est attribuée implicitement à aucun
des deux épisodes. Un épisode de n barres contient n-1 intervalles mesurés.

Une invalidité, un trou inattendu ou un changement de `fit_id` coupe l'épisode.
Aux frontières de l'échantillon et après une coupure, les épisodes peuvent être
censurés : conserver les indicateurs de censure et exclure les épisodes censurés
des distributions principales de durée/amplitude complètes. Rapporter leur nombre.
Le masque commun ne doit jamais recoller deux fragments ni inventer un début.

Pour une représentation continue ou événementielle, l'adaptateur déclare sa règle
de construction d'épisodes et ses paramètres avant la comparaison. Il n'existe
aucun seuil universel sur `confidence`. Une méthode sans règle d'épisode participe
aux fenêtres communes ; sa fréquence d'épisodes reste non applicable.

Les statistiques de fenêtres et d'épisodes sont séparées. Les bornes des épisodes
peuvent différer entre méthodes : c'est un résultat à étudier. Chaque épisode
compte une fois, sans répéter son poids sur chacune de ses barres. Une éventuelle
pondération temporelle est un résultat distinct explicitement nommé.

## 4. Mesures de référence

Pour un trajet valide P_0, ..., P_K, poser x_i = P_i / P_0 - 1,
D = x_K et L = somme, pour i=1..K, de |x_i - x_(i-1)|.
Les ratios sont stockés sans multiplication par 100 ; le pourcentage est un
format d'affichage. Le prix de référence est P_0, sans notion d'entrée en position.

| Dimension | Définition v1 |
|---|---|
| Amplitude | `amplitude_close = max(x_i) - min(x_i)` ; publier aussi le déplacement signé D et son module pour distinguer étendue et progression nette |
| Régularité | `regularite = abs(D) / L` si L > 0 ; 0 si L = 0, avec `flat=true` |
| Durée | Publier K intervalles, K+1 observations et t_K - t_0 en secondes ; les durées de fenêtres sont fixées, celles des épisodes sont observées |
| Excursion opposée | Pour une orientation s dans {-1,+1}, `excursion_opposee = max(0, -min(s*x_i))` |
| Fréquence | Nombre de débuts d'épisodes observés de la catégorie / durée observable commune, affiché par 100 heures de séance valides |

Pour la fréquence, la durée observable est la somme des durées des barres
éligibles intersectées avec les séances, hors fermetures et trous. Un début
d'épisode est compté seulement si la barre précédente attendue est valide et
établit une transition. Un début inconnu à la frontière n'est pas compté ; un
début connu compte même si la fin est censurée. Le dénominateur inclut toutes
les catégories valides, pas seulement la catégorie étudiée. Publier aussi le
nombre de débuts exclus et le taux d'occupation (durée de la catégorie / durée
observable), qui est différent de la fréquence. Dénominateur nul : résultat
indisponible, jamais zéro.

L'orientation est obligatoirement identifiée :

- `realized` : s = signe(D), caractérisation rétrospective du trajet ;
- `declared` : orientation explicitement produite par la méthode à l'ancrage ;
- `both` : publier séparément les excursions pour s=+1 et s=-1, sans sélection.

Le profil descriptif de référence utilise `realized` ; si D=0, l'excursion
orientée est indisponible et les deux excursions sont publiées. Les résultats
de modes d'orientation différents ne sont pas agrégés. Une direction réalisée
ne devient jamais une direction réputée connue à l'ancrage.

Un trajet d'un seul prix (épisode d'une barre) a durée, amplitude et déplacement
nuls ; sa régularité et son excursion orientée sont indisponibles faute de
trajet. Les observations invalides ne sont jamais remplacées par zéro.

### Extension high/low et normalisations

Le profil de référence utilise les clôtures pour les cinq dimensions. Une
extension distincte `ohlc_extrema` mesure les excursions en incorporant P_0 et
les high/low des barres suivant l'ancrage jusqu'à la borne finale incluse.
Les high/low de la barre d'ancrage sont exclus, car antérieurs à sa clôture.
L'étendue est (maximum des prix retenus - minimum) / P_0. L'excursion opposée
pour s=+1 utilise le minimum ; pour s=-1 le maximum, toujours depuis P_0.
La régularité reste calculée sur les clôtures : OHLC ne fournit pas l'ordre
intra-barre des extrêmes. Un temps jusqu'à un extrême est seulement localisé
à sa barre, avec la première barre retenue en cas d'égalité.

Les ratios à P_0 sont la normalisation commune v1. Une normalisation par
volatilité est un profil supplémentaire avec estimateur, fenêtre et disponibilité
déclarés, communs aux méthodes. Les unités ne sont jamais mélangées ; les
résultats par indice restent accessibles même lorsqu'une synthèse est produite.

## 5. Support, comparaison et traçabilité

Construire le support principal par intersection des représentations valides
des méthodes comparées et de la disponibilité du trajet complet. Publier le
support propre à chaque méthode, le support commun et les exclusions par motif.
Changer le groupe de méthodes peut changer cette intersection : toute comparaison
de rapports exige la même liste d'ancrages ou la même empreinte de support.
Un support commun vide produit un rapport sans estimation, avec son motif.

Conserver pour chaque observation les bornes du trajet, les dates de disponibilité,
la catégorie native, le mode temporel, l'orientation et les valeurs mesurées.
Publier effectifs, quantiles (10 %, médiane, 90 %, interpolation linéaire),
couverture et censure. Les fenêtres chevauchantes restent identifiées ; leurs
effectifs bruts ne sont pas présentés comme des observations indépendantes.
Les modalités d'inférence temporelle et de correction multiple seront implémentées
à l'étape statistique, avant tout affichage de significativité.

Un renommage bijectif des catégories doit conserver toutes les mesures, à
renommage près. Les états HMM sont caractérisés par leurs distributions de prix,
pas appariés par leur numéro. Une représentation constante doit apparaître comme
telle : une pureté parfaite ne démontre pas une qualité supérieure.

Le manifeste contient : version du contrat, profil de mesure, mode d'observation,
orientation, horizons, calendrier, liste des méthodes, paramètres résolus,
identifiants d'apprentissage, règles d'adaptation et de validité, empreintes des
données et du support, version du code et dépendances. Toute modification de
formule, de bornes ou de dénominateur implique une nouvelle version du contrat.

## 6. Exemples numériques de référence

Prix espacés d'une heure, clôtures uniquement, orientation `realized`.
Les valeurs en pourcentage ci-dessous sont des affichages des ratios.

| Prix | Amplitude | Déplacement | Régularité | Durée | Excursion opposée |
|---|---:|---:|---:|---:|---:|
| 100, 102, 104 | 4 % | +4 % | 1 | 2 h | 0 % |
| 100, 98, 104 | 6 % | +4 % | 0,5 | 2 h | 2 % |
| 100, 102, 96 | 6 % | -4 % | 0,5 | 2 h | 2 % |
| 100, 102, 100 | 2 % | 0 % | 0 | 2 h | indisponible ; s=+1 : 0 %, s=-1 : 2 % |
| 100, 100, 100 | 0 % | 0 % | 0 (plat) | 2 h | indisponible ; deux excursions nulles |
| 100 | 0 % | 0 % | indisponible | 0 h | indisponible |

Pour 100, 98, 104 : L = 0,02 + 0,06 = 0,08, donc régularité =
0,04 / 0,08 = 0,5. Doubler tous les prix conserve les mesures normalisées.

Sur 10 heures de séance valides communes, deux débuts observés d'une catégorie
donnent une fréquence de 20 épisodes pour 100 heures. Si cette catégorie occupe
4 heures, son occupation est 40 %, indépendamment de cette fréquence.

Exemple OHLC : P_0=100, clôture suivante 102, high=105 et low=97 sur la barre
suivante. L'amplitude sur clôtures est 2 %, l'étendue OHLC est 8 % et l'excursion
opposée orientée à la hausse passe de 0 % sur clôtures à 3 % sur OHLC.

## 7. Critères d'acceptation des prochaines étapes

1. Les exemples numériques sont reproduits par des tests indépendants.
2. Renommer les catégories ou permuter les méthodes ne change pas les mesures.
3. Deux projections identiques donnent des mesures identiques.
4. Un index décalé, même de longueur identique, est rejeté.
5. Les invalidités, censures et fenêtres incomplètes sont visibles et jamais
   assimilées à des mouvements plats.
6. L'ajout ou la perturbation du futur conserve les sorties `as_of` déjà publiées,
   à paramètres et apprentissage historiques inchangés (VPIN, pivots, HMM inclus).
7. Une représentation rétrospective conserve ce statut après découpage.
8. Fenêtres, épisodes, orientations et profils de prix restent identifiables
   séparément dans les exports.

## 8. Écarts connus et ordre d'intégration

Dans le code actuel, [la caractérisation des segments](../projection/features.py)
utilise une amplitude high/low absolue et une efficience déplacement/étendue.
Cette efficience n'est pas la régularité définie ici. Les colonnes existantes
ne doivent pas changer silencieusement de signification : introduire les mesures
versionnées puis documenter la migration.

[La projection](../projection/projection.py) et
[les cibles d'étude](../study/targets.py) seront adaptées au même contrat.
Les termes « partition objective » et « vérité terrain » ne désignent pas une
supériorité intrinsèque de PELT ou d'une famille d'indicateurs.

Ordre de livraison : alignement/validité ; disponibilité temporelle (C1, I1,
I9, I10) ; calcul commun des mesures ; comparaison statistique ; reproductibilité.
Les références détaillées proviennent de
[l'audit du 4 octobre 2026](../../AUDIT_SYNTHETIQUE_2026-10-04.md).
