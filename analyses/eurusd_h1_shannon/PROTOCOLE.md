# Calibration `shannon` — EURUSD H1

Protocole figé le 2026-10-13, **avant tout calcul de résultat**. Toute
modification ultérieure est datée et justifiée dans le journal.

Ce protocole décline celui de [shannon M5](../eurusd_m5_shannon/PROTOCOLE.md)
(lecture de l'indicateur, quantiles, validation glissante, règle
d'adoption) ; seules les différences sont listées ici.

## Données

Source, `combler_sans_tick="60min"`, warmup (jusqu'au 2023-12-31),
évaluation 2024 et lookahead début 2025 identiques à la
[calibration ATR M5](../eurusd_m5_atr_volatility/PROTOCOLE.md). Barres
H1 obtenues par agrégation des ticks (M1) puis `resample` ; pas d'une
barre : 1 heure.

Environ 6 000 barres en 2024 (≈ 1 000 par bloc de test de deux mois). La grille est **réduite par rapport à M5** : 20 classes sur 24 à 50 observations n'ont pas de sens, et la fenêtre 400 (≈ 17 jours) dépasse ce qu'un warmup de six semaines et des blocs de deux mois permettent d'évaluer proprement. La fenêtre 24 (un jour) remplace 400.

## Ce qui est calibré

| Paramètre | Valeur | Statut |
|---|---|---|
| `fenetre` | 24, 50, 100, 200 | calibré |
| `n_bins` | 5, 10 | calibré |
| `seuil_ordonne` | quantile 40 % de H sur le train | déduit, pour chaque couple |
| `seuil_desordre` | quantile 60 % de H sur le train | déduit, pour chaque couple |

Soit 8 candidats.

## Cible et critère

- horizon : **12 barres H1** (option « même nombre de barres » : l'échelle
  de l'horizon suit celle de l'indicateur), soit 12 heures ; barres
  contiguës exigées ;
- cible : efficience ER_fwd des 12 barres suivantes ;
- **S = log médiane(ER_fwd | ORDONNE) − log médiane(ER_fwd | DESORDRE)**,
  global, **sans conditionnement par session** (décision du 2026-10-13 : la
  session est une étude séparée). Seuils et critère sont donc calculés sur
  toutes les heures confondues.

Diagnostics sans rôle dans le choix : S stratifié par heure de New York,
rapport des médianes de `vol_fwd_12` DESORDRE / ORDONNE, proportions de
classes.

## Validation, adoption, limites

Identiques à M5 : blocs de deux mois, seuils recalculés sur le train de
chaque pli, adoption du S signé maximal sur 2024, déclaration « non
informatif » si le meilleur S est ≤ 0 ou proche de zéro et instable. Écrit
dans `eurusd.yaml` sous `methodes.shannon.timeframes.H1`.

Limites : pas de test hors échantillon ; un seul horizon (12 barres) ;
maximum de 8 candidats sur une même cible, donc biaisé à la hausse ;
les résultats ne sont pas comparables en valeur absolue entre timeframes
(horizons différents en durée).

## Journal des modifications

- 2026-10-13 : version initiale.
