# Calibration `shannon` — EURUSD M5

Protocole figé le 2026-10-13, **avant tout calcul de résultat**. Toute
modification ultérieure est datée et justifiée dans le journal.

## Données

Identiques à la [calibration ATR M5](../eurusd_m5_atr_volatility/PROTOCOLE.md)
(source, `combler_sans_tick="60min"`, M5, warmup jusqu'au 2023-12-31,
évaluation 2024, lookahead début 2025).

## Ce que mesure l'indicateur

L'entropie de Shannon normalisée est calculée sur l'histogramme des
log-rendements d'une fenêtre. Elle ne dépend pas de l'ordre temporel des
rendements : elle mesure la dispersion de leur distribution (une fenêtre où
quelques rendements extrêmes étirent l'intervalle a une entropie basse). Le
nom `ORDONNE` est donc une étiquette, pas une propriété de tendance. Cette
réserve est testée, non supposée : l'efficience future est le critère
principal, la volatilité future un diagnostic.

## Ce qui est calibré

| Paramètre | Valeur | Statut |
|---|---|---|
| `fenetre` | 50, 100, 200, 400 | calibré |
| `n_bins` | 5, 10, 20 | calibré |
| `seuil_ordonne` | quantile 40 % de H sur le train | déduit, pour chaque couple |
| `seuil_desordre` | quantile 60 % de H sur le train | déduit, pour chaque couple |

Soit 12 candidats. H dépend fortement de `n_bins` : des seuils absolus ne
sont pas comparables entre candidats, d'où les quantiles (proportions
40 / 20 / 40, comme `er_kaufman` et `adx`). Les quantiles sont pris sur les
barres valides (non partielles, H défini) du train. Seuils arrondis à 0,01.

## Cible et critère

- cible : efficience des 12 barres suivantes (ER_fwd = |net| / chemin),
  barres contiguës, comme [er_kaufman M5](../eurusd_m5_er_kaufman/PROTOCOLE.md) ;
- **S = log médiane(ER_fwd | ORDONNE) − log médiane(ER_fwd | DESORDRE)**,
  global, sans stratification. Le signe est fixé d'avance par la lecture
  naturelle des étiquettes (ordonné = plus efficient).

Diagnostics rapportés sans servir au choix : S stratifié par heure de New
York ; rapport des médianes de `vol_fwd_12` (écart-type des 12 rendements
suivants) DESORDRE / ORDONNE ; proportions de classes.

## Validation glissante (2024 uniquement)

Mêmes blocs et même purge que pour l'ATR. Pour chaque pli, les seuils sont
recalculés **sur le train du pli seulement**, puis appliqués au test.

## Règle d'adoption

- Paramètres finaux : le couple (`fenetre`, `n_bins`) de S maximal (signé)
  sur 2024, avec ses seuils calculés sur toute l'année 2024.
- Écrit dans `eurusd.yaml` sous `methodes.shannon` (`fenetre`, `n_bins`,
  `seuil_ordonne`, `seuil_desordre`). La surcharge générale actuelle
  (fenêtre 100, seuils 0,3 / 0,7) n'est calibrée sur rien ; elle est
  remplacée.
- Si le meilleur S est ≤ 0, ou proche de zéro et instable entre blocs, le
  choix est écrit mais déclaré **non informatif** (cas d'`er_kaufman`).
  Un S négatif significatif est rapporté tel quel, sans inverser les
  étiquettes après coup.
- Optimum au bord de la grille : signalé ; toute extension est consignée.

## Limites déclarées d'avance

- Aucun test hors échantillon indépendant.
- Un seul horizon (1 h).
- Seuils sur toute l'année 2024 : ajustés « en échantillon » ; la
  validation glissante mesure ce que coûte ce biais.
- Les 12 candidats sont comparés sur la même cible : le maximum est biaisé
  à la hausse.

## Journal des modifications

- 2026-10-13 : version initiale.
