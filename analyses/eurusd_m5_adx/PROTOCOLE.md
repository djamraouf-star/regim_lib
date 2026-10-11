# Calibration `adx` — EURUSD M5

Protocole figé le 2026-10-12, **avant tout calcul de résultat**. Toute
modification ultérieure est datée et justifiée dans le journal.

## Données

Identiques à la [calibration ATR M5](../eurusd_m5_atr_volatility/PROTOCOLE.md)
(source, `combler_sans_tick="60min"`, M5, warmup jusqu'au 2023-12-31,
évaluation 2024, lookahead début 2025).

## Ce qui est calibré

Les seuils de l'ADX sont absolus. Un critère de séparation les pousserait
vers les extrêmes ; ils sont donc **déduits des données** et non optimisés :

| Paramètre | Valeur | Statut |
|---|---|---|
| `n_adx` | 5, 7, 14, 28, 56 | calibré |
| `seuil_range` | quantile 40 % de l'ADX sur le train | déduit, pour chaque `n_adx` |
| `seuil_tendance` | quantile 60 % de l'ADX sur le train | déduit, pour chaque `n_adx` |

Les quantiles sont pris sur les barres valides (non partielles, ADX défini)
du train. Ces proportions (40 / 20 / 40) sont celles fixées pour
`er_kaufman`. Les seuils sont arrondis à 0,1.

## Cible et critère

Identiques à [er_kaufman M5](../eurusd_m5_er_kaufman/PROTOCOLE.md) :

- cible : efficience des 12 barres suivantes (ER_fwd), barres contiguës ;
- TENDANCE = `TENDANCE_HAUSSIERE` ∪ `TENDANCE_BAISSIERE` ; pour l'ADX,
  l'absence de tendance est `RANGE` (ADX bas) et non `CHOP` (transition) ;
- **S = log médiane(ER_fwd | TENDANCE) − log médiane(ER_fwd | RANGE)**,
  global, sans stratification.

Diagnostics rapportés sans servir au choix : S stratifié par heure de New
York, accord directionnel à 1 h (signe de +DI − −DI), proportions de
classes obtenues sur la période évaluée.

## Validation glissante (2024 uniquement)

Mêmes blocs et même purge que pour l'ATR. Pour chaque pli, les seuils sont
recalculés **sur le train du pli seulement**, puis appliqués au test.

## Règle d'adoption

- Paramètre final : le `n_adx` de S maximal sur 2024, avec ses seuils
  calculés sur toute l'année 2024.
- Écrit dans `eurusd.yaml` sous `methodes.adx.timeframes.M5`
  (`n_adx`, `seuil_range`, `seuil_tendance`). Les seuils généraux de
  l'exemple existant (22 / 28) restent inchangés pour les autres timeframes.
- Optimum au bord de la grille : signalé ; toute extension est consignée.
- Si, comme pour `er_kaufman`, aucun candidat ne se distingue (S proche de
  zéro et instable entre blocs), le choix est écrit mais déclaré non
  informatif.

## Limites déclarées d'avance

- Aucun test hors échantillon indépendant.
- Un seul horizon (1 h).
- Les seuils sur toute l'année 2024 utilisent la période évaluée : pour le
  choix final, ils sont donc ajustés « en échantillon » ; la validation
  glissante mesure ce que coûte ce biais.

## Journal des modifications

- 2026-10-12 : version initiale.
