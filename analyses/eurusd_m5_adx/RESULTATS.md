# Résultats — calibration `adx` EURUSD M5

Exécution du 2026-10-12 selon [PROTOCOLE.md](PROTOCOLE.md), sans
modification, script [calibrer.py](calibrer.py). Sorties brutes dans
`output/` (non versionné).

## Paramètre retenu

`n_adx = 7`, `seuil_range = 28.1`, `seuil_tendance = 34.6` (quantiles
40 % / 60 % de l'ADX(7) sur 2024). Défaut actuel : 14 / 20 / 25 ; exemple
EURUSD général : 22 / 28.

Écrit dans `eurusd.yaml` sous `methodes.adx.timeframes.M5`.

## Score sur 2024

S = log médiane(ER_fwd | TENDANCE) − log médiane(ER_fwd | RANGE), ER_fwd sur
les 12 barres suivantes. Accord : part des TENDANCE suivies d'un rendement à
1 h dans le sens annoncé (IC 95 % ≈ ± 0,6 point).

| n_adx | Seuils (range / tendance) | S | S stratifié (diag.) | Accord |
|---|---|---|---|---|
| 5 | 33,9 / 41,4 | +0,033 | +0,026 | 47,9 % |
| **7** | **28,1 / 34,6** | **+0,042** | +0,037 | 47,9 % |
| 14 | 19,5 / 24,2 | +0,030 | +0,030 | 48,1 % |
| 28 | 13,9 / 17,4 | +0,009 | +0,010 | 48,4 % |
| 56 | 10,1 / 12,8 | −0,008 | −0,005 | 48,2 % |

Proportions obtenues : 40 % TENDANCE, 40 % RANGE, 20 % CHOP pour tous les
candidats.

## Validation glissante

Seuils recalculés sur le train de chaque pli.

| Test | Retenu | Seuils | S train | S test | Rang test (/5) | Regret | Accord test |
|---|---|---|---|---|---|---|---|
| 2024-03..04 | 7 | 28,6 / 34,9 | 0,067 | +0,084 | 1 | 0,000 | 48,5 % |
| 2024-05..06 | 7 | 28,4 / 34,9 | 0,075 | +0,015 | 2 | 0,008 | 50,1 % |
| 2024-07..08 | 7 | 28,6 / 35,2 | 0,061 | −0,013 | 4 | 0,031 | 46,9 % |
| 2024-09..10 | 7 | 28,2 / 34,6 | 0,039 | +0,058 | 4 | 0,012 | 47,3 % |
| 2024-11..12 | 7 | 28,2 / 34,6 | 0,043 | +0,045 | 1 | 0,000 | 45,0 % |

## Lecture

1. **Choix stable, effet faible.** `n_adx = 7` est retenu dans les cinq plis,
   avec des seuils quasi constants. L'optimum est intérieur (5 et 14 font
   moins bien). Mais l'effet est minime : l'efficience médiane de l'heure
   suivante est 4 % plus élevée après TENDANCE qu'après RANGE (ATR : facteur
   2,4), positive sur 4 blocs sur 5. Le choix est **faiblement
   informatif** ; un peu mieux que `er_kaufman` (2,5 %, instable).
2. **Même retour en arrière qu'avec `er_kaufman`** : 47,9 % d'accord
   directionnel à 1 h, sous 50 % sur la plupart des blocs. Diagnostic
   exploratoire, avant coûts.
3. Les seuils dépendent fortement de `n_adx` (ADX(7) ≈ 28 à 35, ADX(56) ≈ 10
   à 13) : des seuils absolus fixés sans référence à `n_adx` et au timeframe
   (20 / 25 par défaut) donnent des proportions de classes arbitraires.

## Écriture dans le profil

`n_adx`, `seuil_range` et `seuil_tendance` étaient déclarés
`per_timeframe=False`. Décision du 2026-10-12 : ils deviennent réglables par
timeframe dans `adx.py`, puisque des seuils absolus dépendent de `n_adx` et du
timeframe. Les seuils généraux de l'exemple EURUSD (22 / 28) restent
appliqués aux autres timeframes.

## Limites

- Aucun test hors échantillon (décision du 2026-10-12).
- Seuils finaux calculés sur l'année évaluée (en échantillon) ; la
  validation glissante, avec seuils hors échantillon, donne des scores
  comparables.
- Un seul horizon (1 h).
