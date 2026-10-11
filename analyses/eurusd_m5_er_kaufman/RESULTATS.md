# Résultats — calibration `er_kaufman` EURUSD M5

Exécution du 2026-10-12 selon [PROTOCOLE.md](PROTOCOLE.md), sans
modification, script [calibrer.py](calibrer.py). Sorties brutes dans
`output/` (non versionné).

Données : 85 188 barres M5, dont 8 436 de warmup avant 2024. Couverture
valide sur 2024 : 99,1 % pour tous les candidats.

## Paramètre retenu

`n_er = 20`, `fenetre = 6000` (défaut précédent : `n_er = 20`,
`fenetre = 3000`), écrit dans `eurusd.yaml` sous
`methodes.er_kaufman.timeframes.M5` par application de la règle d'adoption.
**Ce choix n'est pas informatif** : voir la lecture.

## Score sur 2024

S = log médiane(ER_fwd | TENDANCE) − log médiane(ER_fwd | CHOP), ER_fwd sur
les 12 barres suivantes. Accord = part des TENDANCE suivies d'un rendement à
1 h dans le sens annoncé (IC 95 % ≈ ± 0,6 point).

| n_er | fenetre | S | S stratifié (diag.) | Accord |
|---|---|---|---|---|
| **20** | **6000** | **+0,025** | +0,022 | 47,9 % |
| 20 | 3000 | +0,021 | +0,019 | 47,9 % |
| 20 | 1500 | +0,018 | +0,017 | 47,9 % |
| 10 | 6000 | +0,016 | +0,010 | 48,1 % |
| 10 | 1500 | +0,011 | +0,007 | 48,1 % |
| 10 | 3000 | +0,010 | +0,007 | 48,1 % |
| 5 | 6000 | 0,000 | +0,008 | 48,0 % |
| 5 | 3000 | 0,000 | +0,008 | 48,1 % |
| 5 | 1500 | 0,000 | +0,004 | 48,2 % |
| 40 | 6000 | −0,003 | −0,002 | 48,5 % |
| 40 | 3000 | −0,007 | −0,006 | 48,5 % |
| 40 | 1500 | −0,008 | −0,006 | 48,4 % |
| 80 | 6000 | −0,036 | −0,031 | 48,3 % |
| 80 | 1500 | −0,038 | −0,032 | 48,2 % |
| 80 | 3000 | −0,038 | −0,033 | 48,2 % |

Les S nuls à `n_er = 5` viennent de médianes identiques : ER_fwd prend des
valeurs discrètes (pas de cotation).

## Validation glissante

| Test | Retenu sur le train | S train | S test | Rang test (/15) | Regret | Accord test |
|---|---|---|---|---|---|---|
| 2024-03..04 | 20 / 3000 | 0,057 | +0,014 | 11 | 0,051 | 48,5 % |
| 2024-05..06 | 10 / 6000 | 0,037 | −0,046 | 15 | 0,079 | 49,4 % |
| 2024-07..08 | 20 / 6000 | 0,036 | −0,027 | 9 | 0,047 | 47,1 % |
| 2024-09..10 | 20 / 6000 | 0,021 | +0,060 | 2 | 0,001 | 47,1 % |
| 2024-11..12 | 20 / 6000 | 0,027 | +0,000 | 2 | 0,000 | 47,4 % |

## Lecture

1. **Pas de persistance de l'efficience.** Au mieux, l'efficience médiane de
   l'heure suivante est 2,5 % plus élevée après TENDANCE qu'après CHOP
   (à comparer à un facteur 2,4 pour l'ATR). D'un bloc à l'autre, S change de
   signe (−0,05 à +0,06), et la sélection sur le train est au niveau du
   hasard. Les paramètres ne sont pas identifiables par ce critère.
2. **Léger retour en arrière après TENDANCE.** Pour tous les candidats et
   tous les blocs, le rendement à 1 h va dans le sens annoncé dans 47 à 49 %
   des cas, significativement sous 50 %. C'est un **diagnostic exploratoire**
   (non choisi comme critère), mesuré avant coûts : le spread EURUSD n'est
   pas pris en compte et il ne s'agit pas d'un avantage exploitable démontré.
3. Conséquence : en M5, un régime TENDANCE `er_kaufman` décrit l'efficience
   **passée** du trajet, sans valeur prédictive démontrée sur l'heure
   suivante. L'horizon (K = 12) a pu jouer ; un horizon plus long ou un
   timeframe supérieur relèverait d'un nouveau protocole.

## Limites

- Aucun test hors échantillon (décision du 2026-10-12).
- Un seul horizon (1 h) ; `q_chop` et `q_tendance` non calibrés.
- Discrétisation des prix Bid aux heures calmes (barres plates sans tick).
