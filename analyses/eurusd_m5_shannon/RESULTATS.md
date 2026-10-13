# Résultats — calibration `shannon` EURUSD M5

Exécution du 2026-10-13 selon [PROTOCOLE.md](PROTOCOLE.md), sans
modification, script [calibrer.py](calibrer.py). Sorties brutes dans
`output/` (non versionné). Le script vérifie que sa classification
coïncide avec celle du détecteur pour le candidat retenu.

Données : 85 188 barres M5 (mêmes que les autres calibrations M5).

## Paramètre retenu

`fenetre = 400`, `n_bins = 10`, `seuil_ordonne = 0,56`,
`seuil_desordre = 0,64`, écrit dans `eurusd.yaml` sous
`methodes.shannon.timeframes.M5` par application de la règle d'adoption.
Les seuils sont devenus réglables par timeframe (`per_timeframe`), comme
pour l'ADX. **Ce choix n'est pas informatif** : voir la lecture.

## Score sur 2024

S = log médiane(ER_fwd | ORDONNE) − log médiane(ER_fwd | DESORDRE), ER_fwd
sur les 12 barres suivantes. Rapport vol = médiane de `vol_fwd_12` en
DESORDRE / en ORDONNE (diagnostic).

| fenetre | n_bins | seuils (ordonné / désordre) | S | S stratifié (diag.) | Rapport vol |
|---|---|---|---|---|---|
| **400** | **10** | **0,56 / 0,64** | **+0,047** | +0,046 | 1,05 |
| 400 | 20 | 0,64 / 0,70 | +0,046 | +0,042 | 1,04 |
| 50 | 5 | 0,80 / 0,85 | +0,038 | +0,041 | 1,02 |
| 400 | 5 | 0,48 / 0,56 | +0,031 | +0,033 | 1,03 |
| 50 | 20 | 0,82 / 0,85 | +0,028 | +0,035 | 1,03 |
| 50 | 10 | 0,83 / 0,86 | +0,028 | +0,028 | 1,03 |
| 200 | 5 | 0,60 / 0,67 | +0,026 | +0,019 | 1,00 |
| 200 | 20 | 0,73 / 0,77 | +0,025 | +0,018 | 0,99 |
| 200 | 10 | 0,67 / 0,73 | +0,022 | +0,017 | 0,98 |
| 100 | 20 | 0,79 / 0,83 | +0,020 | +0,026 | 1,08 |
| 100 | 10 | 0,76 / 0,81 | +0,010 | +0,014 | 1,06 |
| 100 | 5 | 0,72 / 0,77 | +0,009 | +0,017 | 1,04 |

Proportions de classes : 38 à 42 % ORDONNE, 38 à 42 % DESORDRE.

## Validation glissante

| Test | Retenu sur le train | S train | S test | Rang test (/12) | Regret |
|---|---|---|---|---|---|
| 2024-03..04 | 200 / 5 | 0,103 | +0,010 | 12 | 0,084 |
| 2024-05..06 | 50 / 5 | 0,067 | −0,034 | 5 | 0,097 |
| 2024-07..08 | 400 / 20 | 0,039 | +0,062 | 8 | 0,090 |
| 2024-09..10 | 50 / 5 | 0,056 | −0,032 | 9 | 0,023 |
| 2024-11..12 | 50 / 5 | 0,040 | +0,015 | 7 | 0,065 |

## Lecture

1. **Pas de signal d'efficience.** Au mieux, l'efficience médiane de l'heure
   suivante est 4,8 % plus élevée après ORDONNE qu'après DESORDRE
   (à comparer à un facteur 2,4 pour la volatilité ATR). Les 12 candidats
   ont un S positif et faible ; en validation glissante S change de signe
   d'un bloc à l'autre et le candidat choisi sur le train se classe 5ᵉ à 12ᵉ
   sur 12 en test. Les paramètres ne sont pas identifiables par ce critère.
2. **Pas de signal de volatilité non plus.** Le rapport de volatilité future
   DESORDRE / ORDONNE est entre 0,98 et 1,08, donc l'entropie n'oppose pas
   deux niveaux de volatilité à 1 h. Ce n'est qu'un diagnostic, pas un
   critère ; il confirme que les étiquettes `ORDONNE` / `DESORDRE` ne
   décrivent ni tendance ni volatilité futures.
3. **Maximum biaisé.** Le meilleur de 12 candidats sur une même cible
   surestime l'effet ; S = 0,047 est compatible avec zéro.
4. Les seuils de l'exemple précédent (0,3 / 0,7) étaient hors de la plage
   observée de l'entropie (quantiles 40/60 de 0,56 / 0,64 pour fenêtre 400 /
   10 bins, 0,76 / 0,81 pour 100 / 10) : aucune barre en ORDONNE, et presque
   tout en NEUTRE ou en DESORDRE selon la fenêtre. Les seuils en quantiles
   corrigent les proportions de classes, pas la valeur prédictive.

## Limites

- Aucun test hors échantillon (décision du 2026-10-12).
- Un seul horizon (1 h) ; proportions 40/20/40 non calibrées.
- Seuils sur toute l'année 2024, ajustés en échantillon.
- Ce résultat porte sur la prédiction d'efficience et de volatilité à 1 h ;
  il ne dit rien d'autres usages (par exemple décrire la distribution
  passée des rendements).
