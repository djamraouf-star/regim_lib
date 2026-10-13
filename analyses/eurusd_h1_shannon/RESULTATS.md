# Résultats — calibration `shannon` EURUSD H1

Exécution du 2026-10-13 selon [PROTOCOLE.md](PROTOCOLE.md), sans
modification, script [calibrer.py](../eurusd_shannon_multi_tf/calibrer.py)
(argument `H1`). Sorties brutes dans `output/` (non versionné). Le script
vérifie que sa classification coïncide avec celle du détecteur pour le
candidat retenu. Un premier lancement des trois timeframes en parallèle a été
tué par le système (mémoire) avant tout résultat ; ils ont été relancés l'un
après l'autre, sans changement de protocole ni de script.

Données : 7 099 barres H1 (6 250 en 2024).

## Paramètre retenu

`fenetre = 100`, `n_bins = 5`, `seuil_ordonne = 0,60`,
`seuil_desordre = 0,67`, écrit dans `eurusd.yaml` sous
`methodes.shannon.timeframes.H1` par application de la règle d'adoption.
**Ce choix n'est pas informatif** : voir la lecture.

## Score sur 2024

S = log médiane(ER_fwd | ORDONNE) − log médiane(ER_fwd | DESORDRE), ER_fwd
sur les 12 barres suivantes (12 heures). Critère global, sans conditionnement
par session. Rapport vol = médiane de `vol_fwd_12` en DESORDRE / ORDONNE
(diagnostic).

| fenetre | n_bins | seuils (ordonné / désordre) | S | S stratifié (diag.) | Rapport vol |
|---|---|---|---|---|---|
| **100** | 5 | 0,60 / 0,67 | +0,087 | +0,085 | 1,04 |
| 24 | 10 | 0,79 / 0,83 | +0,066 | +0,071 | 0,96 |
| 24 | 5 | 0,79 / 0,85 | +0,060 | +0,060 | 0,95 |
| 100 | 10 | 0,66 / 0,72 | +0,043 | +0,047 | 1,03 |
| 200 | 5 | 0,49 / 0,56 | +0,024 | +0,029 | 1,05 |
| 200 | 10 | 0,57 / 0,63 | +0,015 | +0,010 | 1,12 |
| 50 | 5 | 0,70 / 0,76 | +0,011 | +0,021 | 1,02 |
| 50 | 10 | 0,74 / 0,80 | +0,007 | +0,012 | 1,00 |

## Validation glissante

| Test | Retenu sur le train | S train | S test | Rang test (/8) | Regret |
|---|---|---|---|---|---|
| 2024-03,,04 | 50 / 10 | +0,101 | +0,110 | 1 | 0,000 |
| 2024-05,,06 | 50 / 10 | +0,119 | -0,102 | 7 | 0,202 |
| 2024-07,,08 | 24 / 10 | +0,047 | -0,007 | 6 | 0,106 |
| 2024-09,,10 | 100 / 10 | +0,054 | +0,039 | 6 | 0,264 |
| 2024-11,,12 | 24 / 10 | +0,075 | +0,054 | 6 | 0,126 |

## Lecture

1. **S le plus élevé des quatre timeframes (+0,087), mais non fiable.** Le S
   de test du candidat choisi sur le train va de −0,102 à +0,110 d'un bloc à
   l'autre (3 positifs, 2 négatifs ou nuls), avec un regret de 0,00 à 0,26 :
   la sélection n'est pas stable. Chaque bloc ne contient qu'environ 1 000
   barres, dont 40 % par classe ; les écarts de S entre candidats sont de
   l'ordre du bruit d'échantillonnage.
2. **Volatilité.** Rapport entre 0,95 et 1,12, de signe variable : aucun
   niveau de volatilité n'est associé aux classes.
3. Le maximum de 8 candidats est biaisé à la hausse. Aucun test hors
   échantillon n'est possible avec 6 250 barres d'évaluation.

## Limites

- Aucun test hors échantillon ; un seul horizon (12 barres).
- Seuils sur toute l'année 2024, ajustés en échantillon.
- Critère global : la session n'est pas prise en compte (étude séparée).
- Les S ne sont pas comparables en valeur absolue entre timeframes
  (horizons de durées différentes).
