# Résultats — calibration `shannon` EURUSD M1

Exécution du 2026-10-13 selon [PROTOCOLE.md](PROTOCOLE.md), sans
modification, script [calibrer.py](../eurusd_shannon_multi_tf/calibrer.py)
(argument `M1`). Sorties brutes dans `output/` (non versionné). Le script
vérifie que sa classification coïncide avec celle du détecteur pour le
candidat retenu. Un premier lancement des trois timeframes en parallèle a été
tué par le système (mémoire) avant tout résultat ; ils ont été relancés l'un
après l'autre, sans changement de protocole ni de script.

Données : 425 916 barres M1 (374 982 en 2024).

## Paramètre retenu

`fenetre = 400`, `n_bins = 5`, `seuil_ordonne = 0,56`,
`seuil_desordre = 0,62`, écrit dans `eurusd.yaml` sous
`methodes.shannon.timeframes.M1` par application de la règle d'adoption.
**Ce choix n'est pas informatif** : voir la lecture.

## Score sur 2024

S = log médiane(ER_fwd | ORDONNE) − log médiane(ER_fwd | DESORDRE), ER_fwd
sur les 12 barres suivantes (12 minutes). Critère global, sans conditionnement
par session. Rapport vol = médiane de `vol_fwd_12` en DESORDRE / ORDONNE
(diagnostic).

| fenetre | n_bins | seuils (ordonné / désordre) | S | S stratifié (diag.) | Rapport vol |
|---|---|---|---|---|---|
| **400** | 5 | 0,56 / 0,62 | +0,028 | +0,027 | 1,08 |
| 400 | 20 | 0,70 / 0,75 | +0,026 | +0,020 | 1,14 |
| 200 | 10 | 0,72 / 0,77 | +0,025 | +0,025 | 1,11 |
| 400 | 10 | 0,64 / 0,69 | +0,023 | +0,018 | 1,09 |
| 200 | 5 | 0,66 / 0,72 | +0,023 | +0,025 | 1,10 |
| 200 | 20 | 0,76 / 0,81 | +0,023 | +0,023 | 1,18 |
| 100 | 20 | 0,81 / 0,84 | +0,016 | +0,014 | 1,40 |
| 100 | 10 | 0,79 / 0,83 | +0,015 | +0,016 | 1,30 |
| 100 | 5 | 0,75 / 0,80 | +0,011 | +0,014 | 1,25 |
| 50 | 5 | 0,81 / 0,86 | +0,009 | +0,015 | 1,32 |
| 50 | 20 | 0,82 / 0,85 | +0,009 | +0,013 | 1,52 |
| 50 | 10 | 0,83 / 0,87 | +0,006 | +0,011 | 1,39 |

## Validation glissante

| Test | Retenu sur le train | S train | S test | Rang test (/12) | Regret |
|---|---|---|---|---|---|
| 2024-03,,04 | 100 / 10 | +0,072 | -0,051 | 10 | 0,089 |
| 2024-05,,06 | 400 / 10 | +0,045 | +0,013 | 6 | 0,030 |
| 2024-07,,08 | 400 / 5 | +0,034 | +0,036 | 3 | 0,004 |
| 2024-09,,10 | 400 / 5 | +0,036 | +0,031 | 3 | 0,003 |
| 2024-11,,12 | 400 / 5 | +0,036 | +0,011 | 10 | 0,030 |

## Lecture

1. **Effet très faible mais le plus régulier des trois.** Le meilleur S est
   +0,028 (efficience médiane 2,9 % plus haute après ORDONNE) ; le S de test
   est positif dans 4 blocs sur 5 (de −0,051 à +0,036), et la sélection sur
   le train se classe 3ᵉ à 10ᵉ sur 12. C'est un signal qui n'est pas exclu, pas un
   signal démontré : le maximum de 12 candidats est biaisé à la hausse et
   « proche de zéro » est jugé ici sans seuil chiffré fixé d'avance.
2. **Volatilité.** Le rapport de volatilité future DESORDRE / ORDONNE va de
   1,06 à 1,52 selon la fenêtre, et monte pour les petites fenêtres (50 à 100).
   Ce n'est qu'un diagnostic, hors critère : une calibration sur la
   volatilité future serait un autre protocole, et ne doit pas être décidée
   après avoir vu ce chiffre.
3. Avertissement de données : barres M1 très discrétisées (Bid) et nombreuses
   barres plates aux heures calmes ; l'efficience sur 12 minutes y est
   sensible.

## Limites

- Aucun test hors échantillon ; un seul horizon (12 barres).
- Seuils sur toute l'année 2024, ajustés en échantillon.
- Critère global : la session n'est pas prise en compte (étude séparée).
- Les S ne sont pas comparables en valeur absolue entre timeframes
  (horizons de durées différentes).
