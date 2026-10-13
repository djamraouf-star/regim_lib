# Résultats — calibration `shannon` EURUSD M15

Exécution du 2026-10-13 selon [PROTOCOLE.md](PROTOCOLE.md), sans
modification, script [calibrer.py](../eurusd_shannon_multi_tf/calibrer.py)
(argument `M15`). Sorties brutes dans `output/` (non versionné). Le script
vérifie que sa classification coïncide avec celle du détecteur pour le
candidat retenu. Un premier lancement des trois timeframes en parallèle a été
tué par le système (mémoire) avant tout résultat ; ils ont été relancés l'un
après l'autre, sans changement de protocole ni de script.

Données : 28 396 barres M15 (25 000 en 2024).

## Paramètre retenu

`fenetre = 100`, `n_bins = 5`, `seuil_ordonne = 0,65`,
`seuil_desordre = 0,72`, écrit dans `eurusd.yaml` sous
`methodes.shannon.timeframes.M15` par application de la règle d'adoption.
**Ce choix n'est pas informatif** : voir la lecture.

## Score sur 2024

S = log médiane(ER_fwd | ORDONNE) − log médiane(ER_fwd | DESORDRE), ER_fwd
sur les 12 barres suivantes (3 heures). Critère global, sans conditionnement
par session. Rapport vol = médiane de `vol_fwd_12` en DESORDRE / ORDONNE
(diagnostic).

| fenetre | n_bins | seuils (ordonné / désordre) | S | S stratifié (diag.) | Rapport vol |
|---|---|---|---|---|---|
| **100** | 5 | 0,65 / 0,72 | +0,042 | +0,038 | 0,98 |
| 100 | 20 | 0,75 / 0,80 | +0,031 | +0,031 | 0,99 |
| 100 | 10 | 0,71 / 0,77 | +0,023 | +0,027 | 0,98 |
| 200 | 20 | 0,67 / 0,73 | +0,018 | +0,015 | 1,03 |
| 200 | 10 | 0,60 / 0,68 | +0,012 | +0,011 | 1,02 |
| 50 | 10 | 0,79 / 0,84 | +0,005 | +0,005 | 0,98 |
| 50 | 20 | 0,80 / 0,84 | +0,004 | +0,008 | 0,97 |
| 50 | 5 | 0,76 / 0,82 | +0,001 | +0,005 | 1,00 |
| 200 | 5 | 0,53 / 0,62 | -0,002 | -0,005 | 1,03 |
| 400 | 10 | 0,49 / 0,56 | -0,013 | -0,021 | 1,11 |
| 400 | 5 | 0,42 / 0,47 | -0,017 | -0,024 | 1,09 |
| 400 | 20 | 0,58 / 0,63 | -0,019 | -0,029 | 1,11 |

## Validation glissante

| Test | Retenu sur le train | S train | S test | Rang test (/12) | Regret |
|---|---|---|---|---|---|
| 2024-03,,04 | 200 / 20 | +0,082 | +0,040 | 1 | 0,000 |
| 2024-05,,06 | 200 / 10 | +0,056 | -0,089 | 12 | 0,083 |
| 2024-07,,08 | 100 / 5 | +0,008 | +0,044 | 9 | 0,061 |
| 2024-09,,10 | 100 / 5 | +0,016 | +0,100 | 2 | 0,024 |
| 2024-11,,12 | 100 / 5 | +0,031 | +0,083 | 3 | 0,046 |

## Lecture

1. **Pas de signal.** Le meilleur S est +0,042 ; sept candidats sur douze sont
   entre −0,02 et +0,03, et les fenêtres 400 sont négatives. Le S de test
   du candidat choisi change de signe (−0,089 à +0,100) et son rang en test va
   de 1 à 12 sur 12 : les paramètres ne sont pas identifiables.
2. **Volatilité.** Rapport entre 0,97 et 1,11, sans lien net avec la fenêtre.
3. La fenêtre 100 est choisie pour M15 comme pour H1, mais avec des `n_bins`
   et des seuils différents : rien n'autorise à y voir une structure commune.

## Limites

- Aucun test hors échantillon ; un seul horizon (12 barres).
- Seuils sur toute l'année 2024, ajustés en échantillon.
- Critère global : la session n'est pas prise en compte (étude séparée).
- Les S ne sont pas comparables en valeur absolue entre timeframes
  (horizons de durées différentes).
