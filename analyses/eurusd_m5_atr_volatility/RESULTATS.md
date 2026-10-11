# Résultats — calibration `atr_volatility` EURUSD M5

Exécution du 2026-10-12 selon [PROTOCOLE.md](PROTOCOLE.md), script
[calibrer.py](calibrer.py). Sorties brutes dans `output/` (non versionné) :
`resultats_2024.csv`, `validation_glissante.csv`, `par_heure_retenu.csv`,
`provenance.json`.

Données : 85 188 barres M5, dont 8 436 de warmup avant 2024. Couverture
valide sur 2024 : 99,1 % pour tous les candidats.

Le critère a changé après une première lecture des résultats (voir le
journal du protocole) : le critère stratifié par heure a été remplacé par un
critère de volatilité globale, conforme à l'usage visé. Le choix initial
était `n_atr = 14`, `fenetre = 6000`.

## Paramètre retenu

`n_atr = 7`, `fenetre = 6000` (défaut précédent : `n_atr = 20`,
`fenetre = 6000`), écrit dans `eurusd.yaml` sous
`methodes.atr_volatility.timeframes.M5`.

## Score sur 2024

S = écart log des médianes de volatilité future (1 h) entre HAUT et BAS.
Référence horloge sans prix : S = 0,765.

| n_atr | fenetre | S | Excès / horloge | S stratifié (diag.) |
|---|---|---|---|---|
| **7** | **6000** | **0,895** | **+0,131** | 0,594 |
| 5 | 6000 | 0,893 | +0,128 | 0,550 |
| 7 | 3000 | 0,885 | +0,120 | 0,577 |
| 5 | 3000 | 0,883 | +0,118 | 0,536 |
| 3 | 6000 | 0,880 | +0,116 | 0,482 |
| 7 | 1500 | 0,878 | +0,113 | 0,581 |
| 5 | 1500 | 0,874 | +0,109 | 0,512 |
| 14 | 6000 | 0,873 | +0,109 | 0,628 |
| 3 | 3000 | 0,867 | +0,102 | 0,453 |
| 14 | 3000 | 0,856 | +0,092 | 0,608 |
| 3 | 1500 | 0,855 | +0,090 | 0,434 |
| 14 | 1500 | 0,848 | +0,083 | 0,625 |
| 20 | 6000 | 0,835 | +0,070 | 0,609 |
| 20 | 3000 | 0,820 | +0,055 | 0,607 |
| 20 | 1500 | 0,807 | +0,043 | 0,600 |
| 40 | 6000 | 0,722 | −0,043 | 0,587 |
| 40 | 3000 | 0,700 | −0,065 | 0,545 |
| 40 | 1500 | 0,687 | −0,077 | 0,578 |

En régime HAUT, la volatilité de l'heure suivante est exp(0,895) ≈ 2,4 fois
celle du régime BAS (horloge seule : ≈ 2,1).

## Validation glissante

| Test | Retenu sur le train | S train | S test | Rang test (/18) | Regret | Horloge test |
|---|---|---|---|---|---|---|
| 2024-03..04 | 7 / 6000 | 0,914 | 0,874 | 1 | 0,000 | 0,763 |
| 2024-05..06 | 7 / 6000 | 0,866 | 0,921 | 2 | 0,009 | 0,798 |
| 2024-07..08 | 7 / 6000 | 0,888 | 0,879 | 1 | 0,000 | 0,740 |
| 2024-09..10 | 7 / 6000 | 0,878 | 0,884 | 1 | 0,000 | 0,791 |
| 2024-11..12 | 7 / 6000 | 0,882 | 0,881 | 4 | 0,011 | 0,729 |

## Lecture

1. **Choix stable.** Le même candidat est retenu dans les cinq plis et reste
   dans les quatre meilleurs (sur 18) sur chaque test. Il bat la référence horloge sur
   chaque bloc (de +0,08 à +0,15).
2. **Optimum en `n_atr` autour de 7.** Le score croît de 40 à 7 puis
   décroît légèrement (5 : −0,002 ; 3 : −0,015). Entre 5 et 7 l'écart est
   négligeable ; un ATR plus court suit mieux la volatilité de l'heure
   suivante jusqu'à ce que le bruit d'un ATR de quelques barres l'emporte.
   `fenetre` compte peu (≤ 0,03 à `n_atr` égal).
3. **Grille étendue après lecture.** 3 et 5 ont été ajoutés parce que
   l'optimum était au bord (voir le journal du protocole) ; l'optimum est
   maintenant intérieur et le choix inchangé.
4. **Effet de séance.** Une grande part de S vient de l'heure de la journée
   (horloge seule : 0,765). C'est conforme à l'usage visé ; l'effet de séance
   est traité dans l'étude de criblage.

## Limites

- Critère changé et grille étendue après lecture des résultats ; validation
  glissante vue plusieurs fois : pas de confirmation indépendante.
- Aucun test hors échantillon (décision du 2026-10-12).
- Heures de la référence horloge fixées d'après les données observées.
- `q_low`, `q_high` et `n_direction` non calibrés ; un seul horizon (1 h).
