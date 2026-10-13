# Calibration `er_kaufman` — EURUSD M5

Protocole figé le 2026-10-12, **avant tout calcul de résultat**. Toute
modification ultérieure est datée et justifiée dans le journal ; une période
déjà évaluée ne redevient pas indépendante.

## Données

Identiques à la [calibration ATR M5](../eurusd_m5_atr_volatility/PROTOCOLE.md) :

- source : `actifs/EURUSD/EURUSD_ticks-20231120-20261002.parquet`
  (sha256 `ceecc25c…0936420`), Bid ;
- barres M1 avec `combler_sans_tick="60min"`, puis resample en M5 ;
- lecture limitée à `[2023-11-20, 2025-01-10)` : warmup jusqu'au
  2023-12-31, évaluation sur 2024, lookahead début 2025 pour les cibles.

## Ce qui est calibré

| Paramètre | Candidats | Statut |
|---|---|---|
| `n_er` | 5, 10, 20, 40, 80 | calibré (grille large pour éviter un optimum au bord) |
| `fenetre` | 1500, 3000, 6000 | calibré |
| `min_periods` | `fenetre // 2` | fixé |
| `q_chop`, `q_tendance` | 0.40, 0.60 | fixés : ils fixent les proportions de classes |

Soit 15 candidats. Le script vérifie que la première barre de 2024 dispose
d'une fenêtre complète pour le plus grand candidat.

## Cible

Efficience future sur K = 12 barres (1 h) :

`ER_fwd(t) = |close[t+12] − close[t]| / Σ_{i=t+1..t+12} |close[i] − close[i−1]|`

Une cible n'est retenue que si les barres t+1…t+12 sont contiguës (60 min)
et si son dénominateur est non nul.

## Critère (principal, unique)

Comme pour l'ATR, il s'agit d'un critère **global**, sans stratification par
séance ; l'effet de séance est traité dans l'étude de criblage.

- support : barres valides (`validity_mask`) avec cible retenue ;
- TENDANCE = `TENDANCE_HAUSSIERE` ∪ `TENDANCE_BAISSIERE` ;
- **S = log médiane(ER_fwd | TENDANCE) − log médiane(ER_fwd | CHOP)**.

Plus S est grand, plus l'efficience détectée persiste sur l'heure suivante.

Diagnostics rapportés sans servir au choix :

- S stratifié par heure de New York (même définition que pour l'ATR) ;
- **accord directionnel** : part des barres TENDANCE dont le rendement des
  12 barres suivantes a le signe de la direction annoncée, avec un
  intervalle à 95 % ;
- part de chaque classe et couverture.

## Validation glissante (2024 uniquement)

Même découpage que pour l'ATR : six blocs de deux mois, train expansif
B1…B(f−1), test Bf pour f = 2…6, purge des cibles qui chevauchent le test.
Rapporté par pli : candidat retenu, S train, S test, rang sur le test,
regret.

## Règle d'adoption

- Paramètre final : le candidat de S maximal sur toute l'année 2024.
- La validation glissante est un diagnostic de stabilité et ne modifie pas le
  choix.
- Le choix est écrit dans `eurusd.yaml` sous
  `methodes.er_kaufman.timeframes.M5`.
- Si l'optimum tombe au bord de la grille, il est signalé ; toute extension
  est consignée dans le journal comme décision post-hoc.

## Limites déclarées d'avance

- Aucun test hors échantillon indépendant.
- L'horizon K = 12 favorise peut-être les `n_er` proches de 12 ; un seul
  horizon est évalué.
- L'efficience de prix Bid M5 peut être affectée par la discrétisation
  (pas de cotation, barres plates sans tick) aux heures calmes.

## Journal des modifications

- 2026-10-12 : version initiale.
- 2026-10-13 : le fichier source est remplacé par
  `actifs/EURUSD/EURUSD_Tick_20231120_20261010.parquet` (sha256
  `51ef04fa…9674df7`), voir le journal de la
  [calibration ATR M5](../eurusd_m5_atr_volatility/PROTOCOLE.md). Calibration
  relancée : paramètres retenus inchangés (`n_er = 20`, `fenetre = 6000`).
