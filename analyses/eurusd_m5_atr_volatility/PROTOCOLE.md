# Calibration `atr_volatility` — EURUSD M5

Protocole figé le 2026-10-12, **avant tout calcul de résultat**. Toute
modification ultérieure est datée et justifiée ci-dessous ; une période déjà
évaluée ne redevient pas indépendante.

## Données

- Source : `actifs/EURUSD/EURUSD_ticks-20231120-20261002.parquet`
  (sha256 `ceecc25c…0936420`), Bid.
- Barres : `load_parquet(..., combler_sans_tick="60min")` puis
  `resample(..., "M5")`.
- Lecture limitée à `[2023-11-20, 2025-01-10)` :
  - warmup : 2023-11-20 → 2023-12-31 ;
  - évaluation : barres datées du 2024-01-01 au 2024-12-31 ;
  - lookahead : début janvier 2025, uniquement pour les cibles.
- Rien après 2025-01-10 n'est lu.

## Ce qui est calibré

| Paramètre | Candidats | Statut |
|---|---|---|
| `n_atr` | 3, 5, 7, 14, 20, 40 (3 et 5 ajoutés le 2026-10-12) | calibré |
| `fenetre` | 1500, 3000, 6000 (≈ 5, 10, 21 jours de cotation) | calibré |
| `min_periods` | `fenetre // 2` | fixé |
| `q_low`, `q_high` | 0.33, 0.67 | fixés : ils fixent les proportions de classes, un critère de séparation les pousserait toujours vers les extrêmes |
| `n_direction` | 20 | fixé : ne sert qu'au partage haussier/baissier, hors critère de volatilité |

Soit 18 candidats (12 avant l'extension). Avec `fenetre = 6000`, la première barre de 2024 doit
disposer d'une fenêtre complète ; le script le vérifie et s'arrête sinon.

## Cible

`vol_fwd_12` de `calculer_cible` : écart-type des log-rendements des 12
barres M5 suivantes (1 h). Une cible n'est retenue que si les barres
t+1…t+12 sont contiguës (`index[t+12] − index[t] = 60 min`), ce qui exclut
les week-ends et jours fériés.

## Critère (principal, unique) — révisé le 2026-10-12

L'indicateur doit fournir un **niveau de volatilité global** ; l'effet de
séance est étudié ensuite, dans l'étude de criblage. Le critère n'est donc
pas stratifié :

- support : barres valides (`validity_mask`) avec cible retenue ;
- HAUT = `TENDANCE_HAUSSIERE` ∪ `TENDANCE_BAISSIERE`, BAS = `RANGE` ;
- **S = log médiane(cible | HAUT) − log médiane(cible | BAS)**.

Référence publiée avec S : un détecteur « horloge » sans prix (HAUT de 3 h à
11 h, BAS de 16 h à 1 h, heure de New York du début de barre, CHOP sinon),
évalué sur les barres non partielles avec cible retenue. L'excès
`S − S_horloge` est rapporté ; il ne sert pas au choix.

Diagnostic rapporté sans servir au choix : S stratifié par heure de New York
(moyenne des écarts par heure pondérée par `min(n_HAUT, n_BAS)`, strates
avec `min ≥ 30`), couverture, effectifs.

## Validation glissante (2024 uniquement)

2024 est découpé en six blocs de deux mois B1…B6. Pour f = 2…6 :

- train = B1…B(f−1) (expansif), test = Bf ;
- purge : une barre de train n'est retenue que si sa cible se termine avant
  le début du test ;
- le candidat de S maximal sur le train est sélectionné, puis évalué sur le
  test.

Rapporté par pli : candidat retenu, S train, S test, rang du candidat retenu
sur le test et regret (meilleur S test − S test du retenu).

## Règle d'adoption

- Paramètre final : le candidat de S maximal sur toute l'année 2024.
- La validation glissante sert de diagnostic de stabilité : elle est publiée
  avec le choix et ne le modifie pas.
- Le choix est écrit dans `eurusd.yaml` sous
  `methodes.atr_volatility.timeframes.M5`, avec un renvoi vers ce
  protocole.
- Limite déclarée : aucun test hors échantillon indépendant n'est réalisé
  à ce stade (décision du 2026-10-12).

## Journal des modifications

- 2026-10-12 : version initiale.
- 2026-10-12, précisions d'implémentation fixées dans le script avant
  l'exécution, consignées ici après lecture des résultats : les ticks de la fenêtre de lecture sont
  filtrés par pyarrow puis passés à `agreger_tick` (même chemin que
  `load_parquet`) ; l'heure de strate est celle du **début** de la barre
  M5.
- 2026-10-12, **changement de critère après lecture des résultats.** Le
  critère initial (S stratifié par heure) mesurait l'information au-delà de
  l'heure de la journée. L'utilisateur précise l'usage visé : un niveau de
  volatilité global pour la calibration, l'effet de séance étant traité
  dans l'étude de criblage. Le critère principal devient S non stratifié ;
  le stratifié reste en diagnostic. Avec le critère initial, le choix était
  `n_atr = 14`, `fenetre = 6000`. Les heures de la référence horloge ont été
  fixées d'après la répartition HAUT/BAS observée : la référence est donc
  légèrement optimiste. La validation glissante 2024 a été vue avec les deux
  critères et ne constitue pas une confirmation indépendante.
- 2026-10-12, **extension de la grille après lecture des résultats.** Avec le
  critère global, l'optimum (`n_atr = 7`) était au bord de la grille.
  `n_atr = 3` et `5` sont ajoutés ; critère, cible et règle d'adoption
  inchangés. Cette extension est décidée en connaissant la tendance
  monotone observée.
- 2026-10-13, **changement du fichier source après perte du dossier local.**
  Le fichier `EURUSD_ticks-20231120-20261002.parquet` (sha256 `ceecc25c…0936420`)
  n'est plus disponible. Il est remplacé par
  `actifs/EURUSD/EURUSD_Tick_20231120_20261010.parquet` (sha256
  `51ef04fa…9674df7`), fin plus tardive. `calibrer.py` est mis à jour
  (`SOURCE`, `SHA256`) ; grille, critère, cible et règle d'adoption sont
  inchangés. Les calibrations ATR, ER et ADX ont été relancées : paramètres
  retenus et scores identiques à ceux de `RESULTATS.md` (par exemple
  S = 0,895 pour `n_atr = 7`, `fenetre = 6000`). Les fichiers de `output/`
  sont régénérés avec le nouveau hash. Le fichier d'origine n'a pas été
  comparé octet à octet.
