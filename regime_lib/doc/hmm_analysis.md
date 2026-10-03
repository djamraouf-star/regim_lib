# Analyse HMM gaussien — EURUSD

**Dernière mise à jour** : 2026-09-28
**Source des données** : EURUSD M1, 2025-06-25 → 2026-09-04 (384 212 barres)
**Méthode** : `hmm_gaussian` (`regime_lib/methods/statistical/hmm_gaussian.py`)

---

## 1. Configuration retenue

| Paramètre | Valeur | Justification |
|---|---|---|
| Modèle | GaussianHMM (hmmlearn) | Référence standard |
| Features | `log_return`, `vol_realisee`, `range_norm`, `close_position` | Mix direction + amplitude + position |
| `n_states` | **2** | Voir section 2 |
| `n_iter` | 100 | Convergence EM |
| `seed` | 42 | Reproductibilité |
| `covariance_type` | `diag` | Évite l'overfit |

**Normalisation** : z-score sur toute la série (moyenne et écart-type globaux).

**Lookahead** : `requires_lookahead = True`. La méthode est **offline** :
- la normalisation utilise les statistiques de toute la série ;
- le fit HMM apprend sur toute la série puis prédit sur la même série.

Voir section 5 pour les conséquences.

---

## 2. Choix de `n_states`

### Tests empiriques sur H1 (14 mois, 6 566 barres)

| `n_states` | Ratio vol | Durée médiane | États bruit | Structure détectée |
|---|---|---|---|---|
| **2** | **1.95** | **25 barres** | **0/2** | **Volatilité pure** |
| 3 | 1.96 | 2 barres | 3/3 | Direction dominante |
| 4 | 2.15 | 2.5 barres | 2/4 | Direction + vol |

### Diagnostic

Avec `n_states=3` ou `4`, le HMM gaspille des états sur la **direction court terme** plutôt que d'affiner la structure de volatilité :

- **Ratio de volatilité stagne** entre 2 et 3 états (1.95 → 1.96) : les états supplémentaires n'apportent aucune information de volatilité.
- **Durée médiane s'écroule** (25 barres → 2 barres) : les états ajoutés oscillent à très haute fréquence.
- **`close_position` opposé** entre deux états (ex. 0.19 vs 0.80 sur M1 n=3) : signature d'états directionnels, pas vol.
- **Span de retour augmente** : les états supplémentaires capturent la direction.

### Décision

**`n_states = 2`** est le défaut validé.

Les features de direction (`log_return`, `close_position`) ont une variance par barre beaucoup plus élevée que les features de volatilité (qui sont déjà des moyennes glissantes). Dès que 3 états sont disponibles, l'EM en consacre 2 à la direction et 1 seul à la volatilité.

---

## 3. Caractérisation des états

### 3.1 Statistiques descriptives (n_states=2)

| Timeframe | État | Part | `vol_moy` | `ret_moy` | `close_pos` | Durée médiane |
|---|---|---|---|---|---|---|
| M1 | STATE_0 | 76 % | 0.000066 | ~0 | 0.497 | 40 barres |
| M1 | STATE_1 | 24 % | 0.000164 | ~0 | 0.500 | 5 barres |
| M5 | STATE_0 | 72 % | 0.000146 | ~0 | 0.500 | 48 barres |
| M5 | STATE_1 | 28 % | 0.000357 | ~0 | 0.501 | 21 barres |
| M15 | STATE_0 | 68 % | 0.000255 | ~0 | 0.499 | 43 barres |
| M15 | STATE_1 | 32 % | 0.000584 | ~0 | 0.505 | 22 barres |
| H1 | STATE_0 | 70 % | 0.000573 | ~0 | 0.501 | 30 barres |
| H1 | STATE_1 | 30 % | 0.001117 | ~0 | 0.493 | 20 barres |

**Constantes sur les 4 timeframes :**

| Propriété | STATE_0 | STATE_1 |
|---|---|---|
| Ratio de volatilité | référence | **~2× plus élevé** |
| Retour moyen | ~0 | ~0 |
| `close_position` | ~0.5 | ~0.5 |
| Persistance | 30–48 barres | 5–22 barres |
| Part du temps | 68–76 % | 24–32 % |

**Caractérisation** :
- `STATE_0` = régime **CALME et persistant**
- `STATE_1` = régime **AGITÉ et plus nerveux**

Aucune dimension directionnelle (retour et `close_position` à 0.5).

### 3.2 Les états sont des régimes de **session**

Distribution horaire de `STATE_1` (AGITE) :

| Session | Heures UTC | M1 | M5 | M15 |
|---|---|---|---|---|
| **Asia** | 00–07 | 8.8 % | 9.7 % | 13.5 % |
| **London** | 07–13 | 31.4 % | 35.1 % | 31.0 % |
| **NY** | 13–21 | 35.4 % | 43.7 % | **54.8 %** |
| After | 21–24 | 12.4 % | 12.5 % | 18.7 % |

Pics et creux horaires :

| Timeframe | Pic AGITE | Creux AGITE |
|---|---|---|
| M1 | 14h UTC (**68.4 %**) | 04h UTC (**2.2 %**) |
| M5 | 15h UTC (**73.1 %**) | 04h UTC (**2.7 %**) |
| M15 | 16h UTC (**67.1 %**) | 05h UTC (9.4 %) |

**Conclusion** : `STATE_1` est concentré sur l'overlap London/NY, `STATE_0` domine la session asiatique. Le HMM a découvert la **signature micro-structurelle du forex** — la volatilité explose à l'ouverture européenne puis américaine, s'effondre pendant les heures creuses asiatiques.

### 3.3 Les états capturent les **pics de volatilité**

Définition : top-1% des barres par `|log_return|` (quantile 99%).

| Mesure | M1 | M5 | M15 |
|---|---|---|---|
| Seuil top-1% | 0.000353 | 0.000792 | 0.001392 |
| Nombre de top-1% | 3 843 | 774 | 259 |
| **Précision** (top-1% ∩ AGITE / top-1%) | **100 %** | **100 %** | **100 %** |
| **Rappel** (top-1% ∩ AGITE / AGITE) | 4.2 % | 3.6 % | 3.1 % |

**Précision 100 %** : aucune barre extrême n'est classée CALME. Le HMM capture **systématiquement** les événements violents.

**Rappel 3–4 %** : `STATE_1` contient 25× plus de barres que les top-1%. C'est attendu — c'est un **régime d'ambiance**, pas un détecteur de pics.

### 3.4 Concentration des pics

Cumul des top-1% sur les 5 heures les plus chargées (07h + 12–15h UTC) :

| Timeframe | Cumul |
|---|---|
| M1 | **54.8 %** |
| M5 | **53.7 %** |
| M15 | **54.1 %** |

**5 heures sur 24 concentrent 54 % des pics de volatilité.** Sur-représentation de 2.5–3× par rapport à une distribution uniforme (16.7 % pour 4 heures).

### 3.5 Prédiction d'amplitude future

Ratio des écart-types forward (AGITE / CALME) :

| Horizon | M1 | M5 | M15 |
|---|---|---|---|
| 1 barre | 2.17 | 2.06 | 1.76 |
| 5 barres | 2.00 | 1.88 | 1.57 |
| 20 barres | 1.81 | 1.58 | 1.18 |

**Le régime AGITE prédit une amplitude future ~2× plus grande** à horizon 1 barre. Le ratio décroît avec l'horizon (information se dissipe), conforme au clustering de volatilité documenté.

**Utilité** : sizing de position, filtrage contextuel (éviter de trader en Asia).

---

## 4. Cas particulier : EURUSD M1 sur 14 mois

Proportions mensuelles de `STATE_1` (AGITE) sur M15 :

| Mois | AGITE |
|---|---|
| 2025-07 | 41.0 % |
| 2025-08 | 53.5 % |
| 2025-09 | 35.7 % |
| 2025-10 | 30.2 % |
| 2025-11 | 19.5 % |
| 2025-12 | 16.6 % |
| 2026-01 | 38.1 % |
| 2026-02 | 27.0 % |
| **2026-03** | **67.5 %** |
| 2026-04 | 37.7 % |
| 2026-05 | 25.6 % |
| 2026-07 | 14.1 % |
| 2026-08 | 11.5 % |

Le HMM capture bien des **régimes macro** distincts : décembre 2025 exceptionnellement calme (16.6 %), mars 2026 très agité (67.5 %). Cohérent avec un événement macro (probablement un choc de politique monétaire ou géopolitique).

---

## 5. Limites

### 5.1 Lookahead structurel

La méthode est **offline** par construction :

- Normalisation z-score sur toute la série (moyenne et écart-type incluent le futur).
- Fit HMM sur toute la série, puis prédiction sur la même série.

**Il n'existe pas de version causale** qui préserve l'algorithme. Un fit incrémental produirait une méthode fondamentalement différente.

**Conséquence** : `requires_lookahead = True`. Le CLI exige `--allow-lookahead-override hmm_gaussian:true`.

### 5.2 Biais directionnel apparent — artefact

Les rendements forward moyens montrent un pattern systématique :

| Timeframe | État | Forward 1 | Forward 5 | Forward 20 |
|---|---|---|---|---|
| M1 | AGITE | +0.065 pip | +0.354 pip | +1.059 pip |
| M1 | CALME | −0.021 pip | −0.112 pip | −0.336 pip |
| M15 | AGITE | +0.610 pip | +2.126 pip | +1.284 pip |
| M15 | CALME | −0.292 pip | −1.020 pip | −0.545 pip |
| M5 | AGITE | +0.300 pip | +0.809 pip | +1.471 pip |
| M5 | CALME | −0.118 pip | −0.323 pip | −0.598 pip |

**Ce n'est PAS tradable.** Le HMM a été fit sur toute la série. Le fait que le forward soit corrélé au régime est **mécaniquement induit par le lookahead** : si le HMM sait qu'une hausse arrive, il peut tagger la barre précédente comme AGITE.

Une version causale donnerait probablement des rendements forward proches de zéro.

**Conclusion** : le seul résultat **robuste** est la prédiction d'**amplitude** (cohérente avec la micro-structure, indépendante de la direction). Le « biais directionnel » est à écarter.

### 5.3 Les bascules ne sont pas tradables

Rendement moyen 5 barres après bascule :

| Timeframe | Transition | Moy (pip) | Std (pip) | Ratio |
|---|---|---|---|---|
| M1 | CALME → AGITE | −1.31 | 59.09 | 0.02 |
| M1 | AGITE → CALME | +0.51 | 14.91 | 0.03 |
| M15 | CALME → AGITE | +1.32 | 209.40 | 0.006 |
| M15 | AGITE → CALME | −6.09 | 52.76 | 0.12 |
| M5 | CALME → AGITE | +1.78 | 125.20 | 0.01 |
| M5 | AGITE → CALME | +0.97 | 31.95 | 0.03 |

Signal/bruit < 0.15 partout. Trop peu d'observations (318–713 bascules) et variance trop forte. À ne pas utiliser.

### 5.4 Confidence peu discriminante

La confidence issue de `predict_proba` a une médiane > 0.999 car les états sont bien séparés dans l'espace des features. Elle **discrimine peu** en pratique — à ne pas utiliser comme filtre fin.

---

## 6. Recommandations d'usage

### Usages recommandés

| Usage | Configuration |
|---|---|
| **Filtre de session** (éviter de trader en Asia) | Utiliser `STATE_0` vs `STATE_1` |
| **Sizing de position** | Moduler la taille selon le régime |
| **Feature ML offline** | Ajouter `regime` comme variable explicative |
| **Annotation de dataset** | Marquer les périodes macro distinctes |

### Usages déconseillés

| Usage | Raison |
|---|---|
| **Signal directionnel** | Biais apparent non exploitable (section 5.2) |
| **Timing d'entrée précis** | Rappel 3–4 % sur les pics (section 3.3) |
| **Backtest temps réel** | Lookahead structurel (section 5.1) |

### Choix du timeframe

| Timeframe | Usage |
|---|---|
| **M1** | Micro-structure, événements courts. `STATE_1` dure 5 min en médiane. |
| **M5** | Trading intraday. `STATE_1` dure ~1h45. |
| **M15** | Trading discrétionnaire. `STATE_1` dure ~5h30. |
| **H1** | Positionnement. `STATE_1` dure ~20h. |

Pour un usage « régime de marché », préférer H1 ou plus. Pour un usage « micro-structure », M1 apporte une granularité utile.

---

## 7. Reproductibilité

Pour reproduire les résultats :

```bash
python -m regime_lib \
    --url /chemin/vers/EURUSD_M1_202507_202608.parquet \
    --timeframe M1,M5,M15,H1 \
    --methods hmm_gaussian \
    --allow-lookahead-override hmm_gaussian:true \
    --method-params '{"hmm_gaussian": {"n_states": 2, "n_iter": 100, "seed": 42}}' \
    --output results_hmm_ref
