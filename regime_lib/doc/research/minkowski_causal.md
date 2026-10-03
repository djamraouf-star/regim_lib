# Minkowski causal — détection de régimes par intervalle d'espace-temps

> **Note de recherche / méthode** — cette page documente une hypothèse de
> modélisation propre à `minkowski_causal`, et non un résultat empirique
> validé universellement.

**Dernière mise à jour** : 2026-10-01
**Méthode** : `minkowski_causal`
**Famille** : `methods/misc/`
**Lookahead** : non (causal)

---

## 1. Principe

Le couple (temps, prix) est traité comme un espace-temps 2D muni de la
métrique de Minkowski :

    Δs² = c_mkt² · Δt² − Δx²

Avec Δt = 1 barre (normalisation) et Δx = log-return sur la barre :

    Δs² = c_mkt² − Δx²

`c_mkt` est une **vitesse limite du marché**, calibrée par quantile
glissant sur les valeurs absolues des log-returns passés. Elle joue le
rôle de la vitesse de la lumière : au-delà, le mouvement est considéré
comme non causal.

## 2. Genres d'intervalle

| Genre | Condition | Interprétation |
|---|---|---|
| **TIMELIKE** | Δs² > 0 | Mouvement causal ordinaire (β < 1) |
| **SPACELIKE** | Δs² < 0 | Choc, saut de prix, non causal (β > 1) |
| **LIGHTLIKE** | \|Δs²\| ≈ 0 | Frontière (β ≈ 1) |
| **INCONNU** | chauffe / is_partial | — |

Où **β** = \|Δx\| / c_mkt est la vitesse relative.

## 3. Paramètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `fenetre_c` | 500 | Fenêtre glissante pour `c_mkt` |
| `min_periods_c` | `fenetre_c // 2` | Barres minimales avant calcul |
| `q_c` | 0.90 | Quantile de `c_mkt` |
| `seuil_lightlike` | 0.01 | Tolérance LIGHTLIKE (\|Δs²\| < seuil · c²) |

Tous configurables via `config/profiles/default.yaml` :

```yaml
minkowski_causal:
  fenetre_c: 500
  min_periods_c: 250
  q_c: 0.90
  seuil_lightlike: 0.01
