# Shannon — entropie des rendements

**Dernière mise à jour** : 2026-10-02
**Méthode** : `shannon`
**Famille** : `methods/entropy/`
**Lookahead** : non (causal)

---

## 1. Principe

Sur une fenêtre glissante de rendements, on discrétise leur distribution
en N bins et on calcule l'entropie de Shannon :

    H = -Σ p_i · log(p_i)

H est normalisée par log(N) pour obtenir une mesure bornée :

    H_norm = H / log(N) ∈ [0, 1]

## 2. Interprétation

- H_norm → 0 : distribution concentrée → marché **ordonné**, structuré
- H_norm → 1 : distribution uniforme → marché **chaotique**, imprévisible

**Note importante** : l'entropie normalisée mesure la **forme** de la
distribution, pas son **échelle**. Deux gaussiennes de variances
différentes ont des entropies normalisées proches (~0.86 vs ~0.89).
Pour distinguer les régimes de volatilité, il faut coupler cette mesure
à une mesure d'amplitude (ATR, vol réalisée).

## 3. Régimes produits

| Régime | Condition |
|---|---|
| ORDONNE | H_norm < `seuil_ordonne` |
| NEUTRE | entre les deux seuils |
| DESORDRE | H_norm > `seuil_desordre` |
| INCONNU | phase de chauffe ou barre non clôturée |

## 4. Paramètres

| Paramètre | Défaut (H1) | Rôle |
|---|---|---|
| `fenetre` | 100 | Taille de la fenêtre glissante |
| `n_bins` | 10 | Nombre de bins de discrétisation |
| `seuil_ordonne` | 0.65 | Seuil bas |
| `seuil_desordre` | 0.78 | Seuil haut |

Configurables via `default.yaml` :

```yaml
shannon:
  timeframes:
    M1:  {fenetre: 200, n_bins: 10}
    M5:  {fenetre: 200, n_bins: 10}
    M15: {fenetre: 150, n_bins: 10}
    H1:  {fenetre: 100, n_bins: 10}
    H4:  {fenetre: 50,  n_bins: 10}
    D1:  {fenetre: 30,  n_bins: 10}
  seuil_ordonne: 0.65
  seuil_desordre: 0.78
