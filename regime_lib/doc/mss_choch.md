# MSS / CHOCH — Market Structure Shift et Change of Character

**Dernière mise à jour** : 2026-10-02
**Méthode** : `mss_choch`
**Famille** : `methods/price/`
**Lookahead** : **oui** (structurel)

---

## 1. Concept (style ICT)

La **structure de marché** est définie par les swing highs (sommets) et
swing lows (creux). Une tendance est une séquence cohérente de ces
pivots :

- **Uptrend** : Higher Highs (HH) + Higher Lows (HL)
- **Downtrend** : Lower Highs (LH) + Lower Lows (LL)
- **Neutral** : structure incohérente

Deux événements marquent les **retournements** :

### CHOCH — Change of Character

**Premier signal** de retournement, non confirmé.

- En uptrend : cassure du dernier HL (Higher Low) à la baisse
- En downtrend : cassure du dernier LH (Lower High) à la hausse

### MSS — Market Structure Shift

**Confirmation** du retournement. Après un CHOCH, cassure du swing
majeur opposé.

- En uptrend : cassure du HL précédent (plus ancien) → MSS baissier
- En downtrend : cassure du LH précédent → MSS haussier

## 2. Différence avec `price_action`

| Aspect | `price_action` | `mss_choch` |
|---|---|---|
| Nature | **Régime** (persistant) | **État de cassure** évalué barre par barre |
| Sortie | TENDANCE / RANGE / CHOP | CHOCH / MSS / NEUTRE |
| Utilité | Contexte de marché | Signal de retournement |
| Fréquence | ~30 % RANGE, ~30 % CHOP | ~5 % événements |

Les deux sont **complémentaires** : un CHOCH signale la fin d'une
tendance que `price_action` classifiait comme TENDANCE_*.

## 3. Régimes produits

| Régime | Condition | Confidence |
|---|---|---|
| `MSS_HAUSSIER` | Cassure du LH précédent à la hausse | 0.8 |
| `MSS_BAISSIER` | Cassure du HL précédent à la baisse | 0.8 |
| `CHOCH_HAUSSIER` | Cassure du LH actuel à la hausse | 0.6 |
| `CHOCH_BAISSIER` | Cassure du HL actuel à la baisse | 0.6 |
| `NEUTRE` | Structure stable, pas de cassure | 0.5 |
| `INCONNU` | Chauffe ou barre non clôturée | 0.0 |

La confidence code la **hiérarchie sémantique** : CHOCH (non confirmé)
< MSS (confirmé). Elle n'est pas modulée par l'amplitude de la cassure.
Le label est recalculé à chaque barre et peut persister tant que la clôture
reste au-delà du niveau cassé ; les barres d'un même mouvement ne sont pas
réduites à une seule ligne événementielle.

## 4. Paramètres

| Paramètre | Défaut | Rôle |
|---|---|---|
| `n_fractale` | 5 | Nombre de barres de chaque côté pour définir une fractale |

**Sensibilité** : le paramètre `n_fractale` contrôle la **granularité**
de la détection.

- `n_fractale=3` : plus de pivots, plus d'événements (~10 %), plus de bruit
- `n_fractale=5` : compromis (défaut)
- `n_fractale=10` : moins de pivots, moins d'événements (~2 %), plus fiable

Configurable via `default.yaml` :

```yaml
methodes:
  mss_choch:
    timeframes:
      M1:  {n_fractale: 5}
      M5:  {n_fractale: 5}
      M15: {n_fractale: 5}
      H1:  {n_fractale: 5}
      H4:  {n_fractale: 5}
      D1:  {n_fractale: 5}
```

Le lookahead est obligatoire. Exemple d'activation dans le CLI :

```bash
regime-lib \
  --url data/EURUSD_M1.parquet \
  --timeframe H1 \
  --methods mss_choch \
  --allow-lookahead-override mss_choch:true \
  --output results_mss_choch
```

Les résultats sont donc descriptifs/offline et ne doivent pas être
interprétés comme des signaux disponibles en temps réel à la barre
étiquetée.
