# Volume — méthodes basées sur le volume

**Dernière mise à jour** : 2026-10-03
**Famille** : `methods/volume/`
**Statut** : `ofi` et `divergence_pv` implémentées

---

## 1. Principe

Les méthodes de cette famille exploitent les colonnes de volume issues
de l'agrégation tick :

| Colonne | Origine | Sens |
|---|---|---|
| `volume` | bidVolume + askVolume | Activité totale observée |
| `bid_volume` | somme des bidVolume | Profondeur côté acheteur |
| `ask_volume` | somme des askVolume | Profondeur côté vendeur |
| `tick_count` | nombre de ticks | Participation (proxy du volume) |
| `spread_mean` | moyenne du spread | Coût de transaction observé |

Ces colonnes sont produites par `data_loader.agreger_tick` et propagées
lors du resample.

## 2. Fiabilité des données Dukascopy

⚠ **Le volume Dukascopy n'est pas un volume exécuté.** C'est une
**profondeur observée** : la liquidité disponible chez Dukascopy à
l'instant t, pas un montant notionnel échangé.

Trois implications.

### 2.1 Forex décentralisé

Le marché forex n'a pas de volume centralisé. Chaque broker a sa propre
vision. Le volume Dukascopy ne représente qu'une **fraction** de
l'activité mondiale.

| Marché | Volume | Fiabilité |
|---|---|---|
| Futures (ES, NQ) | Centralisé | Élevée |
| Actions US | Centralisé | Élevée |
| Forex spot | Fragmentaire | Faible |
| XAUUSD spot | Fragmentaire | Faible |

### 2.2 Conséquences sur les méthodes

- **OFI** : interprétable comme proxy de pression côté Dukascopy.
- **Divergence prix/volume** : populaire mais peu validée statistiquement.
- **VPIN** (à venir) : fondement théorique faible sur forex décentralisé.
- **Volume profile** (à venir) : moins fiable que sur futures.

### 2.3 Alternative robuste : `tick_count`

Le `tick_count` est **fiable indépendamment du broker**. Il compte les
mises à jour de prix, mesure directe de la **participation**. C'est un
meilleur proxy du volume réel que les colonnes `bidVolume`/`askVolume`.

**Recommandation** : pour les hypothèses sur la participation, préférer
`tick_count`. Pour les hypothèses sur la pression directionnelle,
utiliser `volume` avec les précautions ci-dessus.

## 3. Méthodes implémentées

### 3.1 `ofi` — Order Flow Imbalance

**Définition** : sur une fenêtre glissante,

    OFI(t) = Σ(askVolume − bidVolume) / Σ(askVolume + bidVolume)

Résultat borné dans [-1, 1] :
- OFI > 0 : pression acheteuse nette
- OFI < 0 : pression vendeuse nette
- OFI ≈ 0 : équilibre

**Régimes produits** :

| Régime | Condition |
|---|---|
| `PRESSION_ACHETEUSE` | `ofi_value > seuil_pression` |
| `PRESSION_VENDEUSE` | `ofi_value < −seuil_pression` |
| `EQUILIBRE` | entre les deux |
| `INCONNU` | chauffe ou données manquantes |

**Mesure exposée** : `ofi_value` ∈ [−1, 1].

**Paramètres** :

| Paramètre | Défaut (H1) | Rôle |
|---|---|---|
| `fenetre` | 20 | Fenêtre glissante |
| `seuil_pression` | 0.1 | Seuil de classification |

**Données requises** : `bid_volume`, `ask_volume`. Refus explicite si
absentes.

### 3.2 `divergence_pv` — Divergence prix / volume

**Principe** : sur une fenêtre glissante de N barres,

- Nouveau plus haut + volume fort → `CONFIRMATION_HAUSSIERE`
- Nouveau plus haut + volume faible → `DIVERGENCE_HAUSSIERE`
- Nouveau plus bas + volume fort → `CONFIRMATION_BAISSIERE`
- Nouveau plus bas + volume faible → `DIVERGENCE_BAISSIERE`
- Ni haut ni bas → `NEUTRE`

**Régimes produits** : 5 labels + `INCONNU`.

**Mesures exposées** :
- `volume_rel` : volume / médiane glissante du volume
- `is_new_high` : booléen
- `is_new_low` : booléen

**Paramètres** :

| Paramètre | Défaut (H1) | Rôle |
|---|---|---|
| `fenetre_extreme` | 20 | Fenêtre pour les nouveaux extrêmes |
| `fenetre_median_vol` | 50 | Fenêtre pour la médiane du volume |
| `ratio_volume_fort` | 1.5 | Seuil « volume fort » |

**Données requises** : `close`, `volume`.

**Confidence** : 0.5 (NEUTRE), 0.6 (CONFIRMATION), 0.7 (DIVERGENCE).

## 4. Méthodes à venir

### 4.1 `vpin` — Volume-Synchronized Probability of Informed Trading

Mesure la **toxicité du flux** (Easley, López de Prado, O'Hara 2012).
Nécessite un **échantillonnage par volume** (bucket bars) et non par
temps. Coût de calcul élevé.

**Fondement théorique faible sur forex décentralisé.** À traiter comme
exploratoire.

Effort estimé : 3–4 h.

### 4.2 `volume_profile` — Zones de valeur

Sur une fenêtre glissante, histogramme du volume par niveau de prix. La
**zone de valeur** est l'intervalle [P20, P80] (80 % du volume).

| Régime | Condition |
|---|---|
| `DANS_ZONE_VALEUR` | `close` dans [P20, P80] |
| `HORS_ZONE_HAUTE` | `close > P80` |
| `HORS_ZONE_BASSE` | `close < P20` |

Effort estimé : 2–3 h.

## 5. Causalité

Les deux méthodes implémentées sont **causales par construction** :
fenêtres strictement passées, aucune fuite du futur.

Test de non-fuite validé sur les deux (`test_no_future_leak_by_truncation`).

## 6. Configuration YAML

Section `methodes:` de `default.yaml` :

```yaml
ofi:
  timeframes:
    M1:  {fenetre: 20}
    M5:  {fenetre: 20}
    M15: {fenetre: 20}
    H1:  {fenetre: 20}
    H4:  {fenetre: 20}
  seuil_pression: 0.1

divergence_pv:
  timeframes:
    M1:  {fenetre_extreme: 20, fenetre_median_vol: 50}
    H1:  {fenetre_extreme: 20, fenetre_median_vol: 50}
  ratio_volume_fort: 1.5
