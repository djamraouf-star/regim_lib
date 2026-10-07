# Volume — méthodes basées sur le volume

**Dernière mise à jour** : 2026-10-07
**Famille** : `methods/volume/`
**Statut** : `ofi`, `divergence_pv`, `vpin` et `volume_profile` implémentées

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
- **VPIN** : fondement théorique faible sur forex décentralisé.
- **Volume profile** : moins fiable que sur futures.

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

## 4. Autres méthodes de volume

### 4.1 `vpin` — Volume-Synchronized Probability of Informed Trading

Le détecteur utilise `bid_volume` et `ask_volume` comme **proxy exploratoire**
de déséquilibre des cotations. Ces données ne représentent pas des volumes
exécutés ; ce score ne certifie pas une probabilité de flux informé.

Chaque ligne est fractionnée proportionnellement entre des buckets de volume
`bucket_volume`. Le ratio de déséquilibre absolu de la ligne est supposé
uniforme sur ses fractions. Le score est la moyenne des `n_buckets` derniers
buckets **clos à la clôture de la ligne**. Le bucket incomplet reste en attente,
et aucune valeur n'est réattribuée aux lignes antérieures. Le déséquilibre
reste la somme des déséquilibres absolus fractionnés, pas la valeur absolue
d'un solde signé : le fractionnement ne transforme pas le proxy en VPIN canonique.

Exemple : une ligne de volume 250 et un bucket de 100 ferment deux buckets et
laissent un reliquat de 50. Il n'y a aucun bucket vide. La chauffe se termine
dès que `n_buckets` buckets sont clos ; une seule ligne peut suffire. Une ligne
de volume nul conserve le dernier score disponible (ou `INCONNU` pendant la
chauffe). Les barres partielles restent masquées dans les sorties.

La mémoire des buckets est bornée à `n_buckets`, indépendamment du volume
cumulé. Les buckets entiers identiques d'une très grosse ligne sont traités en
bloc. Les divisions proches d'une frontière sont arrondies avec une tolérance
de huit epsilons machine en unités de bucket.

**Migration :** régénérer les sorties VPIN pour obtenir les buckets fractionnés.
Les scores peuvent changer par rapport à l'ancien décalage d'un bucket,
notamment aux frontières et pour les gros volumes. Les anciens parquets sans
métadonnées restent exclus des études causales.

### 4.2 `volume_profile` — Zones de valeur

Sur une fenêtre glissante, histogramme du volume par niveau de prix. La
**zone de valeur** est l'intervalle [P20, P80] (80 % du volume).

| Régime | Condition |
|---|---|
| `DANS_ZONE_VALEUR` | `close` dans [P20, P80] |
| `HORS_ZONE_HAUTE` | `close > P80` |
| `HORS_ZONE_BASSE` | `close < P20` |

La méthode est implémentée dans `methods/volume/volume_profile.py`. Elle
approxime la répartition du volume d'une barre uniformément sur les bins de
prix qu'elle traverse ; ce n'est pas un profil basé sur des transactions
individuelles.

## 5. Causalité

Les méthodes `ofi`, `divergence_pv` et `volume_profile` sont **causales par
construction** : elles utilisent uniquement les données présentes ou passées.
VPIN publie les buckets clos à la clôture de la ligne courante, sans révision
des lignes précédentes.

Les tests de non-fuite par troncature couvrent les méthodes correspondantes.

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
  fenetre_extreme: 20
  fenetre_median_vol: 50
  ratio_volume_fort: 1.5

vpin:
  bucket_volume: 10000.0
  n_buckets: 50
  seuil_bas: 0.20
  seuil_haut: 0.40

volume_profile:
  fenetre: 100
  n_bins: 40
  pct_bas: 0.20
  pct_haut: 0.80
