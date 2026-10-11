# Calibration — préalable à la production

## Périmètre public

La calibration est un préalable à la mise en production de l'API, sous la
responsabilité de l'utilisateur. Aucun moteur de calibration par actif,
par famille ou par indicateur ne fait partie du périmètre public prévu.
Chaque utilisateur choisit sa méthode, ses paramètres et ses critères de
validation en fonction de son usage.

Les valeurs de [default.yaml](../config/profiles/default.yaml) sont des
**paramètres arbitraires, propres à aucun actif**. Elles ne sont ni une
calibration EURUSD ni des paramètres validés pour la production. Les fenêtres
et seuils peuvent être remplacés par des profils utilisateur selon les
[règles de configuration](config.md).

## Exemples éventuels

Des calibrations spécifiques à **EURUSD et XAUUSD**, limitées à certains
indicateurs, peuvent être fournies à titre d'exemples. Elles n'imposent aucune
méthode et ne deviennent pas les paramètres par défaut de l'API.
Le profil EURUSD contient des exemples de surcharges ; le profil XAUUSD
hérite actuellement des défauts sans surcharge. Leur présence ne démontre
pas une validation empirique.

## Scores et validation

Un score `confidence` n'est pas nécessairement une probabilité de justesse du
régime. Les métadonnées `confidence_kind`, `confidence_description` et
`confidence_calibrated=False` précisent son interprétation. Une normalisation
dans [0,1] ne constitue pas une calibration probabiliste.

L'utilisateur distingue l'ajustement des paramètres de l'indicateur et,
s'il en a besoin, l'estimation d'une probabilité pour un événement explicite.
Il définit les données, les périodes et les critères avant l'évaluation,
vérifie la disponibilité temporelle des observations et réserve des données
indépendantes pour la validation. Les tests logiciels ne remplacent pas
cette validation ; la calibration seule ne démontre pas l'aptitude au trading.

## Exemples de calibration possible

Ces exemples montrent comment un utilisateur peut calibrer **en dehors de
l'API**, à partir des sorties publiques des détecteurs et des cibles de
`regime_lib.study`. Ils illustrent des choix possibles ; ils ne sont ni une
méthode recommandée ni une validation. `bars` désigne des barres OHLCV H1
horodatées à la clôture, par exemple issues de `load_parquet`.

### Probabilité d'un événement explicite à partir de `confidence`

L'événement est fixé avant l'évaluation : volatilité réalisée sur cinq barres
supérieure à sa médiane du train. Le train est purgé : chaque cible doit être
close avant le début du test.

```python
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

import regime_lib.methods  # noqa: F401 — enregistre les détecteurs
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.study import calculer_cible
from regime_lib.utils.validation import validity_mask

K = 5
regimes = METHOD_REGISTRY["er_kaufman"](timeframe="H1").fit_predict(bars)
cible = calculer_cible(bars, f"vol_fwd_{K}", timeframe="H1")
fin_cible = bars.index + pd.Timedelta(hours=K)  # clôture de t+K

# Périodes figées avant toute lecture des résultats.
debut_test = pd.Timestamp("2024-10-01", tz="UTC")
support = validity_mask(regimes) & cible.notna()
train = support & (fin_cible < debut_test)          # purge par fin de cible
test = support & (bars.index >= debut_test)

seuil = cible[train].median()                        # défini sur le train seul
evenement = (cible > seuil).astype(int)

modele = IsotonicRegression(increasing=True, out_of_bounds="clip")
modele.fit(regimes.confidence[train], evenement[train])
proba = modele.predict(regimes.confidence[test])

brier = np.mean((proba - evenement[test]) ** 2)
reference = np.mean((evenement[train].mean() - evenement[test]) ** 2)
skill = 1 - brier / reference
```

Points d'attention :

- le sens `increasing` est fixé avant le test, jamais choisi sur ses résultats ;
- un taux d'événement qui dérive entre train et test pénalise la référence
  comme le modèle : publier les deux taux avec le skill ;
- `validity_mask` exclut `INCONNU`, la chauffe et les barres partielles ;
  rapporter la couverture obtenue, en particulier par heure de la journée.

### Choix d'un paramètre de détecteur sur le train seul

Les candidats et le critère sont fixés d'avance ; le paramètre retenu est
ensuite figé dans un profil utilisateur, puis évalué une seule fois sur le test.

```python
ret = calculer_cible(bars, f"ret_fwd_{K}", timeframe="H1")
scores = {}
for n_er in [10, 20, 40]:
    r = METHOD_REGISTRY["er_kaufman"](timeframe="H1", n_er=n_er).fit_predict(bars)
    masque = validity_mask(r) & ret.notna() & (fin_cible < debut_test)
    amplitude, labels = ret[masque].abs(), r.regime[masque]
    # Critère : écart d'amplitude future entre tendance et chop.
    scores[n_er] = (amplitude[labels.str.startswith("TENDANCE")].mean()
                    - amplitude[labels.eq("CHOP")].mean())
n_er_retenu = max(scores, key=scores.get)
```

Le choix est ensuite reporté dans un profil YAML, selon les
[règles de configuration](config.md) :

```yaml
methodes:
  er_kaufman:
    timeframes:
      H1:
        n_er: 20   # valeur retenue sur le train
```

Chaque période de test consultée doit être consignée : une période déjà
regardée ne peut plus servir de confirmation indépendante.

## Historique

Des prototypes de calibration (`ConfidenceCalibrator`, inventaire et cibles
de la famille price) ont été développés avant cette décision de périmètre.
Ils ont été **retirés du paquet le 11 octobre 2026** ; aucune fonction de
calibration n'est exportée par `regime_lib`. La
[note historique price](calibration_price.md) conserve le contexte du
chantier abandonné. Le statut historique C5 de l'audit reste inchangé ; sa
confirmation empirique relève de la préparation à la production. Voir la
[feuille de route](TODO.md).
