# Ajouter une méthode de détection

Ce guide détaille les étapes pour implémenter et intégrer une nouvelle méthode de détection de régime dans `regime_lib`.

---

## 1. Étapes d'intégration

1. **Créer le module** dans la sous-famille appropriée :
   - `regime_lib/methods/volatility/` pour les approches basées sur l'amplitude ;
   - `regime_lib/methods/trend/` pour les indicateurs directionnels ou de tendance ;
   - `regime_lib/methods/price/` pour les approches purement graphiques / price action ;
   - `regime_lib/methods/statistical/` pour les modèles statistiques ou d'apprentissage non supervisé.
2. **Définir la classe** héritant de `regime_lib.core.base.RegimeDetector`.
3. **Déclarer les attributs de classe obligatoires** :
   - `name: ClassVar[str]` : nom court snake_case utilisé par le CLI et dans la colonne `method` ;
   - `REGIME_MAP: ClassVar[dict[str, int]]` : dictionnaire stable et bijectif associant chaque libellé de régime à un identifiant entier (ex. `{"CALME": 0, "AGITE": 1, "INCONNU": 2}`) ;
   - `requires_lookahead: ClassVar[bool]` : `True` si la méthode a besoin structurellement du futur, sinon `False`.
4. **Décorer la classe** avec `@register_method` (importé de `regime_lib.core.registry`).
5. **Implémenter `__init__`** :
   - Transmettre `allow_lookahead`, `timeframe` et les hyperparamètres à `super().__init__(...)` pour qu'ils soient inclus dans le calcul de `params_hash`.
6. **Implémenter `fit_predict(df) -> pd.DataFrame`** :
   - Entrée : DataFrame OHLCV avec index `DatetimeIndex` nommé `timestamp` croissant sans doublons.
   - Sortie : DataFrame avec colonnes `["regime", "confidence"]`. Le formatage uniforme complet (colonnes `timeframe`, `asset`, `params_hash`, etc.) est assuré par `to_uniform` dans le pipeline.
7. **Exposer la méthode** dans le `__init__.py` de la sous-famille et dans `regime_lib/methods/__init__.py`.
8. **Ajouter la suite de tests unitaires** dans `tests/test_<nom>.py`.

---

## 2. Squelette minimal de code

```python
from __future__ import annotations

from typing import ClassVar
import pandas as pd
import numpy as np

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method


@register_method
class MaNouvelleMethode(RegimeDetector):
    """Description claire de la méthode et de sa sémantique."""

    name: ClassVar[str] = "ma_methode"
    REGIME_MAP: ClassVar[dict[str, int]] = {
        "CALME": 0,
        "AGITE": 1,
        "INCONNU": 2,
    }
    requires_lookahead: ClassVar[bool] = False

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        seuil: float = 0.5,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            seuil=seuil,
        )
        self.seuil = float(seuil)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        n = len(out)
        if n == 0:
            out["regime"] = pd.Series([], dtype=object)
            out["confidence"] = pd.Series([], dtype=float)
            return out[["regime", "confidence"]]

        # 1) Calcul causal des indicateurs
        # (ex. seuils glissants sur valeur.shift(1))
        
        # 2) Classification
        regimes = np.full(n, "CALME", dtype=object)
        confidence = np.full(n, 1.0, dtype=float)

        # 3) Gestion des barres non clôturées / hors grille
        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
            regimes[partial] = "INCONNU"
            confidence[partial] = 0.0

        out["regime"] = regimes
        out["confidence"] = confidence
        return out[["regime", "confidence"]]
```

---

## 3. Bonnes pratiques de conception

- **Anti-lookahead** : pour les méthodes causales, tout calcul de seuil ou quantile glissant doit être décalé d'une barre (`.shift(1)`) pour ne pas intégrer la barre en cours d'évaluation.
- **Résolution temporelle** : ne pas supposer que le `DatetimeIndex` est en nanosecondes. Utiliser des soustractions de `pd.Timedelta` pour tester l'alignement sur la grille.
- **Nomenclature** : préférer les labels standards (`TENDANCE_HAUSSIERE`, `TENDANCE_BAISSIERE`, `RANGE`, `CHOP`, `INCONNU`) lorsque la méthode s'y prête, ou documenter explicitement toute sémantique spécifique (comme les états bruts `STATE_0`, `STATE_1` pour le HMM).
