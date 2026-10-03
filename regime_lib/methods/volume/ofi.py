"""
Détection de régimes par Order Flow Imbalance (OFI).

Définition
----------
Sur une fenêtre glissante :

    OFI(t) = Σ(askVolume - bidVolume) / Σ(askVolume + bidVolume)

sur les barres de la fenêtre. Résultat borné dans [-1, 1] :
- OFI > 0 : pression acheteuse nette
- OFI < 0 : pression vendeuse nette
- OFI ≈ 0 : équilibre

Interprétation sur forex
------------------------
Le volume Dukascopy est une **profondeur observée**, pas un volume
exécuté. L'OFI mesure un déséquilibre côté Dukascopy, pas une pression
mondiale. À interpréter comme un **proxy exploratoire**.

Régimes produits
----------------
- PRESSION_ACHETEUSE : OFI > seuil_pression
- PRESSION_VENDEUSE  : OFI < -seuil_pression
- EQUILIBRE          : entre les deux
- INCONNU            : chauffe ou données manquantes

Mesure exposée
--------------
- `ofi_value` : OFI brut ∈ [-1, 1]

Causalité
---------
Fenêtre strictement passée. Aucun lookahead.

Données requises
----------------
Colonnes : `bid_volume`, `ask_volume` (issues de l'agrégation tick).
Disponible uniquement sur données tick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method


_TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800,
}


@register_method
class OrderFlowImbalanceDetector(RegimeDetector):
    """
    Détecteur de régimes par Order Flow Imbalance.

    Paramètres
    ----------
    fenetre : int
        Fenêtre glissante pour le calcul de l'OFI.
    seuil_pression : float
        Seuil absolu de l'OFI ∈ (0, 1) pour classer PRESSION_*.
        Défaut : 0.1 (10 % de déséquilibre net).
    """

    name = "ofi"
    REGIME_MAP = {
        "PRESSION_ACHETEUSE": 0,
        "PRESSION_VENDEUSE": 1,
        "EQUILIBRE": 2,
        "INCONNU": 3,
    }
    requires_lookahead = False
    REQUIRES_COLUMNS = ["bid_volume", "ask_volume"]

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        fenetre: int | None = None,
        seuil_pression: float | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            fenetre=fenetre,
            seuil_pression=seuil_pression,
        )
        self.fenetre = int(self._resolve(
            "fenetre", fenetre, default=20, per_timeframe=True,
        ))
        self.seuil_pression = float(self._resolve(
            "seuil_pression", seuil_pression, default=0.1
        ))

        if self.fenetre < 2:
            raise ValueError("fenetre doit être >= 2.")
        if not (0.0 < self.seuil_pression < 1.0):
            raise ValueError("seuil_pression doit être dans (0, 1).")

    @staticmethod
    def _unaligned_mask(
        index: pd.DatetimeIndex, timeframe: str | None
    ) -> np.ndarray:
        step = _TIMEFRAME_SECONDS.get(timeframe or "")
        if step is None or len(index) == 0:
            return np.zeros(len(index), dtype=bool)
        if index.tz is not None:
            epoch = pd.Timestamp("1970-01-01", tz="UTC")
        else:
            epoch = pd.Timestamp("1970-01-01")
        ts_s = np.asarray(
            (index - epoch) // pd.Timedelta(seconds=1), dtype=np.int64
        )
        return (ts_s % step) != 0

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [c.lower() if isinstance(c, str) else c
                       for c in out.columns]
        self.verifier_colonnes(out)

        n = len(out)
        if n == 0:
            out["regime"] = pd.Series([], dtype=object)
            out["confidence"] = pd.Series([], dtype=float)
            out["ofi_value"] = pd.Series([], dtype=float)
            return out[["regime", "confidence", "ofi_value"]]

        bid_vol = out["bid_volume"].to_numpy(dtype=float)
        ask_vol = out["ask_volume"].to_numpy(dtype=float)

        # OFI brut par barre
        total = bid_vol + ask_vol
        with np.errstate(divide="ignore", invalid="ignore"):
            ofi_bar = np.where(
                total > 0, (ask_vol - bid_vol) / total, np.nan
            )

        # Moyenne glissante
        ofi_series = pd.Series(ofi_bar, index=out.index)
        ofi = ofi_series.rolling(
            self.fenetre, min_periods=max(1, self.fenetre // 2)
        ).mean()

        # Copie explicite : `.to_numpy()` peut retourner une vue
        # read-only en pandas 2.x (backend Arrow).
        ofi_v = np.array(ofi.to_numpy(dtype=float), dtype=float, copy=True)

        # Classification
        ready = ~np.isnan(ofi_v)
        mask_achat = ready & (ofi_v > self.seuil_pression)
        mask_vente = ready & (ofi_v < -self.seuil_pression)
        mask_equilibre = ready & ~mask_achat & ~mask_vente

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_achat] = "PRESSION_ACHETEUSE"
        regime[mask_vente] = "PRESSION_VENDEUSE"
        regime[mask_equilibre] = "EQUILIBRE"

        # Confidence
        confidence = np.full(n, 0.0, dtype=float)
        if mask_achat.any():
            conf = (ofi_v[mask_achat] - self.seuil_pression) / (
                1.0 - self.seuil_pression
            )
            confidence[mask_achat] = 0.5 + 0.5 * np.clip(conf, 0.0, 1.0)
        if mask_vente.any():
            conf = (-ofi_v[mask_vente] - self.seuil_pression) / (
                1.0 - self.seuil_pression
            )
            confidence[mask_vente] = 0.5 + 0.5 * np.clip(conf, 0.0, 1.0)
        if mask_equilibre.any():
            confidence[mask_equilibre] = 0.5

        # Bougies non clôturées
        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
        else:
            partial = self._unaligned_mask(out.index, self.timeframe)

        if partial.any():
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
            ofi_v[partial] = np.nan

        out["regime"] = regime
        out["confidence"] = confidence
        out["ofi_value"] = ofi_v
        return out[["regime", "confidence", "ofi_value"]]
