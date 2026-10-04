"""
Détection de régimes par classification causale de Minkowski.

Principe
--------
Le couple (temps, prix) est traité comme un espace-temps 2D muni de la
métrique de Minkowski :

    Δs² = c_mkt² · Δt² − Δx²

Avec Δt = 1 barre et Δx = log-return sur la barre :

    Δs² = c_mkt² − Δx²

où `c_mkt` est une « vitesse limite du marché », calibrée par quantile
glissant sur les |log-return| passés.

Genres d'intervalle
-------------------
- TIMELIKE   : Δs² > 0  → mouvement causal ordinaire
- SPACELIKE  : Δs² < 0  → choc, saut de prix, mouvement non causal
- LIGHTLIKE  : |Δs²| ≈ 0 → frontière (rare, par construction)
- INCONNU    : phase de chauffe ou barre non clôturée

Mesures exposées
----------------
- `minkowski_ds2`  : la valeur brute de Δs² (continue, peut être négative)
- `minkowski_beta` : ratio |Δx| / c_mkt ∈ [0, ~1.5] (vitesse relative)

Résolution des paramètres
-------------------------
Cascade : explicite > profile > default.yaml > fallback.

Causalité
---------
1) `c_mkt` est calculé sur les |log-return| PASSÉS (shift(1)).
2) Classification barre par barre, sans lookahead.
3) `is_partial=True` → INCONNU.

Le c_mkt glissant est paramétrable (fenêtre + quantile), ce qui permet
de l'adapter par actif et par timeframe via le profil YAML.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

import numpy as np
import pandas as pd

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method
from regime_lib.core.utils import (
    TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS,
    unaligned_mask,
)


@register_method
class MinkowskiCausalDetector(RegimeDetector):
    """
    Détecteur de régimes par classification causale de Minkowski.

    Paramètres
    ----------
    fenetre_c : int
        Fenêtre glissante pour le calcul de `c_mkt`.
    q_c : float
        Quantile pour `c_mkt` ∈ (0, 1). Défaut : 0.90.
    seuil_lightlike : float
        Tolérance relative pour la classe LIGHTLIKE.
        |Δs²| < seuil_lightlike · c_mkt² → LIGHTLIKE.
        Défaut : 0.01 (1 % de c²).
    """

    name = "minkowski_causal"
    REGIME_MAP = {
        "TIMELIKE": 0,
        "SPACELIKE": 1,
        "LIGHTLIKE": 2,
        "INCONNU": 3,
    }
    requires_lookahead = False

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        fenetre_c: int | None = None,
        min_periods_c: int | None = None,
        q_c: float | None = None,
        seuil_lightlike: float | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            fenetre_c=fenetre_c,
            min_periods_c=min_periods_c,
            q_c=q_c,
            seuil_lightlike=seuil_lightlike,
        )

        self.fenetre_c = int(self._resolve(
            "fenetre_c", fenetre_c, default=500,
            per_timeframe=True,
        ))
        resolved_min = self._resolve(
            "min_periods_c", min_periods_c, default=None,
            per_timeframe=True,
        )
        if resolved_min is not None:
            self.min_periods_c = int(resolved_min)
        else:
            self.min_periods_c = max(1, self.fenetre_c // 2)

        self.q_c = float(self._resolve("q_c", q_c, default=0.90))
        self.seuil_lightlike = float(self._resolve(
            "seuil_lightlike", seuil_lightlike, default=0.01
        ))

        if self.fenetre_c < 2:
            raise ValueError("fenetre_c doit être >= 2.")
        if self.min_periods_c < 1:
            raise ValueError("min_periods_c doit être >= 1.")
        if self.min_periods_c > self.fenetre_c:
            raise ValueError(
                "min_periods_c doit être <= fenetre_c "
                f"(reçu min={self.min_periods_c}, fenetre={self.fenetre_c})."
            )
        if not (0.0 < self.q_c < 1.0):
            raise ValueError("q_c doit être dans (0, 1).")
        if self.seuil_lightlike < 0.0:
            raise ValueError("seuil_lightlike doit être >= 0.")

    @staticmethod
    def _unaligned_mask(
        index: pd.DatetimeIndex, timeframe: str | None
    ) -> np.ndarray:
        """Delegue vers regime_lib.core.utils.unaligned_mask."""
        return unaligned_mask(index, timeframe)

    @validated_detector
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [c.lower() if isinstance(c, str) else c
                       for c in out.columns]

        n = len(out)
        if n == 0:
            out["regime"] = pd.Series([], dtype=object)
            out["confidence"] = pd.Series([], dtype=float)
            out["minkowski_ds2"] = pd.Series([], dtype=float)
            out["minkowski_beta"] = pd.Series([], dtype=float)
            return out[[
                "regime", "confidence", "minkowski_ds2", "minkowski_beta"
            ]]

        close = out["close"].to_numpy(dtype=float)

        # --- 1) Log-return (Δx) -----------------------------------------
        log_return = np.full(n, np.nan, dtype=float)
        log_return[1:] = np.log(close[1:] / close[:-1])

        # --- 2) c_mkt causal (quantile glissant sur |Δx| passés) --------
        abs_ret = pd.Series(np.abs(log_return), index=out.index)
        c_series = abs_ret.shift(1).rolling(
            self.fenetre_c, min_periods=self.min_periods_c
        ).quantile(self.q_c)
        c = c_series.to_numpy()

        # --- 3) Intervalle de Minkowski : Δs² = c² − Δx² ----------------
        with np.errstate(invalid="ignore"):
            c2 = c ** 2
            ds2 = c2 - log_return ** 2
            seuil_abs = self.seuil_lightlike * c2

        # --- 4) Classification ------------------------------------------
        ready = ~np.isnan(ds2) & ~np.isnan(c) & (c > 0)

        mask_timelike = ready & (ds2 > seuil_abs)
        mask_spacelike = ready & (ds2 < -seuil_abs)
        mask_lightlike = ready & (np.abs(ds2) <= seuil_abs)

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_timelike] = "TIMELIKE"
        regime[mask_spacelike] = "SPACELIKE"
        regime[mask_lightlike] = "LIGHTLIKE"

        # --- 5) Confidence ----------------------------------------------
        # Basée sur |Δs²| / c² : plus on s'éloigne de la frontière, plus
        # la confiance est haute.
        confidence = np.full(n, 0.0, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(c2 > 0, np.abs(ds2) / c2, 0.0)
        ratio = np.clip(ratio, 0.0, 1.0)
        confidence[mask_timelike] = 0.5 + 0.5 * ratio[mask_timelike]
        confidence[mask_spacelike] = 0.5 + 0.5 * ratio[mask_spacelike]
        confidence[mask_lightlike] = 0.5

        # --- 6) Mesures brutes ------------------------------------------
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(c > 0, np.abs(log_return) / c, np.nan)

        # --- 7) Bougies non clôturées -----------------------------------
        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
            ds2[partial] = np.nan
            beta[partial] = np.nan
        else:
            unaligned = unaligned_mask(out.index, self.timeframe)
            regime[unaligned] = "INCONNU"
            confidence[unaligned] = 0.0
            ds2[unaligned] = np.nan
            beta[unaligned] = np.nan

        out["regime"] = regime
        out["confidence"] = confidence
        out["minkowski_ds2"] = ds2
        out["minkowski_beta"] = beta
        return out[[
            "regime", "confidence", "minkowski_ds2", "minkowski_beta"
        ]]
