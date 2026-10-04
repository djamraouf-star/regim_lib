"""
Détection de régimes par ADX (Average Directional Index, Wilder).

Régimes produits
----------------
- TENDANCE_HAUSSIERE : ADX >= seuil_tendance ET +DI > -DI
- TENDANCE_BAISSIERE : ADX >= seuil_tendance ET -DI > +DI
- RANGE               : ADX < seuil_range (absence de tendance)
- CHOP                : ADX entre les deux seuils (zone de transition)
- INCONNU             : phase de chauffe ou barre non clôturée

⚠ Sémantique croisée avec les autres méthodes
---------------------------------------------
Le RANGE ADX (ADX < 20) signifie « absence de tendance », pas « faible
volatilité » comme pour ATR.

Causalité
---------
ADX est causal par construction : aucun lookahead.

Résolution des paramètres
-------------------------
Cascade : explicite > profile > fallback codé en dur.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

import numpy as np
import pandas as pd

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method
from regime_lib.core.utils import (
    TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS,
    true_range as _true_range,
    unaligned_mask,
)


def _directional_movement(
    high: np.ndarray, low: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """+DM et -DM selon Wilder."""
    n = len(high)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    if n < 2:
        return plus_dm, minus_dm

    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]

    mask_plus = (up > down) & (up > 0)
    plus_dm[1:] = np.where(mask_plus, up, 0.0)

    mask_minus = (down > up) & (down > 0)
    minus_dm[1:] = np.where(mask_minus, down, 0.0)

    return plus_dm, minus_dm


def _wilder_smooth(values: np.ndarray, n: int) -> np.ndarray:
    """
    Lissage de Wilder : EMA avec alpha = 1/n, adjust=False.
    NaN sur les n-1 premières barres.
    Note : .tolist() avant np.array pour compatibilité Arrow/pandas 2.x.
    """
    s = pd.Series(values).ewm(alpha=1.0 / n, adjust=False).mean()
    out = np.array(s.tolist(), dtype=float)
    out[: n - 1] = np.nan
    return out


def _compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcule +DI, -DI et ADX selon Wilder."""
    tr = _true_range(high, low, close)
    plus_dm, minus_dm = _directional_movement(high, low)

    atr_s = _wilder_smooth(tr, n)
    plus_dm_s = _wilder_smooth(plus_dm, n)
    minus_dm_s = _wilder_smooth(minus_dm, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(atr_s > 0, 100.0 * plus_dm_s / atr_s, np.nan)
        minus_di = np.where(atr_s > 0, 100.0 * minus_dm_s / atr_s, np.nan)

        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = np.where(di_sum > 0, 100.0 * di_diff / di_sum, np.nan)

    adx_series = pd.Series(dx).ewm(alpha=1.0 / n, adjust=False).mean()
    adx = np.array(adx_series.tolist(), dtype=float)
    adx[: 2 * n - 1] = np.nan

    return plus_di, minus_di, adx


# ---------------------------------------------------------------------------
# Détecteur
# ---------------------------------------------------------------------------
@register_method
class ADXDetector(RegimeDetector):
    """Détecteur de régimes par ADX de Wilder."""

    name = "adx"
    REGIME_MAP = {
        "TENDANCE_HAUSSIERE": 0,
        "TENDANCE_BAISSIERE": 1,
        "RANGE": 2,
        "CHOP": 3,
        "INCONNU": 4,
    }
    requires_lookahead = False

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        n_adx: int | None = None,
        seuil_range: float | None = None,
        seuil_tendance: float | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            n_adx=n_adx,
            seuil_range=seuil_range,
            seuil_tendance=seuil_tendance,
        )
        self.n_adx = int(self._resolve("n_adx", n_adx, default=14))
        self.seuil_range = float(self._resolve(
            "seuil_range", seuil_range, default=20.0
        ))
        self.seuil_tendance = float(self._resolve(
            "seuil_tendance", seuil_tendance, default=25.0
        ))

        if self.n_adx < 2:
            raise ValueError("n_adx doit être >= 2.")
        if not (0.0 < self.seuil_range < self.seuil_tendance < 100.0):
            raise ValueError(
                "Il faut 0 < seuil_range < seuil_tendance < 100."
            )

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
            return out[["regime", "confidence"]]

        high = out["high"].to_numpy(dtype=float)
        low = out["low"].to_numpy(dtype=float)
        close = out["close"].to_numpy(dtype=float)

        plus_di, minus_di, adx = _compute_adx(
            high, low, close, self.n_adx
        )

        ready = (
            ~np.isnan(adx) & ~np.isnan(plus_di) & ~np.isnan(minus_di)
        )

        mask_range = ready & (adx < self.seuil_range)
        mask_chop = (
            ready
            & (adx >= self.seuil_range)
            & (adx < self.seuil_tendance)
        )
        mask_up = (
            ready
            & (adx >= self.seuil_tendance)
            & (plus_di > minus_di)
        )
        mask_down = (
            ready
            & (adx >= self.seuil_tendance)
            & (minus_di > plus_di)
        )
        mask_ambigu = (
            ready
            & (adx >= self.seuil_tendance)
            & (plus_di == minus_di)
        )

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_range] = "RANGE"
        regime[mask_chop] = "CHOP"
        regime[mask_up] = "TENDANCE_HAUSSIERE"
        regime[mask_down] = "TENDANCE_BAISSIERE"
        regime[mask_ambigu] = "CHOP"

        confidence = np.full(n, 0.0, dtype=float)
        confidence[mask_chop] = 0.5
        confidence[mask_ambigu] = 0.5

        if mask_range.any():
            dist = (self.seuil_range - adx[mask_range]) / self.seuil_range
            confidence[mask_range] = 0.5 + 0.5 * np.clip(dist, 0.0, 1.0)

        denom = 100.0 - self.seuil_tendance
        if mask_up.any():
            dist = (adx[mask_up] - self.seuil_tendance) / denom
            confidence[mask_up] = 0.5 + 0.5 * np.clip(dist, 0.0, 1.0)
        if mask_down.any():
            dist = (adx[mask_down] - self.seuil_tendance) / denom
            confidence[mask_down] = 0.5 + 0.5 * np.clip(dist, 0.0, 1.0)

        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
        else:
            unaligned = unaligned_mask(out.index, self.timeframe)
            regime[unaligned] = "INCONNU"
            confidence[unaligned] = 0.0

        out["regime"] = regime
        out["confidence"] = confidence
        return out[["regime", "confidence"]]
