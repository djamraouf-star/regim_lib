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
Résolveur commun : explicite > surcharges du profil > défauts YAML.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

import numpy as np
import pandas as pd

from regime_lib.config.parameters import MethodProfile, Parameter, ParameterSchema, Ordered
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
    """Moyenne initiale de n valeurs, puis récurrence de Wilder.

    Les NaN interrompent la chauffe : une nouvelle fenêtre complète est requise.
    """
    out = np.full(len(values), np.nan)
    count = 0
    total = 0.0
    previous = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            count, total, previous = 0, 0.0, np.nan
            continue
        if count < n:
            count += 1
            total += value
            if count == n:
                previous = total / n
                out[i] = previous
        else:
            previous += (value - previous) / n
            out[i] = previous
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

    # Convention TA-Lib : somme des n-1 premiers mouvements, puis
    # récurrence à partir de la barre n. Pas d'arrondi intermédiaire.
    def smooth_movement(values):
        result = np.full(len(values), np.nan)
        if len(values) <= n:
            return result
        previous = values[1:n].sum()
        for i in range(n, len(values)):
            previous = previous - previous / n + values[i]
            result[i] = previous
        return result

    atr_s = smooth_movement(tr)
    plus_dm_s = smooth_movement(plus_dm)
    minus_dm_s = smooth_movement(minus_dm)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(atr_s > 0, 100.0 * plus_dm_s / atr_s, 0.0)
        minus_di = np.where(atr_s > 0, 100.0 * minus_dm_s / atr_s, 0.0)

        plus_di[~np.isfinite(atr_s)] = np.nan
        minus_di[~np.isfinite(atr_s)] = np.nan
        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = np.where(di_sum > 0, 100.0 * di_diff / di_sum, 0.0)

    dx[~np.isfinite(atr_s)] = np.nan
    adx = _wilder_smooth(dx, n)

    return plus_di, minus_di, adx


# ---------------------------------------------------------------------------
# Détecteur
# ---------------------------------------------------------------------------
@register_method
class ADXDetector(RegimeDetector):
    """Détecteur de régimes par ADX de Wilder."""

    name = "adx"
    confidence_kind = 'heuristic_margin'
    confidence_description = 'Marge aux seuils ADX, transition à 0.5.'
    availability = 'bar_close'
    regime_dimension = 'trend_strength'
    regime_description = 'Force de tendance ADX ; le label ne constitue pas une direction de position.'
    REGIME_MAP = {
        "TENDANCE_HAUSSIERE": 0,
        "TENDANCE_BAISSIERE": 1,
        "RANGE": 2,
        "CHOP": 3,
        "INCONNU": 4,
    }
    requires_lookahead = False

    PARAM_SCHEMA = ParameterSchema(
        parameters={
            "n_adx": Parameter(int, minimum=2),
            "seuil_range": Parameter(float, minimum=0, maximum=100, exclusive_min=True, exclusive_max=True),
            "seuil_tendance": Parameter(float, minimum=0, maximum=100, exclusive_min=True, exclusive_max=True),
        },
        constraints=(Ordered("seuil_range", "seuil_tendance"),),
    )

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | MethodProfile | None = None,
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
