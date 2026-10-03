"""
Détection de régimes par entropie de Shannon des rendements.

Principe
--------
Sur une fenêtre glissante de rendements, on discrétise leur distribution
en N bins et on calcule l'entropie de Shannon :

    H = -Σ p_i · log(p_i)

H est normalisée par log(N) : H_norm = H / log(N) ∈ [0, 1].

Régimes produits : ORDONNE / NEUTRE / DESORDRE / INCONNU.
Mesure exposée  : shannon_value ∈ [0, 1].
Causalité       : fenêtre strictement passée, aucun lookahead.
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


def _shannon_windowed(
    returns: np.ndarray, fenetre: int, n_bins: int,
) -> np.ndarray:
    """
    Entropie de Shannon normalisée, calculée sur fenêtre glissante.

    Tolérance numérique : si (hi - lo) est négligeable devant l'échelle,
    la distribution est considérée dégénérée et l'entropie est fixée à 0.
    """
    n = len(returns)
    out = np.full(n, np.nan, dtype=float)
    if n < fenetre:
        return out

    log_nbins = np.log(n_bins)
    for t in range(fenetre - 1, n):
        window = returns[t - fenetre + 1 : t + 1]
        window = window[~np.isnan(window)]
        if len(window) < fenetre // 2:
            continue
        lo, hi = window.min(), window.max()

        scale = max(abs(lo), abs(hi), 1e-12)
        if (hi - lo) / scale < 1e-9:
            out[t] = 0.0
            continue

        hist, _ = np.histogram(window, bins=n_bins, range=(lo, hi))
        p = hist / hist.sum()
        p = p[p > 0]
        h = -np.sum(p * np.log(p))
        out[t] = h / log_nbins
    return out


@register_method
class ShannonEntropyDetector(RegimeDetector):
    """Détecteur de régimes par entropie de Shannon."""

    name = "shannon"
    REGIME_MAP = {
        "ORDONNE": 0,
        "NEUTRE": 1,
        "DESORDRE": 2,
        "INCONNU": 3,
    }
    requires_lookahead = False

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        fenetre: int | None = None,
        n_bins: int | None = None,
        seuil_ordonne: float | None = None,
        seuil_desordre: float | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            fenetre=fenetre,
            n_bins=n_bins,
            seuil_ordonne=seuil_ordonne,
            seuil_desordre=seuil_desordre,
        )
        self.fenetre = int(self._resolve(
            "fenetre", fenetre, default=100, per_timeframe=True,
        ))
        self.n_bins = int(self._resolve("n_bins", n_bins, default=10))
        self.seuil_ordonne = float(self._resolve(
            "seuil_ordonne", seuil_ordonne, default=0.65
        ))
        self.seuil_desordre = float(self._resolve(
            "seuil_desordre", seuil_desordre, default=0.78
        ))

        if self.fenetre < 10:
            raise ValueError("fenetre doit être >= 10.")
        if self.n_bins < 2:
            raise ValueError("n_bins doit être >= 2.")
        if not (0.0 < self.seuil_ordonne < self.seuil_desordre < 1.0):
            raise ValueError(
                "Il faut 0 < seuil_ordonne < seuil_desordre < 1."
            )

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

        n = len(out)
        if n == 0:
            out["regime"] = pd.Series([], dtype=object)
            out["confidence"] = pd.Series([], dtype=float)
            out["shannon_value"] = pd.Series([], dtype=float)
            return out[["regime", "confidence", "shannon_value"]]

        close = out["close"].to_numpy(dtype=float)

        log_ret = np.full(n, np.nan, dtype=float)
        log_ret[1:] = np.log(close[1:] / close[:-1])

        h_norm = _shannon_windowed(log_ret, self.fenetre, self.n_bins)

        ready = ~np.isnan(h_norm)
        mask_ordonne = ready & (h_norm < self.seuil_ordonne)
        mask_desordre = ready & (h_norm > self.seuil_desordre)
        mask_neutre = ready & ~mask_ordonne & ~mask_desordre

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_ordonne] = "ORDONNE"
        regime[mask_neutre] = "NEUTRE"
        regime[mask_desordre] = "DESORDRE"

        confidence = np.full(n, 0.0, dtype=float)
        if mask_ordonne.any():
            conf_ord = 1.0 - (h_norm[mask_ordonne] / self.seuil_ordonne)
            confidence[mask_ordonne] = 0.5 + 0.5 * np.clip(
                conf_ord, 0.0, 1.0
            )
        if mask_desordre.any():
            conf_des = (h_norm[mask_desordre] - self.seuil_desordre) / (
                1.0 - self.seuil_desordre
            )
            confidence[mask_desordre] = 0.5 + 0.5 * np.clip(
                conf_des, 0.0, 1.0
            )
        if mask_neutre.any():
            confidence[mask_neutre] = 0.5

        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
            h_norm[partial] = np.nan
        else:
            unaligned = self._unaligned_mask(out.index, self.timeframe)
            regime[unaligned] = "INCONNU"
            confidence[unaligned] = 0.0
            h_norm[unaligned] = np.nan

        out["regime"] = regime
        out["confidence"] = confidence
        out["shannon_value"] = h_norm
        return out[["regime", "confidence", "shannon_value"]]
