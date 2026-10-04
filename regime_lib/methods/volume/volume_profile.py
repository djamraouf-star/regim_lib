"""
Détection de régimes par Volume Profile (zones de valeur glissantes).

Définition
----------
Sur une fenêtre glissante de `fenetre` barres, on construit un
histogramme du volume par niveau de prix (bins réguliers entre le low
minimum et le high maximum de la fenêtre). La **zone de valeur** est
l’intervalle [P20, P80] contenant 80 % du volume total.

Régimes produits
----------------
- DANS_ZONE_VALEUR : close ∈ [va_low, va_high]
- HORS_ZONE_HAUTE  : close > va_high
- HORS_ZONE_BASSE  : close < va_low
- INCONNU          : chauffe ou données manquantes

Mesures exposées
----------------
- `va_low`, `va_high` : bornes de la zone de valeur
- `distance_va`       : distance signée du close à la zone
                        (0 si dans la zone, <0 sous, >0 au-dessus)

Causalité
---------
Fenêtre strictement passée. Aucun lookahead.

Données requises
----------------
Colonnes : `high`, `low`, `close`, `volume`.

Note performance
----------------
Le calcul est O(n × fenetre) avec une boucle Python sur les fenêtres.
Sur 1H (≈ 8 k barres), c’est instantané. Sur 5min (≈ 95 k barres),
compter quelques secondes à quelques dizaines de secondes. Une
vectorisation par `sliding_window_view` + `np.add.at` est possible
en optimisation ultérieure.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

import numpy as np
import pandas as pd

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method


_TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800,
}


@register_method
class VolumeProfileDetector(RegimeDetector):
    """
    Détecteur de régimes par Volume Profile.

    Paramètres
    ----------
    fenetre : int
        Taille de la fenêtre glissante (en barres). Défaut : 100.
    n_bins : int
        Nombre de bins de prix dans l’histogramme. Défaut : 40.
    pct_bas : float
        Quantile de volume pour la borne basse de la zone de valeur.
        Défaut : 0.20.
    pct_haut : float
        Quantile de volume pour la borne haute de la zone de valeur.
        Défaut : 0.80. Doit être > `pct_bas`.
    """

    name = "volume_profile"
    REGIME_MAP = {
        "HORS_ZONE_BASSE": 0,
        "DANS_ZONE_VALEUR": 1,
        "HORS_ZONE_HAUTE": 2,
        "INCONNU": 3,
    }
    requires_lookahead = False
    REQUIRES_COLUMNS = ["high", "low", "close", "volume"]

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        fenetre: int | None = None,
        n_bins: int | None = None,
        pct_bas: float | None = None,
        pct_haut: float | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            fenetre=fenetre,
            n_bins=n_bins,
            pct_bas=pct_bas,
            pct_haut=pct_haut,
        )
        self.fenetre = int(self._resolve(
            "fenetre", fenetre, default=100, per_timeframe=True,
        ))
        self.n_bins = int(self._resolve(
            "n_bins", n_bins, default=40,
        ))
        self.pct_bas = float(self._resolve(
            "pct_bas", pct_bas, default=0.20,
        ))
        self.pct_haut = float(self._resolve(
            "pct_haut", pct_haut, default=0.80,
        ))

        if self.fenetre < 2:
            raise ValueError("fenetre doit être >= 2.")
        if self.n_bins < 2:
            raise ValueError("n_bins doit être >= 2.")
        if not (0.0 < self.pct_bas < 1.0):
            raise ValueError("pct_bas doit être dans (0, 1).")
        if not (0.0 < self.pct_haut < 1.0):
            raise ValueError("pct_haut doit être dans (0, 1).")
        if self.pct_haut <= self.pct_bas:
            raise ValueError("pct_haut doit être > pct_bas.")

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

    # ------------------------------------------------------------------
    # Calcul de la zone de valeur sur une fenêtre
    # ------------------------------------------------------------------
    def _value_area(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        vols: np.ndarray,
    ) -> tuple[float, float]:
        """
        Retourne (va_low, va_high) ou (nan, nan) si la fenêtre est vide.

        Le volume de chaque barre est réparti uniformément entre les bins
        traversés par [low, high]. Approximation : les bins partiellement
        recouverts aux extrémités reçoivent une part entière.
        """
        lo, hi = float(np.nanmin(lows)), float(np.nanmax(highs))
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return np.nan, np.nan

        edges = np.linspace(lo, hi, self.n_bins + 1)
        hist = np.zeros(self.n_bins, dtype=float)

        for h, l, v in zip(highs, lows, vols):
            if not (np.isfinite(h) and np.isfinite(l)) or v <= 0:
                continue
            if h <= l:
                idx = int(np.clip(
                    np.searchsorted(edges, l, side="right") - 1,
                    0, self.n_bins - 1,
                ))
                hist[idx] += v
            else:
                i_lo = int(np.clip(
                    np.searchsorted(edges, l, side="left") - 1,
                    0, self.n_bins - 1,
                ))
                i_hi = int(np.clip(
                    np.searchsorted(edges, h, side="left") - 1,
                    0, self.n_bins - 1,
                ))
                n_touched = i_hi - i_lo + 1
                hist[i_lo : i_hi + 1] += v / n_touched

        total = hist.sum()
        if total <= 0:
            return np.nan, np.nan

        cum = np.cumsum(hist) / total
        i_low = int(np.clip(
            np.searchsorted(cum, self.pct_bas, side="left"),
            0, self.n_bins - 1,
        ))
        i_high = int(np.clip(
            np.searchsorted(cum, self.pct_haut, side="left"),
            0, self.n_bins - 1,
        ))
        # On retourne les bornes du bin (pas son centre)
        return float(edges[i_low]), float(edges[i_high + 1])

    # ------------------------------------------------------------------
    # fit_predict
    # ------------------------------------------------------------------
    @validated_detector
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [c.lower() if isinstance(c, str) else c
                       for c in out.columns]
        self.verifier_colonnes(out)

        n = len(out)
        if n == 0:
            return pd.DataFrame({
                "regime": pd.Series([], dtype=object),
                "confidence": pd.Series([], dtype=float),
                "va_low": pd.Series([], dtype=float),
                "va_high": pd.Series([], dtype=float),
                "distance_va": pd.Series([], dtype=float),
            })

        highs = out["high"].to_numpy(dtype=float)
        lows = out["low"].to_numpy(dtype=float)
        closes = out["close"].to_numpy(dtype=float)
        vols = out["volume"].to_numpy(dtype=float)

        va_low = np.full(n, np.nan, dtype=float)
        va_high = np.full(n, np.nan, dtype=float)

        for i in range(self.fenetre - 1, n):
            a = i - self.fenetre + 1
            lo, hi = self._value_area(
                highs[a : i + 1],
                lows[a : i + 1],
                vols[a : i + 1],
            )
            va_low[i] = lo
            va_high[i] = hi

        # --- Classification ------------------------------------------
        ready = np.isfinite(va_low) & np.isfinite(va_high)
        mask_bas = ready & (closes < va_low)
        mask_haut = ready & (closes > va_high)
        mask_dans = ready & ~mask_bas & ~mask_haut

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_dans] = "DANS_ZONE_VALEUR"
        regime[mask_bas] = "HORS_ZONE_BASSE"
        regime[mask_haut] = "HORS_ZONE_HAUTE"

        # --- Distance signée -----------------------------------------
        distance = np.full(n, np.nan, dtype=float)
        distance[mask_dans] = 0.0
        distance[mask_bas] = closes[mask_bas] - va_low[mask_bas]
        distance[mask_haut] = closes[mask_haut] - va_high[mask_haut]

        # --- Confiance -----------------------------------------------
        # Distance normalisée par la largeur de la zone de valeur ;
        # 0.5 = pile sur une borne, → 1 = très loin dans le régime.
        confidence = np.full(n, 0.0, dtype=float)
        width = np.where(
            ready & (va_high > va_low),
            va_high - va_low,
            np.nan,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            conf_abs = np.abs(distance) / width
        conf_abs = np.clip(
            np.where(np.isfinite(conf_abs), conf_abs, 0.0),
            0.0, 1.0,
        )
        confidence[mask_dans] = 0.5
        confidence[mask_bas | mask_haut] = 0.5 + 0.5 * conf_abs[
            mask_bas | mask_haut
        ]

        # --- Barres partielles ---------------------------------------
        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
        else:
            partial = self._unaligned_mask(out.index, self.timeframe)

        if partial.any():
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
            va_low[partial] = np.nan
            va_high[partial] = np.nan
            distance[partial] = np.nan

        return pd.DataFrame(
            {
                "regime": regime,
                "confidence": confidence,
                "va_low": va_low,
                "va_high": va_high,
                "distance_va": distance,
            },
            index=out.index,
        )
