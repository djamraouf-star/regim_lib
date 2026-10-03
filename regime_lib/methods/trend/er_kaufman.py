"""
Détection de régimes par Efficiency Ratio (Kaufman) et quantiles glissants.

Définition de l'ER
------------------
``ER(t) = abs(close[t] - close[t-n]) / sum(abs(close[i] - close[i-1]), i in (t-n, t])``

L'ER est borné dans [0, 1] :
  - proche de 1 : mouvement directionnel efficient ;
  - proche de 0 : bruit / chop.

Régimes produits
----------------
- CHOP                : ER <= quantile bas (bruit, pas de direction)
- RANGE               : ER strictement entre les deux quantiles
- TENDANCE_HAUSSIERE  : ER >= quantile haut ET direction haussière
- TENDANCE_BAISSIERE  : ER >= quantile haut ET direction baissière
- INCONNU             : phase de chauffe, ER indéfini, ou barre non clôturée

⚠ Sémantique croisée avec `atr_volatility`
------------------------------------------
- `atr_volatility` : `RANGE` correspond à une volatilité faible, `CHOP` à une
  volatilité intermédiaire.
- `er_kaufman` : `RANGE` correspond à une efficience intermédiaire, `CHOP` à
  une efficience faible (bruit).

Résolution des paramètres
-------------------------
Cascade : explicite > profile (avec timeframes) > fallback codé en dur.

Causalité et anti-lookahead
---------------------------
1) Les seuils sont calculés sur l'ER décalé d'une barre (`er.shift(1)`).
2) Classification causale.
3) `is_partial=True` → INCONNU.
4) Masque hors grille agnostique à la résolution DatetimeIndex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method
from regime_lib.core.utils import (
    TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS,
    direction,
    unaligned_mask,
)


@register_method
class ERKaufmanDetector(RegimeDetector):
    """Détecteur de régimes par Efficiency Ratio de Kaufman."""

    name = "er_kaufman"
    REGIME_MAP = {
        "TENDANCE_HAUSSIERE": 0,
        "TENDANCE_BAISSIERE": 1,
        "RANGE": 2,
        "CHOP": 3,
        "INCONNU": 4,
    }
    requires_lookahead = False

    N_ER: dict[str, int] = {
        "M1": 20, "M5": 20, "M15": 20, "M30": 20,
        "H1": 20, "H4": 20, "D1": 20, "W1": 20,
    }
    FENETRE_QUANTILE: dict[str, int] = {
        "M1": 8000, "M5": 3000, "M15": 2000, "M30": 1500,
        "H1": 700, "H4": 200, "D1": 40, "W1": 20,
    }

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        q_chop: float | None = None,
        q_tendance: float | None = None,
        n_er: int | None = None,
        fenetre: int | None = None,
        min_periods: int | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            q_chop=q_chop,
            q_tendance=q_tendance,
            n_er=n_er,
            fenetre=fenetre,
            min_periods=min_periods,
        )

        self.q_chop = float(self._resolve(
            "q_chop", q_chop, default=0.40
        ))
        self.q_tendance = float(self._resolve(
            "q_tendance", q_tendance, default=0.60
        ))
        self.n_er = int(self._resolve(
            "n_er", n_er,
            default=self.N_ER.get(timeframe or "", 20),
            per_timeframe=True,
        ))
        self.fenetre = int(self._resolve(
            "fenetre", fenetre,
            default=self.FENETRE_QUANTILE.get(timeframe or "", 700),
            per_timeframe=True,
        ))

        resolved_min = self._resolve(
            "min_periods", min_periods, default=None, per_timeframe=True,
        )
        if resolved_min is not None:
            self.min_periods = int(resolved_min)
        else:
            self.min_periods = max(1, self.fenetre // 2)

        if self.n_er < 1:
            raise ValueError("n_er doit être >= 1.")
        if self.fenetre < 2:
            raise ValueError("fenetre doit être >= 2.")
        if not (0.0 < self.q_chop < self.q_tendance < 1.0):
            raise ValueError("Il faut 0 < q_chop < q_tendance < 1.")
        if self.min_periods < 1:
            raise ValueError("min_periods doit être >= 1.")
        if self.min_periods > self.fenetre:
            raise ValueError(
                "min_periods doit être <= fenetre "
                f"(reçu min_periods={self.min_periods}, fenetre={self.fenetre})."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _efficiency_ratio(close: np.ndarray, n: int) -> np.ndarray:
        """Efficiency Ratio de Kaufman, vectorisé numpy."""
        length = len(close)
        er = np.full(length, np.nan, dtype=float)
        if length <= n:
            return er

        change = np.abs(close[n:] - close[:-n])
        abs_diff = np.abs(np.diff(close))
        csum = np.concatenate([[0.0], np.cumsum(abs_diff)])
        volatility = csum[n:] - csum[:-n]

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(volatility > 0, change / volatility, np.nan)

        er[n:] = np.clip(ratio, 0.0, 1.0)
        return er

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [
            c.lower() if isinstance(c, str) else c for c in out.columns
        ]

        n_rows = len(out)
        if n_rows == 0:
            out["regime"] = pd.Series([], dtype=object)
            out["confidence"] = pd.Series([], dtype=float)
            return out[["regime", "confidence"]]

        close = out["close"].to_numpy(dtype=float)

        er = self._efficiency_ratio(close, self.n_er)
        direction_arr = direction(close, self.n_er)

        er_series = pd.Series(er, index=out.index)
        er_past = er_series.shift(1)
        seuil_chop = er_past.rolling(
            self.fenetre, min_periods=self.min_periods
        ).quantile(self.q_chop)
        seuil_tendance = er_past.rolling(
            self.fenetre, min_periods=self.min_periods
        ).quantile(self.q_tendance)

        s_chop = seuil_chop.to_numpy()
        s_tend = seuil_tendance.to_numpy()

        ready = ~np.isnan(er) & ~np.isnan(s_chop) & ~np.isnan(s_tend)

        mask_chop = ready & (er <= s_chop)
        mask_range = ready & (er > s_chop) & (er < s_tend)
        mask_up = ready & (er >= s_tend) & (direction_arr > 0)
        mask_down = ready & (er >= s_tend) & (direction_arr < 0)
        mask_ambigu = ready & (er >= s_tend) & (direction_arr == 0)

        regime = np.full(n_rows, "INCONNU", dtype=object)
        regime[mask_chop] = "CHOP"
        regime[mask_range] = "RANGE"
        regime[mask_up] = "TENDANCE_HAUSSIERE"
        regime[mask_down] = "TENDANCE_BAISSIERE"
        regime[mask_ambigu] = "CHOP"

        with np.errstate(divide="ignore", invalid="ignore"):
            dist_chop = np.where(
                s_chop > 0, (s_chop - er) / s_chop, 0.0
            )
            dist_tend = np.where(
                s_tend < 1.0, (er - s_tend) / (1.0 - s_tend), 0.0
            )
        dist_chop = np.clip(dist_chop, 0.0, 1.0)
        dist_tend = np.clip(dist_tend, 0.0, 1.0)

        confidence = np.full(n_rows, 0.0, dtype=float)
        confidence[mask_chop] = 0.5 + 0.5 * dist_chop[mask_chop]
        confidence[mask_range] = 0.5
        confidence[mask_up] = 0.5 + 0.5 * dist_tend[mask_up]
        confidence[mask_down] = 0.5 + 0.5 * dist_tend[mask_down]
        confidence[mask_ambigu] = 0.5

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
