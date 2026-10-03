"""
Market Structure Shift (MSS) et Change of Character (CHOCH) — style ICT.

Concept
-------
La structure de marché est définie par les swing highs (sommets) et
swing lows (creux) :

  - Uptrend   : Higher Highs (HH) + Higher Lows (HL)
  - Downtrend : Lower Highs (LH) + Lower Lows (LL)

Deux événements marquent les retournements :

- CHOCH (Change of Character) : premier signal de retournement.
  En uptrend, cassure du dernier HL → CHOCH baissier.
  En downtrend, cassure du dernier LH → CHOCH haussier.

- MSS (Market Structure Shift) : confirmation du retournement.
  Après un CHOCH, cassure du swing opposé majeur → MSS confirmé.

Régimes produits
----------------
- MSS_HAUSSIER   : cassure haussière confirmée (retournement vers le haut)
- MSS_BAISSIER   : cassure baissière confirmée
- CHOCH_HAUSSIER : premier signal haussier (non confirmé)
- CHOCH_BAISSIER : premier signal baissier (non confirmé)
- NEUTRE         : structure stable, pas de cassure
- INCONNU        : chauffe ou barre non clôturée

⚠ MÉTHODE OFFLINE — LOOKAHEAD STRUCTUREL
----------------------------------------
Comme `price_action`, cette méthode utilise des fractales dont la
détection NÉCESSITE n_fractale barres futures. Le flag
`requires_lookahead = True` empêche l'utilisation accidentelle.

Différence avec `price_action`
------------------------------
- `price_action` : classifie un RÉGIME (persistant) → TENDANCE/RANGE/CHOP
- `mss_choch`    : classe l'état de cassure barre par barre → CHOCH/MSS

Les deux sont complémentaires : un CHOCH signale la fin d'une tendance
que `price_action` classifiait comme TENDANCE_*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method


_TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800,
}


def _detect_fractals(
    high: np.ndarray, low: np.ndarray, n_fractale: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Détection de fractales strictes (même logique que `price_action`).
    """
    n = len(high)
    window = 2 * n_fractale + 1
    is_sommet = np.zeros(n, dtype=bool)
    is_creux = np.zeros(n, dtype=bool)
    if n < window:
        return is_sommet, is_creux

    h_win = sliding_window_view(high, window)
    l_win = sliding_window_view(low, window)

    center = n_fractale
    mask = np.ones(window, dtype=bool)
    mask[center] = False

    is_sommet_win = np.all(h_win[:, center, None] > h_win[:, mask], axis=1)
    is_creux_win = np.all(l_win[:, center, None] < l_win[:, mask], axis=1)

    is_sommet[n_fractale:n - n_fractale] = is_sommet_win
    is_creux[n_fractale:n - n_fractale] = is_creux_win
    return is_sommet, is_creux


def _build_alternating_pivots(
    is_sommet: np.ndarray,
    is_creux: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> list[tuple[int, str, float]]:
    """
    Construit la liste chronologique des pivots strictement alternés S/C.
    """
    n = len(high)
    pivots: list[tuple[int, str, float]] = []
    for i in range(n):
        if is_sommet[i]:
            if not pivots or pivots[-1][1] == 'C':
                pivots.append((i, 'S', float(high[i])))
            elif high[i] > pivots[-1][2]:
                pivots[-1] = (i, 'S', float(high[i]))
        if is_creux[i]:
            if not pivots or pivots[-1][1] == 'S':
                pivots.append((i, 'C', float(low[i])))
            elif low[i] < pivots[-1][2]:
                pivots[-1] = (i, 'C', float(low[i]))
    return pivots


def _classify_mss_choch(
    pivots: list[tuple[int, str, float]],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Classification MSS/CHOCH barre par barre.

    Algorithme :
    1. Construire un index des pivots par indice de barre.
    2. Pour chaque barre t :
       a. Récupérer les 2 derniers sommets et 2 derniers creux confirmés
          jusqu'à t (via les pivots ayant index <= t).
       b. Déterminer la structure :
          - Uptrend  : S1 > S2 ET C1 > C2
          - Downtrend: S1 < S2 ET C1 < C2
          - Neutral  : sinon
       c. Si uptrend :
          - close[t] < C1 → CHOCH_BAISSIER
          - close[t] < C2 → MSS_BAISSIER (plus fort)
       d. Si downtrend :
          - close[t] > S1 → CHOCH_HAUSSIER
          - close[t] > S2 → MSS_HAUSSIER (plus fort)
       e. Sinon NEUTRE
    """
    regime = np.full(n, "INCONNU", dtype=object)

    # Index des pivots par ordre d'apparition (croissant)
    pivot_indices = [p[0] for p in pivots]

    # Pointeur courant dans les pivots (avance au fil du temps)
    j = 0  # prochain pivot pas encore disponible à t
    derniers_s: list[float] = []  # [S1 (récent), S2 (ancien)] ou moins
    derniers_c: list[float] = []  # [C1 (récent), C2 (ancien)] ou moins

    for t in range(n):
        # Intégrer tous les pivots ayant index <= t
        while j < len(pivots) and pivots[j][0] <= t:
            _, typ, val = pivots[j]
            if typ == 'S':
                # Alternance garantie : si on a déjà un S récent, on
                # remplace par le plus extrême (mais _build_alternating
                # a déjà appliqué cette règle). On prépend simplement.
                derniers_s.insert(0, val)
                derniers_s = derniers_s[:2]
            else:
                derniers_c.insert(0, val)
                derniers_c = derniers_c[:2]
            j += 1

        if len(derniers_s) < 2 or len(derniers_c) < 2:
            continue

        s1, s2 = derniers_s  # s1 = plus récent
        c1, c2 = derniers_c  # c1 = plus récent

        uptrend = (s1 > s2) and (c1 > c2)
        downtrend = (s1 < s2) and (c1 < c2)

        if uptrend:
            # c1 = dernier Higher Low, c2 = HL précédent
            if close[t] < c2:
                regime[t] = "MSS_BAISSIER"
            elif close[t] < c1:
                regime[t] = "CHOCH_BAISSIER"
            else:
                regime[t] = "NEUTRE"
        elif downtrend:
            # s1 = dernier Lower High, s2 = LH précédent
            if close[t] > s2:
                regime[t] = "MSS_HAUSSIER"
            elif close[t] > s1:
                regime[t] = "CHOCH_HAUSSIER"
            else:
                regime[t] = "NEUTRE"
        else:
            regime[t] = "NEUTRE"

    return regime


@register_method
class MSSCHOCHDetector(RegimeDetector):
    """
    Détecteur de retournements MSS/CHOCH (style ICT).

    Méthode OFFLINE : utilise du lookahead structurel (détection de
    fractales). Voir docstring module.
    """

    name = "mss_choch"
    REGIME_MAP = {
        "MSS_HAUSSIER": 0,
        "MSS_BAISSIER": 1,
        "CHOCH_HAUSSIER": 2,
        "CHOCH_BAISSIER": 3,
        "NEUTRE": 4,
        "INCONNU": 5,
    }
    requires_lookahead = True

    N_FRACTALE: dict[str, int] = {
        "M1": 5, "M5": 5, "M15": 5, "M30": 5,
        "H1": 5, "H4": 5, "D1": 5, "W1": 5,
    }

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        n_fractale: int | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            n_fractale=n_fractale,
        )
        self.n_fractale = int(self._resolve(
            "n_fractale", n_fractale,
            default=self.N_FRACTALE.get(timeframe or "", 5),
            per_timeframe=True,
        ))
        if self.n_fractale < 1:
            raise ValueError("n_fractale doit être >= 1.")

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
            return out[["regime", "confidence"]]

        high = out["high"].to_numpy(dtype=float)
        low = out["low"].to_numpy(dtype=float)
        close = out["close"].to_numpy(dtype=float)

        # 1) Fractales puis pivots alternés
        is_sommet, is_creux = _detect_fractals(high, low, self.n_fractale)
        pivots = _build_alternating_pivots(is_sommet, is_creux, high, low)

        # 2) Classification
        regime = _classify_mss_choch(pivots, high, low, close, n)

        # 3) Confidence : plus élevée pour MSS (confirmé) que CHOCH
        confidence = np.full(n, 0.0, dtype=float)
        confidence[regime == "NEUTRE"] = 0.5
        confidence[regime == "CHOCH_HAUSSIER"] = 0.6
        confidence[regime == "CHOCH_BAISSIER"] = 0.6
        confidence[regime == "MSS_HAUSSIER"] = 0.8
        confidence[regime == "MSS_BAISSIER"] = 0.8

        # 4) Bougies non clôturées
        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
        else:
            unaligned = self._unaligned_mask(out.index, self.timeframe)
            regime[unaligned] = "INCONNU"
            confidence[unaligned] = 0.0

        out["regime"] = regime
        out["confidence"] = confidence
        return out[["regime", "confidence"]]
