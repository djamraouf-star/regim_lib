"""
Détection de régimes par Price Action (fractales strictes + pivots alternés).

⚠ MÉTHODE OFFLINE — LOOKAHEAD STRUCTUREL
----------------------------------------
Cette méthode utilise un lookahead **par construction**. Un sommet fractal
à l'indice `i` n'est identifiable qu'après avoir observé les `n_fractale`
barres suivantes. La variante distincte price_action_causal publie les
événements à confirmation ; elle ne réécrit pas les labels historiques.

Usage prévu :
  - Analyse descriptive a posteriori (post-mortem, annotation de dataset) ;
  - Construction de labels pour l'apprentissage supervisé offline ;
  - Comparaison avec les méthodes causales pour mesurer l'écart.


Usage DÉCONSEILLÉ :
  - Signal tradable en temps réel ;
  - Criblage de « potentiel capturable » au sens strict.

Le flag `requires_lookahead = True` empêche l'utilisation accidentelle.
Le CLI exige `--allow-lookahead-override price_action:true` pour la
lancer, et trace le run comme « descriptif » dans `run_meta.json`.

Régimes produits
----------------
- TENDANCE_HAUSSIERE : 2 derniers sommets ascendants ET 2 derniers creux ascendants
- TENDANCE_BAISSIERE : 2 derniers sommets descendants ET 2 derniers creux descendants
- RANGE               : sommets et creux quasi-plats (tolérance ATR)
- CHOP                : configuration mixte (ambiguë)
- INCONNU             : phase de chauffe ou barre non clôturée

Confidence
----------
- INCONNU : 0.0
- CHOP    : 0.5 (configuration ambiguë par construction)
- RANGE   : `0.5 + 0.5 * (1 - max(abs(diff_s), abs(diff_c)) / tol)`
- TENDANCE_* : `0.5 + 0.5 * min(margin_s, margin_c) / tol`

où `margin = abs(diff) - tol` et `tol = k_atr * ATR`.

Optimisation
------------
La détection de fractales est vectorisée via `sliding_window_view` de
numpy, sans dépendance externe. Équivalent strict à `argrelextrema` de
scipy pour des inégalités strictes, avec exclusion explicite des bords
(cas que scipy gère par défaut en mode 'clip').

Résolution des paramètres
-------------------------
Résolveur commun : explicite > surcharges du profil > défauts YAML.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

import numpy as np
import pandas as pd

from regime_lib.config.parameters import MethodProfile, Parameter, ParameterSchema
from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method   # ← IMPORT CRITIQUE
from regime_lib.core.utils import (
    TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS,
    true_range,
    detect_fractals as _detect_fractals,
    build_alternating_pivots as _build_alternating_pivots,
    unaligned_mask,
)


# ---------------------------------------------------------------------------
# Helpers (hors classe, testables unitairement)
# ---------------------------------------------------------------------------
def _atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    n: int, min_periods: int = 1,
) -> np.ndarray:
    """Average True Range, causal (pour la tolérance uniquement)."""
    tr = true_range(high, low, close)
    return pd.Series(tr).rolling(n, min_periods=min_periods).mean().to_numpy()


def _classify_from_pivots(
    pivots: list[tuple[int, str, float]],
    atr: np.ndarray,
    n: int,
    k_atr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classification par pivot, indexée par l'indice du pivot."""
    regime_arr = np.full(n, "INCONNU", dtype=object)
    diff_s_arr = np.full(n, np.nan, dtype=float)
    diff_c_arr = np.full(n, np.nan, dtype=float)
    tol_arr = np.full(n, np.nan, dtype=float)

    for j in range(3, len(pivots)):
        derniers_s: list[float] = []
        derniers_c: list[float] = []
        for k in range(j, -1, -1):
            if pivots[k][1] == 'S' and len(derniers_s) < 2:
                derniers_s.append(pivots[k][2])
            elif pivots[k][1] == 'C' and len(derniers_c) < 2:
                derniers_c.append(pivots[k][2])
            if len(derniers_s) == 2 and len(derniers_c) == 2:
                break

        if len(derniers_s) < 2 or len(derniers_c) < 2:
            continue

        s1, s2 = derniers_s
        c1, c2 = derniers_c
        idx = pivots[j][0]
        a = atr[idx]
        tol = k_atr * a if not np.isnan(a) else 0.0005
        if tol <= 0:
            tol = 1e-12

        diff_s = s1 - s2
        diff_c = c1 - c2

        if diff_s > tol and diff_c > tol:
            regime_arr[idx] = "TENDANCE_HAUSSIERE"
        elif diff_s < -tol and diff_c < -tol:
            regime_arr[idx] = "TENDANCE_BAISSIERE"
        elif abs(diff_s) <= tol and abs(diff_c) <= tol:
            regime_arr[idx] = "RANGE"
        else:
            regime_arr[idx] = "CHOP"

        diff_s_arr[idx] = diff_s
        diff_c_arr[idx] = diff_c
        tol_arr[idx] = tol

    return regime_arr, diff_s_arr, diff_c_arr, tol_arr


def _propagate_and_confidence(
    regime_arr: np.ndarray,
    diff_s_arr: np.ndarray,
    diff_c_arr: np.ndarray,
    tol_arr: np.ndarray,
    index: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagation forward-fill du régime + calcul de la confidence.

    Note : on passe par `.tolist()` avant `np.array` pour garantir
    un tableau en écriture (compatibilité Arrow/pandas 2.x).
    """
    n = len(regime_arr)

    regime_series = pd.Series(regime_arr, index=index)
    regime_series = (regime_series
                     .replace("INCONNU", np.nan)
                     .ffill()
                     .fillna("INCONNU"))
    regime_prop = np.array(regime_series.tolist(), dtype=object)

    conf_at_pivot = np.full(n, np.nan, dtype=float)

    mask = (regime_arr == "CHOP") & ~np.isnan(tol_arr)
    conf_at_pivot[mask] = 0.5

    mask = (regime_arr == "RANGE") & ~np.isnan(tol_arr)
    if mask.any():
        d = np.maximum(np.abs(diff_s_arr[mask]), np.abs(diff_c_arr[mask]))
        t = tol_arr[mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(t > 0, d / t, 0.0)
        conf_at_pivot[mask] = 0.5 + 0.5 * (1.0 - np.clip(ratio, 0.0, 1.0))

    for label, sign in (("TENDANCE_HAUSSIERE", 1.0),
                         ("TENDANCE_BAISSIERE", -1.0)):
        mask = (regime_arr == label) & ~np.isnan(tol_arr)
        if mask.any():
            margin = np.minimum(sign * diff_s_arr[mask],
                                sign * diff_c_arr[mask]) - tol_arr[mask]
            t = tol_arr[mask]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(t > 0, margin / t, 1.0)
            conf_at_pivot[mask] = 0.5 + 0.5 * np.clip(ratio, 0.0, 1.0)

    conf_series = pd.Series(conf_at_pivot, index=index).ffill()
    conf_prop = np.array(
        conf_series.fillna(0.0).tolist(), dtype=float
    )

    conf_prop[regime_prop == "INCONNU"] = 0.0

    return regime_prop, conf_prop


# ---------------------------------------------------------------------------
# Détecteur
# ---------------------------------------------------------------------------
@register_method
class PriceActionDetector(RegimeDetector):
    """
    Détecteur de régimes par Price Action (fractales strictes).

    Méthode OFFLINE : utilise du lookahead structurel. Voir docstring
    module pour la justification et les cas d'usage.
    """

    name = "price_action"
    confidence_kind = 'heuristic_margin'
    confidence_description = 'Marge des différences de pivots rapportée à la tolérance ATR.'
    availability = 'retrospective'
    regime_dimension = 'price_structure'
    regime_description = 'Structure rétrospective des pivots, révisable.'
    REGIME_MAP = {
        "TENDANCE_HAUSSIERE": 0,
        "TENDANCE_BAISSIERE": 1,
        "RANGE": 2,
        "CHOP": 3,
        "INCONNU": 4,
    }
    requires_lookahead = True

    PARAM_SCHEMA = ParameterSchema(
        parameters={
            "n_fractale": Parameter(int, minimum=1, per_timeframe=True),
            "k_atr": Parameter(float, minimum=0, exclusive_min=True),
            "n_atr_tol": Parameter(int, minimum=1),
        },
    )

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | MethodProfile | None = None,
        n_fractale: int | None = None,
        k_atr: float | None = None,
        n_atr_tol: int | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            n_fractale=n_fractale,
            k_atr=k_atr,
            n_atr_tol=n_atr_tol,
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

        atr = _atr(high, low, close, self.n_atr_tol, min_periods=1)
        is_sommet, is_creux = _detect_fractals(high, low, self.n_fractale)
        pivots = _build_alternating_pivots(is_sommet, is_creux, high, low)

        regime_arr, diff_s, diff_c, tol = _classify_from_pivots(
            pivots, atr, n, self.k_atr,
        )

        regime_arr, confidence_arr = _propagate_and_confidence(
            regime_arr, diff_s, diff_c, tol, out.index,
        )

        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
            regime_arr[partial] = "INCONNU"
            confidence_arr[partial] = 0.0
        else:
            unaligned = unaligned_mask(out.index, self.timeframe)
            regime_arr[unaligned] = "INCONNU"
            confidence_arr[unaligned] = 0.0

        out["regime"] = regime_arr
        out["confidence"] = confidence_arr
        return out[["regime", "confidence"]]
