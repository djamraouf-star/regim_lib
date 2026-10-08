"""Cassures rétrospectives simplifiées de pivots.

Nom descriptif : retrospective_pivot_breakout. L'identifiant mss_choch et
ses labels sont conservés pour compatibilité. Ils désignent des tests de
niveaux à chaque barre, pas une séquence CHOCH puis confirmation MSS.
La méthode ne mémorise aucun état de retournement et révise les pivots.
Utiliser uniquement pour annotation rétrospective, avec lookahead explicite.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

import numpy as np
import pandas as pd

from regime_lib.config.parameters import MethodProfile, Parameter, ParameterSchema
from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method
from regime_lib.core.utils import (
    detect_fractals as _detect_fractals,
    build_alternating_pivots as _build_alternating_pivots,
    unaligned_mask,
)


def _classify_mss_choch(
    pivots: list[tuple[int, str, float]],
    close: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Classification MSS/CHOCH barre par barre.

    Algorithme :
    1. Construire un index des pivots par indice de barre.
    2. Pour chaque barre t :
       a. Récupérer les 2 derniers sommets et 2 derniers creux rétrospectifs
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
    Identifiant historique de la classification simplifiée de cassures.

    Méthode OFFLINE : utilise du lookahead structurel (détection de
    fractales). Voir docstring module.
    """

    name = "mss_choch"
    confidence_kind = 'label_constant'
    confidence_description = 'Valeur conventionnelle attachée au label de cassure simplifiée.'
    availability = 'retrospective'
    regime_dimension = 'breakout_event'
    indicator_name = 'retrospective_pivot_breakout'
    regime_description = 'Cassures rétrospectives simplifiées de niveaux de pivots ; sans état de confirmation MSS/CHOCH.'
    REGIME_MAP = {
        "MSS_HAUSSIER": 0,
        "MSS_BAISSIER": 1,
        "CHOCH_HAUSSIER": 2,
        "CHOCH_BAISSIER": 3,
        "NEUTRE": 4,
        "INCONNU": 5,
    }
    requires_lookahead = True

    PARAM_SCHEMA = ParameterSchema(
        parameters={
            "n_fractale": Parameter(int, minimum=1, per_timeframe=True),
        },
    )

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | MethodProfile | None = None,
        n_fractale: int | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            n_fractale=n_fractale,
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

        # 1) Fractales puis pivots alternés
        is_sommet, is_creux = _detect_fractals(high, low, self.n_fractale)
        pivots = _build_alternating_pivots(is_sommet, is_creux, high, low)

        # 2) Classification
        regime = _classify_mss_choch(pivots, close, n)

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
            unaligned = unaligned_mask(out.index, self.timeframe)
            regime[unaligned] = "INCONNU"
            confidence[unaligned] = 0.0

        out["regime"] = regime
        out["confidence"] = confidence
        return out[["regime", "confidence"]]


@register_method
class RetrospectivePivotBreakoutDetector(MSSCHOCHDetector):
    """Tests de cassure de niveaux, sans séquence de confirmation."""
    name = "retrospective_pivot_breakout"
    REGIME_MAP = {"CASSURE_HAUTE_ANCIENNE": 0, "CASSURE_BASSE_ANCIENNE": 1,
                  "CASSURE_HAUTE_RECENTE": 2, "CASSURE_BASSE_RECENTE": 3,
                  "NEUTRE": 4, "INCONNU": 5}

    def fit_predict(self, df):
        result = super().fit_predict(df)
        result["regime"] = result["regime"].replace({
            "MSS_HAUSSIER": "CASSURE_HAUTE_ANCIENNE", "MSS_BAISSIER": "CASSURE_BASSE_ANCIENNE",
            "CHOCH_HAUSSIER": "CASSURE_HAUTE_RECENTE", "CHOCH_BAISSIER": "CASSURE_BASSE_RECENTE"})
        return result
