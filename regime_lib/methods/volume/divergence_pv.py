"""
Détection de régimes par divergence prix / volume.

Principe
--------
Sur une fenêtre glissante de N barres :

- Nouveau plus haut (close = max de la fenêtre) :
    - volume fort (> médiane glissante) → CONFIRMATION_HAUSSIERE
    - volume faible → DIVERGENCE_HAUSSIERE (signal de fragilité)

- Nouveau plus bas (close = min de la fenêtre) :
    - volume fort → CONFIRMATION_BAISSIERE
    - volume faible → DIVERGENCE_BAISSIERE

- Ni nouveau haut ni nouveau bas → NEUTRE

⚠ La divergence prix/volume est un signal **populaire** chez les
traders discrétionnaires, mais sa validité statistique est **moins
établie** que d'autres concepts de microstructure. À traiter comme
une hypothèse à tester, pas comme une règle établie.

Régimes produits
----------------
- CONFIRMATION_HAUSSIERE : nouveau haut + volume fort
- CONFIRMATION_BAISSIERE : nouveau bas + volume fort
- DIVERGENCE_HAUSSIERE   : nouveau haut + volume faible
- DIVERGENCE_BAISSIERE   : nouveau bas + volume faible
- NEUTRE                 : pas de nouveau extrême
- INCONNU                : chauffe ou données manquantes

Mesures exposées
----------------
- `volume_rel` : volume / médiane glissante du volume
- `is_new_high` : booléen
- `is_new_low`  : booléen

Causalité
---------
Fenêtre strictement passée. Aucun lookahead.

Données requises
----------------
Colonnes : `close`, `volume`.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

import numpy as np
import pandas as pd

from regime_lib.config.parameters import MethodProfile, Parameter, ParameterSchema
from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method
from regime_lib.core.utils import (
    TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS,
    unaligned_mask,
)


@register_method
class DivergencePVDetector(RegimeDetector):
    """
    Détecteur de régimes par divergence prix / volume.

    Paramètres
    ----------
    fenetre_extreme : int
        Fenêtre pour déterminer « nouveau haut / nouveau bas ».
    fenetre_median_vol : int
        Fenêtre pour la médiane glissante du volume.
    ratio_volume_fort : float
        Ratio volume / médiane au-dessus duquel le volume est
        considéré « fort ». Défaut : 1.5.
    """

    name = "divergence_pv"
    availability = 'bar_close'
    regime_dimension = 'price_volume_divergence'
    regime_description = 'Relation entre variations de prix et volumes.'
    REGIME_MAP = {
        "CONFIRMATION_HAUSSIERE": 0,
        "CONFIRMATION_BAISSIERE": 1,
        "DIVERGENCE_HAUSSIERE": 2,
        "DIVERGENCE_BAISSIERE": 3,
        "NEUTRE": 4,
        "INCONNU": 5,
    }
    requires_lookahead = False
    REQUIRES_COLUMNS = ["close", "volume"]

    PARAM_SCHEMA = ParameterSchema(
        parameters={
            "fenetre_extreme": Parameter(int, minimum=2, per_timeframe=True),
            "fenetre_median_vol": Parameter(int, minimum=2, per_timeframe=True),
            "ratio_volume_fort": Parameter(float, minimum=0, exclusive_min=True),
        },
    )

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | MethodProfile | None = None,
        fenetre_extreme: int | None = None,
        fenetre_median_vol: int | None = None,
        ratio_volume_fort: float | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            fenetre_extreme=fenetre_extreme,
            fenetre_median_vol=fenetre_median_vol,
            ratio_volume_fort=ratio_volume_fort,
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
        self.verifier_colonnes(out)

        n = len(out)
        if n == 0:
            out["regime"] = pd.Series([], dtype=object)
            out["confidence"] = pd.Series([], dtype=float)
            out["volume_rel"] = pd.Series([], dtype=float)
            out["is_new_high"] = pd.Series([], dtype=bool)
            out["is_new_low"] = pd.Series([], dtype=bool)
            return out[[
                "regime", "confidence", "volume_rel",
                "is_new_high", "is_new_low",
            ]]

        close = out["close"]
        volume = out["volume"]

        # Médiane glissante du volume (strictement passée)
        med_vol = volume.rolling(
            self.fenetre_median_vol,
            min_periods=max(1, self.fenetre_median_vol // 2),
        ).median().shift(1)

        with np.errstate(divide="ignore", invalid="ignore"):
            volume_rel_arr = np.where(
                med_vol > 0, volume / med_vol, np.nan
            )
        # Copie explicite : `.to_numpy()` peut être read-only.
        volume_rel_arr = np.array(volume_rel_arr, dtype=float, copy=True)

        # Nouveaux extrêmes (strictement passés, on exclut la barre t)
        max_passe = close.rolling(
            self.fenetre_extreme,
            min_periods=max(1, self.fenetre_extreme // 2),
        ).max().shift(1)
        min_passe = close.rolling(
            self.fenetre_extreme,
            min_periods=max(1, self.fenetre_extreme // 2),
        ).min().shift(1)

        # Copie explicite (arrays booléens issus de comparaisons pandas).
        is_new_high = np.array(
            (close > max_passe).to_numpy(), dtype=bool, copy=True
        )
        is_new_low = np.array(
            (close < min_passe).to_numpy(), dtype=bool, copy=True
        )

        # Masque ready
        ready = (
            ~np.isnan(volume_rel_arr)
            & ~np.isnan(max_passe.to_numpy())
            & ~np.isnan(min_passe.to_numpy())
        )

        volume_fort = volume_rel_arr > self.ratio_volume_fort
        volume_faible = ~volume_fort

        mask_conf_h = ready & is_new_high & volume_fort
        mask_conf_b = ready & is_new_low & volume_fort
        mask_div_h = ready & is_new_high & volume_faible
        mask_div_b = ready & is_new_low & volume_faible
        mask_neutre = ready & ~is_new_high & ~is_new_low

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_conf_h] = "CONFIRMATION_HAUSSIERE"
        regime[mask_conf_b] = "CONFIRMATION_BAISSIERE"
        regime[mask_div_h] = "DIVERGENCE_HAUSSIERE"
        regime[mask_div_b] = "DIVERGENCE_BAISSIERE"
        regime[mask_neutre] = "NEUTRE"

        # Confidence
        confidence = np.full(n, 0.0, dtype=float)
        confidence[mask_neutre] = 0.5
        confidence[mask_conf_h] = 0.6
        confidence[mask_conf_b] = 0.6
        confidence[mask_div_h] = 0.7
        confidence[mask_div_b] = 0.7

        # Bougies non clôturées
        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
        else:
            partial = unaligned_mask(out.index, self.timeframe)

        if partial.any():
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
            volume_rel_arr[partial] = np.nan
            is_new_high[partial] = False
            is_new_low[partial] = False

        out["regime"] = regime
        out["confidence"] = confidence
        out["volume_rel"] = volume_rel_arr
        out["is_new_high"] = is_new_high
        out["is_new_low"] = is_new_low
        return out[[
            "regime", "confidence", "volume_rel",
            "is_new_high", "is_new_low",
        ]]
