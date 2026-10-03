"""
Détection de régimes par volatilité (ATR + direction).

Régimes produits
----------------
- TENDANCE_HAUSSIERE : ATR >= quantile haut ET direction haussière
- TENDANCE_BAISSIERE : ATR >= quantile haut ET direction baissière
- RANGE               : ATR <= quantile bas
- CHOP                : volatilité intermédiaire
- INCONNU             : phase de chauffe ou barre non clôturée

⚠ Sémantique croisée avec `er_kaufman`
--------------------------------------
Les labels sont partagés avec `er_kaufman`, mais leur sens diffère :

- `atr_volatility` : `RANGE` correspond à une volatilité faible, `CHOP` à une
  volatilité intermédiaire.
- `er_kaufman` : `RANGE` correspond à une efficience intermédiaire, `CHOP` à
  une efficience faible (bruit).

Un `RANGE` ATR n'est pas un `RANGE` ER. Ne pas agréger naïvement les deux
méthodes sans tenir compte de cette distinction.

Résolution des paramètres
-------------------------
Par ordre de priorité décroissante :

    1. Argument explicite au constructeur (ex. n_atr=20)
    2. Sous-profil `profile` (via _resolve, avec recherche par timeframe)
    3. Valeur codée en dur (N_ATR, FENETRE_QUANTILE)

Voir `regime_lib/config/profiles/default.yaml` pour le profil par défaut et
`regime_lib/config/loader.py` pour le chargement des profils.

Causalité et anti-lookahead
---------------------------
1) Les seuils de quantile sont calculés sur l'ATR **décalé d'une barre**
   (`atr.shift(1)`). La barre courante n'entre donc pas dans le calcul de
   son propre seuil.

2) La classification de la barre `t` n'utilise que des informations
   disponibles à sa clôture.

3) Les barres marquées `is_partial=True` sont classées INCONNU avec
   confiance 0.0.

4) En l'absence de la colonne `is_partial`, un masque numpy détecte les
   barres hors grille du timeframe. Ce masque est agnostique à la
   résolution interne du DatetimeIndex ('s', 'ms', 'us', 'ns').
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method
from regime_lib.core.utils import (
    TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS,
    true_range,
    direction,
    unaligned_mask,
)


@register_method
class ATRVolatilityDetector(RegimeDetector):
    """
    Détecteur de régimes par volatilité (ATR) et direction.

    Paramètres dépendant du timeframe
    ---------------------------------
    N_ATR et FENETRE_QUANTILE sont indexés par timeframe. Ces valeurs
    servent de fallback si aucun profil n'est fourni.

    Phase de chauffe (`min_periods`)
    --------------------------------
    Par défaut `min_periods = fenetre // 2`, compromis entre significativité
    statistique du quantile et données utilisables en début de série.

    Confidence
    ----------
    - INCONNU : 0.0
    - CHOP    : 0.5
    - RANGE   : 0.5 + 0.5 * clip((seuil_low  - atr) / seuil_low,  0, 1)
    - TENDANCE_* : 0.5 + 0.5 * clip((atr - seuil_high) / seuil_high, 0, 1)
    """

    name = "atr_volatility"
    REGIME_MAP = {
        "TENDANCE_HAUSSIERE": 0,
        "TENDANCE_BAISSIERE": 1,
        "RANGE": 2,
        "CHOP": 3,
        "INCONNU": 4,
    }
    requires_lookahead = False

    # Fallback : valeurs par défaut par timeframe (utilisées si aucun
    # profil et aucun argument explicite).
    N_ATR: dict[str, int] = {
        "M1": 20, "M5": 20, "M15": 14, "M30": 14,
        "H1": 14, "H4": 14, "D1": 14, "W1": 14,
    }
    FENETRE_QUANTILE: dict[str, int] = {
        "M1": 30_000, "M5": 6_000, "M15": 2_000, "M30": 1_000,
        "H1": 500, "H4": 125, "D1": 21, "W1": 12,
    }

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        n_direction: int | None = None,
        q_low: float | None = None,
        q_high: float | None = None,
        n_atr: int | None = None,
        fenetre: int | None = None,
        min_periods: int | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            n_direction=n_direction,
            q_low=q_low,
            q_high=q_high,
            n_atr=n_atr,
            fenetre=fenetre,
            min_periods=min_periods,
        )

        # --- Résolution en cascade --------------------------------------
        # Priorité : explicite > profil > fallback codé en dur.
        self.n_direction = int(self._resolve(
            "n_direction", n_direction, default=20
        ))
        self.q_low = float(self._resolve(
            "q_low", q_low, default=0.33
        ))
        self.q_high = float(self._resolve(
            "q_high", q_high, default=0.67
        ))
        self.n_atr = int(self._resolve(
            "n_atr", n_atr,
            default=self.N_ATR.get(timeframe or "", 14),
            per_timeframe=True,
        ))
        self.fenetre = int(self._resolve(
            "fenetre", fenetre,
            default=self.FENETRE_QUANTILE.get(timeframe or "", 500),
            per_timeframe=True,
        ))

        # min_periods : cas particulier — la valeur par défaut dépend de
        # fenetre (fenetre // 2). On la résout après fenetre.
        resolved_min = self._resolve(
            "min_periods", min_periods, default=None,
            per_timeframe=True,
        )
        if resolved_min is not None:
            self.min_periods = int(resolved_min)
        else:
            self.min_periods = max(1, self.fenetre // 2)

        # --- Validations ------------------------------------------------
        if self.n_direction < 1:
            raise ValueError("n_direction doit être >= 1.")
        if self.n_atr < 1:
            raise ValueError("n_atr doit être >= 1.")
        if self.fenetre < 2:
            raise ValueError("fenetre doit être >= 2.")
        if not (0.0 < self.q_low < self.q_high < 1.0):
            raise ValueError("Il faut 0 < q_low < q_high < 1.")
        if self.min_periods < 1:
            raise ValueError("min_periods doit être >= 1.")
        if self.min_periods > self.fenetre:
            raise ValueError(
                "min_periods doit être <= fenetre "
                f"(reçu min_periods={self.min_periods}, fenetre={self.fenetre})."
            )

    # ------------------------------------------------------------------
    # fit_predict
    # ------------------------------------------------------------------
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [c.lower() if isinstance(c, str) else c
                       for c in out.columns]

        n = len(out)
        if n == 0:
            # Préserve le DatetimeIndex (nommé "timestamp") pour que
            # to_uniform puisse faire reset_index() sans erreur.
            out["regime"] = pd.Series([], dtype=object)
            out["confidence"] = pd.Series([], dtype=float)
            return out[["regime", "confidence"]]

        high = out["high"].to_numpy(dtype=float)
        low = out["low"].to_numpy(dtype=float)
        close = out["close"].to_numpy(dtype=float)

        # --- 1) ATR causal ---
        tr = true_range(high, low, close)
        atr = pd.Series(tr, index=out.index).rolling(
            self.n_atr, min_periods=1
        ).mean()

        # --- 2) Direction causale ---
        direction_arr = direction(close, self.n_direction)

        # --- 3) Seuils de quantile CAUSAUX ---
        atr_past = atr.shift(1)
        seuil_low = atr_past.rolling(
            self.fenetre, min_periods=self.min_periods
        ).quantile(self.q_low)
        seuil_high = atr_past.rolling(
            self.fenetre, min_periods=self.min_periods
        ).quantile(self.q_high)

        # --- 4) Classification ---
        atr_v = atr.to_numpy()
        s_low = seuil_low.to_numpy()
        s_high = seuil_high.to_numpy()

        ready = ~np.isnan(atr_v) & ~np.isnan(s_low) & ~np.isnan(s_high)
        mask_range = ready & (atr_v <= s_low)
        mask_up = ready & (atr_v >= s_high) & (direction_arr > 0)
        mask_down = ready & (atr_v >= s_high) & (direction_arr < 0)

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_range] = "RANGE"
        regime[mask_up] = "TENDANCE_HAUSSIERE"
        regime[mask_down] = "TENDANCE_BAISSIERE"
        regime[ready & (regime == "INCONNU")] = "CHOP"

        # --- 5) Confidence ---
        with np.errstate(divide="ignore", invalid="ignore"):
            dist_low = np.where(
                s_low > 0, (s_low - atr_v) / s_low, 0.0
            )
            dist_high = np.where(
                s_high > 0, (atr_v - s_high) / s_high, 0.0
            )
        dist_low = np.clip(dist_low, 0.0, 1.0)
        dist_high = np.clip(dist_high, 0.0, 1.0)

        confidence = np.full(n, 0.0, dtype=float)
        confidence[regime == "CHOP"] = 0.5
        confidence[mask_range] = 0.5 + 0.5 * dist_low[mask_range]
        confidence[mask_up] = 0.5 + 0.5 * dist_high[mask_up]
        confidence[mask_down] = 0.5 + 0.5 * dist_high[mask_down]

        # --- 6) Bougies non clôturées ---
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
