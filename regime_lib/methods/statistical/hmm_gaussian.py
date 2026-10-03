"""
Détection de régimes par HMM gaussien (hmmlearn).

⚠ MÉTHODE OFFLINE — LOOKAHEAD STRUCTUREL
----------------------------------------
Cette méthode utilise du lookahead à deux niveaux :
  1) Normalisation z-score sur toute la série ;
  2) Fit HMM sur toute la série.

Il n'existe pas de version causale qui préserve l'algorithme.

États produits
--------------
Les états sont retournés bruts : STATE_0, STATE_1, ..., STATE_{n-1}.
Aucun label sémantique n'est imposé.

Résolution des paramètres
-------------------------
Cascade : explicite > profile > fallback codé en dur.
`n_states` est résolu AVANT super().__init__() car il détermine
la construction dynamique de REGIME_MAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method
from regime_lib.core.utils import (
    TIMEFRAME_SECONDS as _TIMEFRAME_SECONDS,
    true_range,
    unaligned_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _construire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features d'entrée du HMM :
      - log_return     : log(close[t] / close[t-1])
      - vol_realisee   : écart-type glissant de log_return sur 20 barres
      - range_norm     : (high - low) / ATR(14)
      - close_position : (close - low) / (high - low)
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    log_return = np.log(close / close.shift(1))
    vol_realisee = log_return.rolling(20).std()

    tr_arr = true_range(
        high.to_numpy(dtype=float),
        low.to_numpy(dtype=float),
        close.to_numpy(dtype=float),
    )
    tr = pd.Series(tr_arr, index=df.index)
    atr = tr.rolling(14).mean()

    range_norm = (high - low) / atr.replace(0, np.nan)
    close_position = (
        ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)
    )

    features = pd.DataFrame(
        {
            "log_return": log_return,
            "vol_realisee": vol_realisee,
            "range_norm": range_norm,
            "close_position": close_position,
        },
        index=df.index,
    )
    return features.dropna()


def _normaliser(features: pd.DataFrame) -> pd.DataFrame:
    """Z-score sur toute la série. ⚠ lookahead."""
    std = features.std().replace(0, np.nan)
    return ((features - features.mean()) / std).dropna()


def _entrainer_hmm(
    xdf: pd.DataFrame, n_states: int, n_iter: int, seed: int,
):
    """Entraîne un GaussianHMM et retourne (model, states, posteriors)."""
    from hmmlearn.hmm import GaussianHMM

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
    )
    model.fit(xdf.values)
    states = model.predict(xdf.values)
    posteriors = model.predict_proba(xdf.values)
    return model, states, posteriors


def _confidence_from_posteriors(
    posteriors: np.ndarray, n_states: int
) -> np.ndarray:
    """
    Confidence à partir des probabilités a posteriori.
    Formule : 0.5 + 0.5 * (max_post - 1/n) / (1 - 1/n).
    """
    if n_states <= 1:
        return np.ones(len(posteriors), dtype=float)
    max_post = posteriors.max(axis=1)
    baseline = 1.0 / n_states
    ratio = (max_post - baseline) / (1.0 - baseline)
    return 0.5 + 0.5 * np.clip(ratio, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Détecteur
# ---------------------------------------------------------------------------
@register_method
class HMMGaussianDetector(RegimeDetector):
    """
    Détecteur de régimes par HMM gaussien (hmmlearn).

    Méthode OFFLINE : utilise du lookahead structurel.
    """

    name = "hmm_gaussian"
    REGIME_MAP: dict[str, int] = {}
    requires_lookahead = True

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        n_states: int | None = None,
        n_iter: int | None = None,
        seed: int | None = None,
    ) -> None:
        # --- Résolution AVANT super() car REGIME_MAP en dépend --------
        profile_dict = dict(profile) if profile else {}

        def _resolve_pre(key, explicit, default):
            if explicit is not None:
                return explicit
            if key in profile_dict:
                return profile_dict[key]
            return default

        # --- n_states : int strict >= 1 (bool exclu) -------------------
        raw_n_states = _resolve_pre("n_states", n_states, 2)
        if isinstance(raw_n_states, bool) or not isinstance(raw_n_states, int):
            raise ValueError(
                f"n_states doit être un entier >= 1, "
                f"reçu {raw_n_states!r} (type {type(raw_n_states).__name__})."
            )
        if raw_n_states < 1:
            raise ValueError(
                f"n_states doit être >= 1, reçu {raw_n_states}."
            )
        resolved_n_states = raw_n_states

        # --- n_iter : int strict >= 1 (bool exclu) ---------------------
        raw_n_iter = _resolve_pre("n_iter", n_iter, 100)
        if isinstance(raw_n_iter, bool) or not isinstance(raw_n_iter, int):
            raise ValueError(
                f"n_iter doit être un entier >= 1, "
                f"reçu {raw_n_iter!r} (type {type(raw_n_iter).__name__})."
            )
        if raw_n_iter < 1:
            raise ValueError(
                f"n_iter doit être >= 1, reçu {raw_n_iter}."
            )
        resolved_n_iter = raw_n_iter

        # --- seed : int strict (peut être négatif) ---------------------
        raw_seed = _resolve_pre("seed", seed, 42)
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ValueError(
                f"seed doit être un entier, "
                f"reçu {raw_seed!r} (type {type(raw_seed).__name__})."
            )
        resolved_seed = raw_seed

        # --- Construction dynamique du REGIME_MAP ----------------------
        self.REGIME_MAP = {
            f"STATE_{i}": i for i in range(resolved_n_states)
        }
        self.REGIME_MAP["INCONNU"] = resolved_n_states

        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            n_states=resolved_n_states,
            n_iter=resolved_n_iter,
            seed=resolved_seed,
        )
        self.n_states = resolved_n_states
        self.n_iter = resolved_n_iter
        self.seed = resolved_seed

    @staticmethod
    def _unaligned_mask(
        index: pd.DatetimeIndex, timeframe: str | None
    ) -> np.ndarray:
        """Delegue vers regime_lib.core.utils.unaligned_mask."""
        return unaligned_mask(index, timeframe)

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.columns = [
            c.lower() if isinstance(c, str) else c for c in out.columns
        ]

        n = len(out)
        if n == 0:
            out["regime"] = pd.Series([], dtype=object)
            out["confidence"] = pd.Series([], dtype=float)
            return out[["regime", "confidence"]]

        features = _construire_features(out)
        if len(features) == 0:
            out["regime"] = "INCONNU"
            out["confidence"] = 0.0
            return out[["regime", "confidence"]]

        xdf = _normaliser(features)
        if len(xdf) == 0:
            out["regime"] = "INCONNU"
            out["confidence"] = 0.0
            return out[["regime", "confidence"]]

        model, states, posteriors = _entrainer_hmm(
            xdf, self.n_states, self.n_iter, self.seed
        )

        labels = np.array([f"STATE_{s}" for s in states], dtype=object)
        confidence = _confidence_from_posteriors(posteriors, self.n_states)

        regime_series = pd.Series(labels, index=xdf.index)
        confidence_series = pd.Series(confidence, index=xdf.index)

        out["regime"] = regime_series.reindex(out.index).fillna("INCONNU")
        out["confidence"] = (
            confidence_series.reindex(out.index).fillna(0.0)
        )

        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
            out.loc[partial, "regime"] = "INCONNU"
            out.loc[partial, "confidence"] = 0.0
        else:
            unaligned = unaligned_mask(out.index, self.timeframe)
            out.loc[unaligned, "regime"] = "INCONNU"
            out.loc[unaligned, "confidence"] = 0.0

        return out[["regime", "confidence"]]
