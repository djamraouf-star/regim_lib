"""
Détection de régimes par VPIN (Volume-Synchronized Probability of Informed Trading).

Définition
----------
Échantillonnage par **buckets de volume constant** (et non par temps) :

    Chaque ligne est fractionnée proportionnellement entre des buckets
    de volume bucket_volume. Son ratio |ask_vol − bid_vol| / volume
    est supposé uniforme sur les fractions.
    VPIN_bucket = somme des déséquilibres absolus fractionnés / bucket_volume
    VPIN(t) = moyenne des n_buckets derniers buckets clos à la ligne t.

Interprétation sur forex
------------------------
Le fondement théorique du VPIN repose sur un modèle de market maker
informé. Sur forex décentralisé, avec un volume Dukascopy qui est une
**profondeur observée** et non un volume exécuté, l’interprétation en
termes de « toxicité » est à prendre comme un **proxy exploratoire**
de déséquilibre de carnet, pas comme une mesure de flux informé.

Régimes produits
----------------
- CALME   : VPIN < seuil_bas
- NORMAL  : entre les deux
- TOXIQUE : VPIN > seuil_haut
- INCONNU : chauffe (moins de n_buckets buckets observés) ou données manquantes

Mesure exposée
--------------
- `vpin_value` : VPIN lissé ∈ [0, 1]

Causalité
---------
Les buckets sont remplis dans l'ordre des observations. Seuls les buckets
clos à la clôture de la ligne courante contribuent au score ; le reliquat
incomplet est conservé pour la suite. Aucune valeur n'est réattribuée aux
lignes antérieures. La chauffe dure jusqu'à la clôture de n_buckets buckets.
Une ligne volumineuse peut clore plusieurs buckets et terminer la chauffe.
Les anciens parquets sans métadonnées restent exclus des études causales :
ils peuvent provenir de la version avec réattribution historique (constat C1).

Données requises
----------------
Colonnes : `bid_volume`, `ask_volume` (issues de l’agrégation tick).
Accepte les barres OHLCV enrichies de ces volumes ; le CLI agrège les ticks
avant le calcul au timeframe demandé.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

from collections import deque
import math

import numpy as np
import pandas as pd

from regime_lib.config.parameters import MethodProfile, Parameter, ParameterSchema, Ordered
from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method


from regime_lib.core.temporal import unaligned_mask


def _closed_bucket_values(total, imbalance, bucket_volume, window):
    """Remplit des buckets fixes ; mémoire bornée à la fenêtre, même pour un gros tick."""
    if not np.isfinite(total).all():
        raise ValueError("Le volume total bid + ask doit rester fini.")
    values = np.full(len(total), np.nan)
    closed = deque(maxlen=window)
    rolling_sum = 0.0
    filled = 0.0  # Fraction du bucket courant, entre 0 et 1.
    partial = 0.0  # Déséquilibre fractionné, normalisé par bucket_volume.
    tolerance = 8 * np.finfo(float).eps

    def publish(score):
        nonlocal rolling_sum
        if len(closed) == window:
            rolling_sum -= closed[0]
        closed.append(score)
        rolling_sum += score

    for i, volume in enumerate(total):
        if volume > 0:
            units = float(volume / bucket_volume)
            if not math.isfinite(units):
                raise ValueError("Rapport volume / bucket_volume trop grand.")
            ratio = float(imbalance[i] / volume)
            if filled:
                needed = 1.0 - filled
                if units < needed and needed - units > tolerance:
                    filled += units
                    partial += units * ratio
                    units = 0.0
                else:
                    partial += needed * ratio
                    publish(float(np.clip(partial, 0.0, 1.0)))
                    units = max(0.0, units - needed)
                    filled = partial = 0.0
            # Les buckets entiers contenus dans la même ligne ont le même ratio.
            # Au-delà de la fenêtre, inutile de les matérialiser un à un.
            if units > 0:
                whole = math.floor(units)
                remainder = units - whole
                if 1.0 - remainder <= tolerance:
                    whole += 1
                    remainder = 0.0
                if whole >= window:
                    closed = deque([ratio] * window, maxlen=window)
                    rolling_sum = ratio * window
                else:
                    for _ in range(whole):
                        publish(ratio)
                filled = remainder
                partial = remainder * ratio
        if len(closed) == window:
            values[i] = np.clip(rolling_sum / window, 0.0, 1.0)
    return values


@register_method
class VpinDetector(RegimeDetector):
    """
    Détecteur de régimes par VPIN.

    Paramètres
    ----------
    bucket_volume : float
        Volume cible (bid_vol + ask_vol cumulés) par bucket. Défaut : 10 000.
        À calibrer par actif : sur EURUSD Dukascopy tick, viser des buckets
        de quelques secondes à quelques minutes.
    n_buckets : int
        Nombre de buckets pour la moyenne glissante du VPIN. Défaut : 50.
    seuil_bas : float
        Seuil ∈ (0, 1). En-dessous → CALME. Défaut : 0.20.
    seuil_haut : float
        Seuil ∈ (0, 1). Au-dessus → TOXIQUE. Défaut : 0.40.
        Doit être strictement supérieur à `seuil_bas`.
    """

    name = "vpin"
    availability = 'bar_close'
    legacy_availability = 'retrospective'
    regime_dimension = 'quote_toxicity_proxy'
    regime_description = 'Proxy sur cotations ; buckets fractionnés clos à date.'
    REGIME_MAP = {
        "CALME": 0,
        "NORMAL": 1,
        "TOXIQUE": 2,
        "INCONNU": 3,
    }
    requires_lookahead = False
    REQUIRES_COLUMNS = ["bid_volume", "ask_volume"]

    PARAM_SCHEMA = ParameterSchema(
        parameters={
            "bucket_volume": Parameter(float, minimum=0, exclusive_min=True),
            "n_buckets": Parameter(int, minimum=2),
            "seuil_bas": Parameter(float, minimum=0, maximum=1, exclusive_min=True, exclusive_max=True),
            "seuil_haut": Parameter(float, minimum=0, maximum=1, exclusive_min=True, exclusive_max=True),
        },
        constraints=(Ordered("seuil_bas", "seuil_haut"),),
    )

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | MethodProfile | None = None,
        bucket_volume: float | None = None,
        n_buckets: int | None = None,
        seuil_bas: float | None = None,
        seuil_haut: float | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            bucket_volume=bucket_volume,
            n_buckets=n_buckets,
            seuil_bas=seuil_bas,
            seuil_haut=seuil_haut,
        )

    _unaligned_mask = staticmethod(unaligned_mask)

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
                "vpin_value": pd.Series([], dtype=float),
            })

        bid_vol = out["bid_volume"].to_numpy(dtype=float)
        ask_vol = out["ask_volume"].to_numpy(dtype=float)

        total = bid_vol + ask_vol
        imbalance = np.abs(ask_vol - bid_vol)

        vpin_row = _closed_bucket_values(
            total, imbalance, self.bucket_volume, self.n_buckets,
        )

        # --- Classification ------------------------------------------
        ready = ~np.isnan(vpin_row)
        mask_calme = ready & (vpin_row < self.seuil_bas)
        mask_toxique = ready & (vpin_row > self.seuil_haut)
        mask_normal = (
            ready & ~mask_calme & ~mask_toxique
        )

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_calme] = "CALME"
        regime[mask_toxique] = "TOXIQUE"
        regime[mask_normal] = "NORMAL"

        # --- Confiance : distance au bord du régime ------------------
        confidence = np.full(n, 0.0, dtype=float)
        if mask_calme.any():
            conf = (self.seuil_bas - vpin_row[mask_calme]) / self.seuil_bas
            confidence[mask_calme] = 0.5 + 0.5 * np.clip(conf, 0.0, 1.0)
        if mask_toxique.any():
            conf = (
                vpin_row[mask_toxique] - self.seuil_haut
            ) / (1.0 - self.seuil_haut)
            confidence[mask_toxique] = 0.5 + 0.5 * np.clip(conf, 0.0, 1.0)
        if mask_normal.any():
            confidence[mask_normal] = 0.5

        # --- Barres partielles ---------------------------------------
        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
        else:
            partial = self._unaligned_mask(out.index, self.timeframe)

        if partial.any():
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
            vpin_row[partial] = np.nan

        return pd.DataFrame(
            {
                "regime": regime,
                "confidence": confidence,
                "vpin_value": vpin_row,
            },
            index=out.index,
        )
