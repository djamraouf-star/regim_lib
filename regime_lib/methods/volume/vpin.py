"""
Détection de régimes par VPIN (Volume-Synchronized Probability of Informed Trading).

Définition
----------
Échantillonnage par **buckets de volume constant** (et non par temps) :

    bucket_id(t)   = floor( Σ_{s≤t} (bid_vol + ask_vol) / bucket_volume )
    VPIN_bucket(b) = Σ_{i∈b} |ask_vol_i − bid_vol_i| / Σ_{i∈b} (bid_vol_i + ask_vol_i)
    VPIN(t)        = moyenne glissante de VPIN_bucket sur n_buckets

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
Buckets et moyenne glissante strictement passés. Aucun lookahead.
Le dernier bucket (potentiellement partiel) est invalidé.

Données requises
----------------
Colonnes : `bid_volume`, `ask_volume` (issues de l’agrégation tick).
Disponible uniquement sur données tick.
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
    REGIME_MAP = {
        "CALME": 0,
        "NORMAL": 1,
        "TOXIQUE": 2,
        "INCONNU": 3,
    }
    requires_lookahead = False
    REQUIRES_COLUMNS = ["bid_volume", "ask_volume"]

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
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
        self.bucket_volume = float(self._resolve(
            "bucket_volume", bucket_volume, default=10_000.0,
        ))
        self.n_buckets = int(self._resolve(
            "n_buckets", n_buckets, default=50,
        ))
        self.seuil_bas = float(self._resolve(
            "seuil_bas", seuil_bas, default=0.20,
        ))
        self.seuil_haut = float(self._resolve(
            "seuil_haut", seuil_haut, default=0.40,
        ))

        if self.bucket_volume <= 0:
            raise ValueError("bucket_volume doit être > 0.")
        if self.n_buckets < 2:
            raise ValueError("n_buckets doit être >= 2.")
        if not (0.0 < self.seuil_bas < 1.0):
            raise ValueError("seuil_bas doit être dans (0, 1).")
        if not (0.0 < self.seuil_haut < 1.0):
            raise ValueError("seuil_haut doit être dans (0, 1).")
        if self.seuil_haut <= self.seuil_bas:
            raise ValueError("seuil_haut doit être > seuil_bas.")

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

        # --- Bucketisation volume-clock -------------------------------
        # bucket_id(t) = floor(cum_total(t) / bucket_volume)
        cum_total = np.cumsum(total)
        bucket_id = np.floor(
            cum_total / self.bucket_volume
        ).astype(np.int64)

        n_buckets_total = int(bucket_id[-1]) + 1

        # Agrégation des imbalances par bucket
        sum_imb = np.zeros(n_buckets_total, dtype=float)
        sum_tot = np.zeros(n_buckets_total, dtype=float)
        np.add.at(sum_imb, bucket_id, imbalance)
        np.add.at(sum_tot, bucket_id, total)

        with np.errstate(divide="ignore", invalid="ignore"):
            vpin_bucket = np.where(
                sum_tot > 0, sum_imb / sum_tot, np.nan
            )

        # Le dernier bucket est presque sûrement partiel : on l'invalide
        # pour éviter un VPIN calculé sur moins de `bucket_volume`.
        if n_buckets_total > 0:
            vpin_bucket[-1] = np.nan

        # --- Moyenne glissante sur n_buckets --------------------------
        vpin_serie = pd.Series(vpin_bucket).rolling(
            self.n_buckets, min_periods=self.n_buckets
        ).mean()
        vpin_bucket_smooth = vpin_serie.to_numpy(dtype=float)

        # --- Re-mapping bucket → ligne -------------------------------
        vpin_row = vpin_bucket_smooth[bucket_id]
        vpin_row = np.array(vpin_row, dtype=float, copy=True)

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
