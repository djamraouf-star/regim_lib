"""
Détection de régimes par VPIN (Volume-Synchronized Probability of Informed Trading).

Définition
----------
Échantillonnage par **buckets de volume constant** (et non par temps) :

    bucket_id(t)   = floor( Σ_{s≤t} (bid_vol + ask_vol) / bucket_volume )
    VPIN_bucket(b) = Σ_{i∈b} |ask_vol_i − bid_vol_i| / Σ_{i∈b} (bid_vol_i + ask_vol_i)
    VPIN(t)        = moyenne glissante de VPIN_bucket sur n_buckets,
                     décalée d'un bucket (buckets complets uniquement)

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
Une ligne du bucket b ne reçoit que la valeur lissée des buckets complets
≤ b-1 (décalage d'un bucket après la moyenne glissante). Le bucket courant,
encore incomplet, n'est jamais utilisé : il n'y a plus de réattribution
rétrospective de la valeur finale d'un bucket à ses lignes. La valeur sur un
préfixe de la série est identique à celle sur la série complète.
Conséquences : chauffe de n_buckets + 1 buckets, retard d'un bucket de volume.
Les anciens parquets sans métadonnées restent exclus des études causales :
ils peuvent provenir de la version avec réattribution historique (constat C1).

Données requises
----------------
Colonnes : `bid_volume`, `ask_volume` (issues de l’agrégation tick).
Disponible uniquement sur données tick.
"""

from __future__ import annotations

from regime_lib.utils.validation import validated_detector

import numpy as np
import pandas as pd

from regime_lib.config.parameters import MethodProfile, Parameter, ParameterSchema, Ordered
from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import register_method


from regime_lib.core.temporal import unaligned_mask


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
    regime_description = 'Proxy sur cotations ; valeur des seuls buckets antérieurs clos.'
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

        # --- Moyenne glissante sur n_buckets --------------------------
        vpin_serie = pd.Series(vpin_bucket).rolling(
            self.n_buckets, min_periods=self.n_buckets
        ).mean()

        # Causalité : une ligne du bucket b ne voit que les buckets
        # complets <= b-1. Le bucket courant (incomplet) n'est jamais
        # utilisé, donc le dernier bucket n'a plus besoin d'être invalidé.
        vpin_bucket_smooth = vpin_serie.shift(1).to_numpy(dtype=float)

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
