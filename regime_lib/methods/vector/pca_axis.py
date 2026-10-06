"""
Détection de régimes par axe principal (PCA sur fenêtre glissante).

Principe
--------
Pour chaque fenêtre passée, on applique la PCA aux points `(temps,
log(close))` standardisés. PC1 décrit l'axe de variance maximale de la
trajectoire du prix. Son signe est orienté pour que sa composante temps
soit positive ; la composante prix indique alors l'orientation haussière
ou baissière.

`pca_direction` est la corrélation signée entre le temps et le log-prix,
équivalente à la différence normalisée des deux valeurs propres avec le
signe de l'axe. Contrairement à la variance seule, elle distingue donc
l'orientation de la trajectoire. Les deux ratios exposés sont égaux pour
cette PCA à deux dimensions.

Régimes produits
----------------
- HAUSSIER : corrélation temps / log-prix positive et axe concentré
- BAISSIER : corrélation temps / log-prix négative et axe concentré
- NEUTRE   : corrélation ambiguë ou axe peu concentré
- INCONNU  : phase de chauffe ou barre non clôturée

Mesures exposées
----------------
- `pca_direction`      : corrélation temps / log-prix signée ∈ [-1, 1]
- `pca_eigen_ratio`    : λ1 / Σλ ∈ [0, 1] (concentration de l'axe)
- `pca_variance_ratio` : λ1 / (λ1 + λ2) ∈ [0, 1]

Causalité
---------
La fenêtre contient uniquement les observations antérieures à t. Aucun
lookahead.
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


def _construire_features(df: pd.DataFrame) -> np.ndarray:
    """
    Construit les points de trajectoire (n, 2) : [temps, log(close)].

    Le temps est exprimé en rang d'observation ; chaque fenêtre est ensuite
    centrée et standardisée, ce qui rend la PCA indépendante des unités.
    """
    close = pd.to_numeric(df["close"], errors="raise").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(close)
    time = np.arange(len(close), dtype=float)
    return np.column_stack([time, log_close])


def _pca_windowed(
    X: np.ndarray, fenetre: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule pour chaque t :
      - l'orientation signée de l'axe principal temps / log-prix
      - λ1 / Σλ
      - λ1 / (λ1 + λ2)

    Retourne (direction, eigen_ratio, var_ratio), NaN sur la chauffe.
    """
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X doit contenir les colonnes temps et log-prix.")
    n = X.shape[0]
    direction = np.full(n, np.nan, dtype=float)
    eigen_ratio = np.full(n, np.nan, dtype=float)
    var_ratio = np.full(n, np.nan, dtype=float)

    if n < fenetre + 1:
        return direction, eigen_ratio, var_ratio

    for t in range(fenetre, n):
        window = X[t - fenetre : t]  # strictement passé
        # Ignorer les lignes avec NaN
        valid = np.all(np.isfinite(window), axis=1)
        window = window[valid]
        if len(window) < fenetre // 2:
            continue

        mu = window.mean(axis=0)
        sigma = window.std(axis=0)
        # Éviter division par zéro
        sigma_safe = np.where(sigma > 0, sigma, 1.0)
        window_std = (window - mu) / sigma_safe

        # Matrice de covariance (2 × 2), sur temps et log-prix.
        cov = np.cov(window_std, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # eigh retourne les valeurs propres en ordre croissant.
        lam = np.maximum(eigvals[::-1], 0.0)
        v1 = eigvecs[:, -1]

        total = lam.sum()
        if total <= 0:
            continue

        # Le signe d'un vecteur propre étant arbitraire, on fixe son
        # orientation dans le sens du temps avant de lire le sens du prix.
        if v1[0] < 0:
            v1 = -v1
        orientation = float(np.sign(v1[1]))
        concentration = float((lam[0] - lam[1]) / total)
        direction[t] = orientation * concentration
        eigen_ratio[t] = lam[0] / total
        if lam[0] + lam[1] > 0:
            var_ratio[t] = lam[0] / (lam[0] + lam[1])

    return direction, eigen_ratio, var_ratio


@register_method
class PCAAxisDetector(RegimeDetector):
    """Détecteur de régimes par axe principal (PCA glissant)."""

    name = "pca_axis"
    availability = 'bar_close'
    regime_dimension = 'trajectory_geometry'
    regime_description = 'Géométrie de la fenêtre de prix par axes principaux.'
    REGIME_MAP = {
        "HAUSSIER": 0,
        "BAISSIER": 1,
        "NEUTRE": 2,
        "INCONNU": 3,
    }
    requires_lookahead = False

    PARAM_SCHEMA = ParameterSchema(
        parameters={
            "fenetre": Parameter(int, minimum=20, per_timeframe=True),
            "seuil_direction": Parameter(float, minimum=0, maximum=1, exclusive_min=True, exclusive_max=True),
            "seuil_eigen_ratio": Parameter(float, minimum=0, maximum=1, exclusive_min=True, exclusive_max=True),
        },
    )

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | MethodProfile | None = None,
        fenetre: int | None = None,
        seuil_direction: float | None = None,
        seuil_eigen_ratio: float | None = None,
    ) -> None:
        super().__init__(
            allow_lookahead=allow_lookahead,
            timeframe=timeframe,
            profile=profile,
            fenetre=fenetre,
            seuil_direction=seuil_direction,
            seuil_eigen_ratio=seuil_eigen_ratio,
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
            out["pca_direction"] = pd.Series([], dtype=float)
            out["pca_eigen_ratio"] = pd.Series([], dtype=float)
            out["pca_variance_ratio"] = pd.Series([], dtype=float)
            return out[[
                "regime", "confidence",
                "pca_direction", "pca_eigen_ratio", "pca_variance_ratio",
            ]]

        X = _construire_features(out)
        direction, eigen_ratio, var_ratio = _pca_windowed(X, self.fenetre)

        ready = (
            ~np.isnan(direction)
            & ~np.isnan(eigen_ratio)
            & (eigen_ratio > self.seuil_eigen_ratio)
        )

        mask_haussier = ready & (direction > self.seuil_direction)
        mask_baissier = ready & (direction < -self.seuil_direction)
        mask_neutre = (
            ~np.isnan(direction)
            & ~mask_haussier
            & ~mask_baissier
        )

        regime = np.full(n, "INCONNU", dtype=object)
        regime[mask_neutre] = "NEUTRE"
        regime[mask_haussier] = "HAUSSIER"
        regime[mask_baissier] = "BAISSIER"

        confidence = np.full(n, 0.0, dtype=float)
        if mask_haussier.any():
            conf_h = (direction[mask_haussier] - self.seuil_direction) / (
                1.0 - self.seuil_direction
            )
            confidence[mask_haussier] = 0.5 + 0.5 * np.clip(
                conf_h, 0.0, 1.0
            )
        if mask_baissier.any():
            conf_b = (-direction[mask_baissier] - self.seuil_direction) / (
                1.0 - self.seuil_direction
            )
            confidence[mask_baissier] = 0.5 + 0.5 * np.clip(
                conf_b, 0.0, 1.0
            )
        if mask_neutre.any():
            confidence[mask_neutre] = 0.5

        if "is_partial" in out.columns:
            partial = out["is_partial"].to_numpy(dtype=bool)
            regime[partial] = "INCONNU"
            confidence[partial] = 0.0
            direction[partial] = np.nan
            eigen_ratio[partial] = np.nan
            var_ratio[partial] = np.nan
        else:
            unaligned = unaligned_mask(out.index, self.timeframe)
            regime[unaligned] = "INCONNU"
            confidence[unaligned] = 0.0
            direction[unaligned] = np.nan
            eigen_ratio[unaligned] = np.nan
            var_ratio[unaligned] = np.nan

        out["regime"] = regime
        out["confidence"] = confidence
        out["pca_direction"] = direction
        out["pca_eigen_ratio"] = eigen_ratio
        out["pca_variance_ratio"] = var_ratio
        return out[[
            "regime", "confidence",
            "pca_direction", "pca_eigen_ratio", "pca_variance_ratio",
        ]]
