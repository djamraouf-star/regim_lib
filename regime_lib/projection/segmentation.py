"""
Segmentation du prix par détection de points de changement (PELT).

Principe
--------
On segmente le log-prix en segments où moyenne et variance sont
homogènes. C'est une partition **objective** du temps, dérivée du prix
seul, sans hypothèse sur la nature des régimes.

Algorithme
----------
PELT (Pruned Exact Linear Time, Killick et al. 2012) trouve les
points de changement optimaux pour un coût quadratique et une pénalité
donnée. Complexité O(n) en moyenne.

Paramètres clés
---------------
- `modele` : type de coût
    - 'l2'     : changement de moyenne seul
    - 'normal' : changement de moyenne ET de variance
    - 'rbf'    : noyau RBF (détecte aussi les changements de forme)
- `pen` : pénalité. Plus elle est haute, moins il y a de segments.
    À calibrer par actif / timeframe (typiquement 1–10).
- `min_size` : taille minimale d'un segment, en barres.

Lookahead
---------
PELT utilise toute la série. Méthode offline par construction. Toute
utilisation doit être descriptive (post-mortem) ou via un mécanisme
de lookahead explicite (comme `hmm_gaussian`).

Référence
---------
Killick, Fearnhead, Eckley (2012). "Optimal detection of changepoints
with a linear computational cost". JASA 107(500), 1590-1598.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import ruptures as rpt


ModeleSegmentation = Literal["l2", "normal", "rbf"]


def segmenter_pelt(
    signal: np.ndarray,
    modele: ModeleSegmentation = "normal",
    pen: float = 3.0,
    min_size: int = 20,
) -> np.ndarray:
    """
    Segmente une série 1D par PELT.

    Parameters
    ----------
    signal : np.ndarray
        Signal à segmenter (log-prix).
    modele : {'l2', 'normal', 'rbf'}
        Modèle de coût.
    pen : float
        Pénalité. Plus elle est haute, moins il y a de segments.
    min_size : int
        Taille minimale d'un segment (en barres).

    Returns
    -------
    np.ndarray
        Indices des fins de segment. La convention `ruptures` inclut
        l'index `len(signal)` en dernier élément (fin de série).
    """
    if len(signal) < 2 * min_size:
        raise ValueError(
            f"Série trop courte ({len(signal)} barres) pour "
            f"min_size={min_size}."
        )
    if pen <= 0:
        raise ValueError(f"pen doit être > 0, reçu {pen}.")

    algo = rpt.Pelt(model=modele, min_size=min_size).fit(signal)
    breakpoints = algo.predict(pen=pen)
    return np.asarray(breakpoints, dtype=int)


def segmenter_dataframe(
    df: pd.DataFrame,
    colonne: str = "close",
    modele: ModeleSegmentation = "normal",
    pen: float = 3.0,
    min_size: int = 20,
) -> pd.Series:
    """
    Segmente un DataFrame OHLCV et retourne un Series `segment_id`.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV indexé par DatetimeIndex.
    colonne : str
        Colonne à segmenter (défaut : 'close').
    modele, pen, min_size
        Passés à `segmenter_pelt`.

    Returns
    -------
    pd.Series
        Identifiant de segment par barre (entier croissant, 0-indexé).
        Nom : 'segment_id'.
    """
    if colonne not in df.columns:
        raise ValueError(f"Colonne {colonne!r} absente du DataFrame.")

    valeurs = df[colonne].to_numpy(dtype=float)
    if np.any(valeurs <= 0):
        raise ValueError(
            f"La colonne {colonne!r} contient des valeurs <= 0, "
            f"impossible de calculer le log."
        )

    log_prix = np.log(valeurs)
    breakpoints = segmenter_pelt(
        log_prix, modele=modele, pen=pen, min_size=min_size,
    )

    # Construire segment_id depuis les breakpoints.
    # Convention : breakpoints[-1] == n.
    n = len(df)
    segment_id = np.zeros(n, dtype=int)
    prev = 0
    for seg_id, bp in enumerate(breakpoints):
        segment_id[prev:bp] = seg_id
        prev = bp

    return pd.Series(segment_id, index=df.index, name="segment_id")
