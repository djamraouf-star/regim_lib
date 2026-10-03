"""
Utilitaires partagés entre les détecteurs de régimes.

Centralise les fonctions statistiques et les helpers temporels dupliqués
dans les modules `regime_lib.methods.*`. Toute nouvelle méthode doit
utiliser ces helpers plutôt que de réimplémenter localement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# Grille attendue (secondes) par timeframe.
# Servait de repli quand `is_partial` n'est pas fourni dans les données
# d'entrée : une barre dont le timestamp n'est pas un multiple entier de
# `TIMEFRAME_SECONDS[tf]` depuis l'epoch UTC est considérée tronquée.
TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D1": 86400,
    "W1": 604800,
}


def true_range(
    high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    """
    True Range vectorisé.

    Pour t = 0, on utilise close[0] comme close précédent, ce qui donne
    TR[0] = high[0] - low[0].
    """
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr_hl = high - low
    tr_hc = np.abs(high - prev_close)
    tr_lc = np.abs(low - prev_close)
    return np.fmax(np.fmax(tr_hl, tr_hc), tr_lc)


def direction(close: np.ndarray, n: int) -> np.ndarray:
    """
    Signe du rendement en arrière de n barres :
    sign(close[t] - close[t-n]). Retourne NaN pour les n premières barres.
    """
    out = np.full(close.shape, np.nan, dtype=float)
    if n < len(close):
        out[n:] = np.sign(close[n:] - close[:-n])
    return out


def unaligned_mask(
    index: pd.DatetimeIndex, timeframe: str | None
) -> np.ndarray:
    """
    Masque numpy : True si le timestamp n'est pas aligné sur la grille
    du timeframe (multiple du pas en secondes depuis l'epoch UTC).

    Sert de repli quand `is_partial` n'est pas fourni : une barre hors
    grille est très probablement une barre tronquée.

    La résolution native du DatetimeIndex peut être 's', 'ms', 'us' ou
    'ns' selon la façon dont il a été construit. On ne suppose PAS de
    résolution fixe : la conversion est effectuée par une soustraction
    de Timedelta, indépendante de la résolution interne.
    """
    step = TIMEFRAME_SECONDS.get(timeframe or "")
    if step is None or len(index) == 0:
        return np.zeros(len(index), dtype=bool)
    epoch = pd.Timestamp("1970-01-01", tz="UTC") if index.tz is not None \
        else pd.Timestamp("1970-01-01")
    ts_s = np.asarray(
        (index - epoch) // pd.Timedelta(seconds=1), dtype=np.int64
    )
    return (ts_s % step) != 0


def detect_fractals(
    high: np.ndarray, low: np.ndarray, n_fractale: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Détection de fractales strictes, vectorisée en numpy pur.

    Retourne deux tableaux booléens `(is_sommet, is_creux)` de même
    forme que `high`. Une position `i` est marquée True s'il s'agit
    d'un sommet (resp. creux) local sur une fenêtre de
    `2 * n_fractale + 1` barres centrée en `i`.

    Les bordures (positions ne pouvant pas former une fenêtre complète)
    sont systématiquement False, contrairement à
    `scipy.signal.argrelextrema` qui gère les bords en mode 'clip'.
    """
    n = len(high)
    window = 2 * n_fractale + 1
    is_sommet = np.zeros(n, dtype=bool)
    is_creux = np.zeros(n, dtype=bool)
    if n < window:
        return is_sommet, is_creux

    h_win = sliding_window_view(high, window)
    l_win = sliding_window_view(low, window)

    center = n_fractale
    mask = np.ones(window, dtype=bool)
    mask[center] = False

    is_sommet_win = np.all(h_win[:, center, None] > h_win[:, mask], axis=1)
    is_creux_win = np.all(l_win[:, center, None] < l_win[:, mask], axis=1)

    is_sommet[n_fractale:n - n_fractale] = is_sommet_win
    is_creux[n_fractale:n - n_fractale] = is_creux_win
    return is_sommet, is_creux


def build_alternating_pivots(
    is_sommet: np.ndarray,
    is_creux: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> list[tuple[int, str, float]]:
    """
    Construit la liste chronologique des pivots strictement alternés
    sommet / creux (abrégés 'S' / 'C').

    Deux fractales successives de même nature (deux sommets par exemple)
    sont fusionnées : on ne garde que le plus extrême (le plus haut pour
    un sommet, le plus bas pour un creux).

    Chaque élément de la liste retournée est un tuple
    ``(indice_barre, type, valeur)`` où ``type`` vaut ``'S'`` ou ``'C'``.
    """
    n = len(high)
    pivots: list[tuple[int, str, float]] = []
    for i in range(n):
        if is_sommet[i]:
            if not pivots or pivots[-1][1] == 'C':
                pivots.append((i, 'S', float(high[i])))
            elif high[i] > pivots[-1][2]:
                pivots[-1] = (i, 'S', float(high[i]))
        if is_creux[i]:
            if not pivots or pivots[-1][1] == 'S':
                pivots.append((i, 'C', float(low[i])))
            elif low[i] < pivots[-1][2]:
                pivots[-1] = (i, 'C', float(low[i]))
    return pivots
