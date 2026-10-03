"""
Helpers temporels partagés par les modules du package `context`.

Factorise les calculs communs :
  - durée d'un timeframe (dupliquée depuis `core.data_loader` pour
    éviter un import circulaire) ;
  - conversion "HH:MM" → minutes depuis minuit ;
  - masque d'appartenance à un intervalle, avec gestion du
    chevauchement minuit ;
  - calcul du début de barre (convention étiquette à droite).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Durées par timeframe (clés identiques à TIMEFRAME_RULES).
DUREES: dict[str, pd.Timedelta] = {
    "M1": pd.Timedelta("1min"),
    "M5": pd.Timedelta("5min"),
    "M15": pd.Timedelta("15min"),
    "M30": pd.Timedelta("30min"),
    "H1": pd.Timedelta("1h"),
    "H4": pd.Timedelta("4h"),
    "D1": pd.Timedelta("1D"),
    "W1": pd.Timedelta("1W"),
}


def verifier_timeframe(timeframe: str) -> pd.Timedelta:
    """Retourne la durée associée au timeframe, ou lève une ValueError."""
    if timeframe not in DUREES:
        raise ValueError(
            f"Timeframe inconnu : {timeframe!r}. "
            f"Valides : {sorted(DUREES)}."
        )
    return DUREES[timeframe]


def verifier_index_utc(index: pd.DatetimeIndex) -> None:
    """Vérifie que l'index est tz-aware. Lève une ValueError sinon."""
    if index.tz is None:
        raise ValueError(
            "L'index doit être tz-aware (UTC). Utiliser "
            "data_loader.load_parquet qui force UTC."
        )


def minutes_depuis_minuit(hhmm: str) -> int:
    """Convertit 'HH:MM' en minutes depuis minuit."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def masque_intervalle(
    minutes: np.ndarray, debut: int, fin: int
) -> np.ndarray:
    """
    Masque True si minutes ∈ [debut, fin), avec chevauchement minuit.

    Si debut > fin, l'intervalle couvre [debut, 1440) ∪ [0, fin).
    """
    if debut <= fin:
        return (minutes >= debut) & (minutes < fin)
    return (minutes >= debut) | (minutes < fin)


def debut_de_barre(
    index: pd.DatetimeIndex, timeframe: str
) -> pd.DatetimeIndex:
    """
    Retourne l'index des débuts de barre (étiquette - durée).

    Convention du projet : barres étiquetées à droite
    (`label='right'`, `closed='left'`). Une barre étiquetée T couvre
    `[T - durée, T)`.
    """
    duree = verifier_timeframe(timeframe)
    return index - duree


def heures_minutes(
    index_utc: pd.DatetimeIndex, tz: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convertit un index UTC en heure locale `tz`.

    Retourne (heures, minutes, total_minutes_depuis_minuit).
    """
    local = index_utc.tz_convert(tz)
    h = np.asarray(local.hour, dtype=int)
    m = np.asarray(local.minute, dtype=int)
    return h, m, h * 60 + m

