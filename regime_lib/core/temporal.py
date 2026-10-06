"""Grilles de clôture communes au chargement, aux détecteurs et aux études."""
from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME_SECONDS = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400, "W1": 604800,
}
# Semaine UTC : lundi 00:00 à lundi 00:00, même origine pour tous les usages.
TIMEFRAME_RULES = {k: f"{v}s" for k, v in TIMEFRAME_SECONDS.items()}


def grid_origin(index: pd.DatetimeIndex, timeframe: str) -> pd.Timestamp:
    return pd.Timestamp("1970-01-05" if timeframe == "W1" else "1970-01-01",
                        tz=index.tz)


def unaligned_mask(index: pd.DatetimeIndex, timeframe: str | None) -> np.ndarray:
    if timeframe not in TIMEFRAME_SECONDS or len(index) == 0:
        return np.zeros(len(index), dtype=bool)
    step = pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    return np.asarray((index - grid_origin(index, timeframe)) % step != pd.Timedelta(0))


def resolve_timeframe(df: pd.DataFrame, explicit: str | None = None) -> str:
    """Inférer seulement une grille régulière ; les séries trouées doivent la déclarer."""
    declared = df.attrs.get("timeframe")
    if explicit and declared and explicit != declared:
        raise ValueError("Timeframe explicite incompatible avec les métadonnées.")
    tf = explicit or declared
    if tf is None and len(df) >= 2:
        deltas = df.index[1:] - df.index[:-1]
        if (deltas == deltas[0]).all():
            tf = next((k for k, v in TIMEFRAME_SECONDS.items()
                       if deltas[0] == pd.Timedelta(seconds=v)), None)
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError("Fréquence source ambiguë : déclarer un timeframe supporté.")
    if unaligned_mask(df.index, tf).any():
        raise ValueError(f"Timestamps hors grille {tf} de clôture UTC.")
    return tf


def expected_grid(index: pd.DatetimeIndex, timeframe: str,
                  expected_index: pd.DatetimeIndex | None = None) -> pd.DatetimeIndex:
    """Une grille explicite représente les clôtures attendues d'un calendrier de séances."""
    from regime_lib.utils.validation import validate_time_index
    validate_time_index(index)
    if expected_index is not None:
        validate_time_index(expected_index)
        if unaligned_mask(expected_index, timeframe).any():
            raise ValueError("Calendrier attendu hors grille du timeframe.")
        if not index.isin(expected_index).all():
            raise ValueError("Des observations sont absentes du calendrier attendu.")
        return expected_index
    if index.empty:
        return index
    return pd.date_range(index[0], index[-1], freq=TIMEFRAME_RULES[timeframe])


def session_grid(start, end, *, timezone: str, opens: str, closes: str,
                 timeframe: str, weekdays=(0, 1, 2, 3, 4), holidays=()) -> pd.DatetimeIndex:
    """Clôtures UTC de séances définies en heure locale, bornes (ouverture, clôture].

    start/end sont des dates locales d'ouverture inclusives. Une clôture <= à
    l'ouverture est le lendemain (séance de nuit). Les heures ambiguës ou
    inexistantes aux changements DST sont refusées, jamais déplacées en silence.
    Les fermetures exceptionnelles sont les dates locales fournies dans holidays.
    La grille reste celle du contrat UTC ; une borne non alignée est refusée.
    """
    from datetime import time
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError("Timeframe inconnu.")
    opening, closing = time.fromisoformat(opens), time.fromisoformat(closes)
    if opening.tzinfo or closing.tzinfo:
        raise ValueError("Les heures de séance doivent être locales sans fuseau.")
    if any(isinstance(day, bool) or not isinstance(day, int) or day not in range(7) for day in weekdays):
        raise ValueError("weekdays doit contenir des entiers entre 0 et 6.")
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first.tz is not None or last.tz is not None or first != first.normalize() or last != last.normalize() or first > last:
        raise ValueError("start/end doivent être des dates locales ordonnées.")
    excluded = {pd.Timestamp(day).date() for day in holidays}
    result = pd.DatetimeIndex([], tz="UTC")
    step = pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    for day in pd.date_range(first, last, freq="D"):
        if day.weekday() not in weekdays or day.date() in excluded:
            continue
        begin = pd.Timestamp.combine(day.date(), opening)
        finish = pd.Timestamp.combine(day.date(), closing)
        if closing <= opening:
            finish += pd.Timedelta(days=1)
        begin = begin.tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
        finish = finish.tz_localize(timezone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
        if unaligned_mask(pd.DatetimeIndex([begin, finish]), timeframe).any():
            raise ValueError("Bornes de séance hors grille UTC ; utiliser un timeframe plus fin.")
        result = result.union(pd.date_range(begin + step, finish, freq=step))
    result.name = "timestamp"
    return result
