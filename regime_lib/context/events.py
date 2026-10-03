"""
Calcul des fenêtres d'événements macro.

Chaque événement est défini par un datetime en heure NY et un type
(NFP, CPI, FOMC). Une barre est marquée `ctx_in_event = True` si son
**début** tombe dans une fenêtre `[event - avant, event + après[`.

Convention de fenêtre : `avant` et `après` sont en minutes (voir
`context.events.fenetres` du profil).

Causalité : pure fonction du timestamp et des dates d'événements.
Aucun lookahead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regime_lib.context._time import (
    debut_de_barre,
    verifier_index_utc,
    verifier_timeframe,
)
from regime_lib.context.calendar import _resoudre_chemin

if TYPE_CHECKING:
    from regime_lib.config.schema import ContextConfig


def charger_events(chemin: str | Path) -> pd.DataFrame:
    """
    Charge un CSV d'événements.

    Format attendu : colonnes `datetime_ny` (YYYY-MM-DD HH:MM) et
    `event_type`.

    Returns
    -------
    pd.DataFrame
        Colonnes : `datetime_ny` (UTC après conversion), `event_type`.
    """
    path = _resoudre_chemin(chemin)
    df = pd.read_csv(path)

    colonnes_requises = {"datetime_ny", "event_type"}
    manquantes = colonnes_requises - set(df.columns)
    if manquantes:
        raise ValueError(
            f"Le CSV d'événements doit contenir les colonnes "
            f"{colonnes_requises}. Manquantes : {manquantes}."
        )

    # Parse en heure NY, conversion en UTC
    dt_ny = pd.to_datetime(df["datetime_ny"])
    dt_ny = dt_ny.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    )
    df = df.assign(datetime_utc=dt_ny.dt.tz_convert("UTC"))

    return df[["datetime_utc", "event_type"]].dropna().reset_index(drop=True)


def calculer_events(
    index: pd.DatetimeIndex,
    timeframe: str,
    config: "ContextConfig",
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcule les colonnes d'événements.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index UTC des barres.
    timeframe : str
        Timeframe des barres.
    config : ContextConfig
        Configuration (fenêtres par type d'événement).
    events : pd.DataFrame | None
        Sortie de `charger_events`, ou None.

    Returns
    -------
    pd.DataFrame
        Indexé par `index`, colonnes :
          - ctx_in_event   (bool)
          - ctx_event_type (str, vide si pas d'événement)
    """
    verifier_timeframe(timeframe)

    if len(index) == 0:
        return pd.DataFrame(
            columns=["ctx_in_event", "ctx_event_type"],
            index=index,
        )
    verifier_index_utc(index)

    in_event = np.zeros(len(index), dtype=bool)
    event_type = np.full(len(index), "", dtype=object)

    if events is None or len(events) == 0:
        return pd.DataFrame(
            {"ctx_in_event": in_event, "ctx_event_type": event_type},
            index=index,
        )

    debut = debut_de_barre(index, timeframe)
    fenetres = config.events.get("fenetres", {})

    for _, ev in events.iterrows():
        type_ev = ev["event_type"]
        dt_ev = ev["datetime_utc"]

        # Récupérer la fenêtre pour ce type, ou défaut
        cfg = fenetres.get(type_ev, {"avant_min": 5, "apres_min": 60})
        avant = pd.Timedelta(minutes=cfg["avant_min"])
        apres = pd.Timedelta(minutes=cfg["apres_min"])

        masque = (debut >= dt_ev - avant) & (debut < dt_ev + apres)
        in_event |= masque
        # Le type est écrasé par le dernier événement chevauchant,
        # mais en pratique les fenêtres sont disjointes.
        event_type[masque] = type_ev

    return pd.DataFrame(
        {"ctx_in_event": in_event, "ctx_event_type": event_type},
        index=index,
    )


def enrichir_events(
    df: pd.DataFrame,
    timeframe: str,
    config: "ContextConfig",
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ajoute les colonnes d'événements à un DataFrame."""
    ctx = calculer_events(df.index, timeframe, config, events)
    return pd.concat([df, ctx], axis=1)
