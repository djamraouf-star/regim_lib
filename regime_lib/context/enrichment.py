"""
Orchestration : applique toutes les dimensions de contexte à un
DataFrame OHLCV.

Fonction principale : `enrichir_contexte`.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from regime_lib.context._time import debut_de_barre
from regime_lib.context.calendar import (
    charger_holidays,
    enrichir_calendrier,
)
from regime_lib.context.events import (
    charger_events,
    enrichir_events,
)
from regime_lib.context.sessions import enrichir_sessions

if TYPE_CHECKING:
    from regime_lib.config.schema import ContextConfig


def enrichir_contexte(
    df: pd.DataFrame,
    timeframe: str,
    config: "ContextConfig",
) -> pd.DataFrame:
    """
    Enrichit un DataFrame OHLCV avec toutes les colonnes `ctx_*`.

    Colonnes produites :
      - ctx_session      (str)
      - ctx_hour_ny      (int)
      - ctx_hour_utc     (int)
      - ctx_day_of_week  (int)
      - ctx_is_rollover  (bool)
      - ctx_is_holiday   (bool)
      - ctx_in_event     (bool)
      - ctx_event_type   (str)

    Parameters
    ----------
    df : pd.DataFrame
        Barres OHLCV indexées par DatetimeIndex UTC.
    timeframe : str
        Timeframe (clé de TIMEFRAME_RULES).
    config : ContextConfig
        Configuration du contexte (issue du profil).

    Returns
    -------
    pd.DataFrame
        Copie de `df` avec les 8 colonnes `ctx_*` ajoutées à la fin.
    """
    # 1. Sessions
    df = enrichir_sessions(df, timeframe, config)

    # 2. Calendrier (jours + rollover + fériés)
    holidays = None
    chemin_holidays = config.holidays.get("fichier")
    if chemin_holidays:
        holidays = charger_holidays(chemin_holidays)

    df = enrichir_calendrier(df, timeframe, config, holidays)

    # 3. Événements macro
    events = None
    chemin_events = config.events.get("fichier")
    if chemin_events:
        events = charger_events(chemin_events)
        if len(df) > 0 and len(events) > 0:
            dernier_debut = debut_de_barre(df.index, timeframe).max()
            dernier_event = events["datetime_utc"].max()
            dernier_debut = dernier_debut.tz_convert(
                config.reference_tz
            ).normalize()
            dernier_event = dernier_event.tz_convert(
                config.reference_tz
            ).normalize()
            if dernier_debut > dernier_event:
                warnings.warn(
                    "Les données de barres dépassent la date du dernier "
                    f"événement ({dernier_event.date()}) dans {chemin_events}. "
                    "Les événements manquants ne seront pas marqués ; "
                    "mettez à jour le calendrier.",
                    UserWarning,
                    stacklevel=2,
                )

    df = enrichir_events(df, timeframe, config, events)

    return df
