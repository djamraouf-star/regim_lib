"""
Calcul des sessions de marché.

Définition
----------
Les sessions sont définies dans `context.sessions` du profil, avec des
bornes au format "HH:MM" dans le fuseau `context.reference_tz`.

Une barre étiquetée à droite couvre `[étiquette - durée, étiquette)`.
Son **début** détermine sa session.

Cas limites
-----------
- Une session peut chevaucher minuit (ex. ASIE 19:00 → 03:00).
- Une barre non couverte reçoit la valeur "HORS_SESSION".
- En cas de chevauchement entre sessions, la dernière déclarée dans le
  YAML l'emporte (l'ordre du dict compte).

Causalité
---------
Pure fonction du timestamp et du fuseau. Aucun lookahead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regime_lib.context._time import (
    debut_de_barre,
    heures_minutes,
    masque_intervalle,
    minutes_depuis_minuit,
    verifier_index_utc,
    verifier_timeframe,
)

if TYPE_CHECKING:
    from regime_lib.config.schema import ContextConfig


_VALEUR_HORS_SESSION = "HORS_SESSION"


def calculer_sessions(
    index: pd.DatetimeIndex,
    timeframe: str,
    config: "ContextConfig",
) -> pd.DataFrame:
    """
    Calcule les colonnes de session pour un index de barres.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index temporel des barres, en UTC (tz-aware). Convention
        d'étiquetage à droite.
    timeframe : str
        Timeframe des barres (clé de `_time.DUREES`).
    config : ContextConfig
        Configuration du contexte (fuseau + sessions).

    Returns
    -------
    pd.DataFrame
        Indexé par `index`, colonnes :
          - ctx_session   (str) : nom de la session, ou "HORS_SESSION"
          - ctx_hour_ny   (int) : heure locale du début de barre
          - ctx_hour_utc  (int) : heure UTC du début de barre
    """
    verifier_timeframe(timeframe)

    if len(index) == 0:
        return pd.DataFrame(
            columns=["ctx_session", "ctx_hour_ny", "ctx_hour_utc"],
            index=index,
        )
    verifier_index_utc(index)

    debut = debut_de_barre(index, timeframe)
    hour_ny, _, min_ny = heures_minutes(debut, config.reference_tz)
    hour_utc = np.asarray(
        debut.tz_convert("UTC").hour, dtype=int
    )

    # Classification (la dernière session déclarée l'emporte en cas
    # de chevauchement)
    session = np.full(len(index), _VALEUR_HORS_SESSION, dtype=object)
    for nom, bornes in config.sessions.items():
        d = minutes_depuis_minuit(bornes["debut"])
        f = minutes_depuis_minuit(bornes["fin"])
        session[masque_intervalle(min_ny, d, f)] = nom

    return pd.DataFrame(
        {
            "ctx_session": session,
            "ctx_hour_ny": hour_ny,
            "ctx_hour_utc": hour_utc,
        },
        index=index,
    )


def enrichir_sessions(
    df: pd.DataFrame,
    timeframe: str,
    config: "ContextConfig",
) -> pd.DataFrame:
    """Ajoute les colonnes de session à un DataFrame."""
    ctx = calculer_sessions(df.index, timeframe, config)
    return pd.concat([df, ctx], axis=1)
