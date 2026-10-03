"""
Calcul du calendrier : jour de semaine, rollover, jours fériés.

Colonnes produites
------------------
- ctx_day_of_week : int 0–6 (0 = lundi, NY)
- ctx_is_rollover : bool (fenêtre définie dans context.rollover)
- ctx_is_holiday  : bool (depuis context.holidays.fichier)

Convention de rollover
----------------------
Une barre est marquée rollover si son **début** (étiquette − durée)
tombe dans la fenêtre configurée. Pour une barre H4 qui chevauche
partiellement le rollover, seule la barre dont le début est dans la
fenêtre est marquée — cohérent avec l'affectation des sessions.

Cette convention est **conservatrice par défaut**. Pour marquer toute
barre qui chevauche la fenêtre, utiliser `par_chevauchement=True`
(option à venir, voir TODO).
"""

from __future__ import annotations

from pathlib import Path
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


# Répertoire `regime_lib/context/data/`, où vivent les CSV du contexte.
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _resoudre_chemin(chemin: str | Path) -> Path:
    """
    Résout un chemin vers un fichier de données du contexte.

    - Chemin absolu : utilisé tel quel.
    - Chemin relatif commençant par 'context/data/' : préfixe retiré,
      puis résolu sous `regime_lib/context/data/`.
    - Autre chemin relatif : résolu sous `regime_lib/context/data/`.
    """
    p = Path(chemin)
    if p.is_absolute():
        return p
    # Tolérer le préfixe "context/data/" pour rester compatible avec
    # les chemins écrits dans le YAML.
    parts = p.parts
    if len(parts) >= 2 and parts[0] == "context" and parts[1] == "data":
        p = Path(*parts[2:])
    return _DATA_DIR / p


def charger_holidays(chemin: str | Path) -> pd.DatetimeIndex:
    """
    Charge un CSV de jours fériés.

    Format attendu : colonne `date` (YYYY-MM-DD), colonne `label`
    optionnelle.

    Parameters
    ----------
    chemin : str | Path
        Chemin absolu ou relatif à la racine du projet.

    Returns
    -------
    pd.DatetimeIndex
        Dates normalisées à minuit UTC, sans doublons.
    """
    path = _resoudre_chemin(chemin)
    if not path.exists():
        raise FileNotFoundError(f"Fichier fériés introuvable : {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(
            f"Le CSV des fériés doit contenir une colonne 'date' : {path}"
        )

    dates = pd.to_datetime(df["date"], utc=True).dt.normalize()
    return pd.DatetimeIndex(dates.drop_duplicates().sort_values())


def calculer_calendrier(
    index: pd.DatetimeIndex,
    timeframe: str,
    config: "ContextConfig",
    holidays: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Calcule les colonnes de calendrier.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index UTC des barres.
    timeframe : str
        Timeframe des barres.
    config : ContextConfig
        Configuration (fuseau, rollover, fériés).
    holidays : pd.DatetimeIndex | None
        Dates des fériés, normalisées à minuit UTC. None = pas de fériés.

    Returns
    -------
    pd.DataFrame
        Indexé par `index`, colonnes :
          - ctx_day_of_week (int)
          - ctx_is_rollover (bool)
          - ctx_is_holiday  (bool)
    """
    verifier_timeframe(timeframe)

    if len(index) == 0:
        return pd.DataFrame(
            columns=["ctx_day_of_week", "ctx_is_rollover", "ctx_is_holiday"],
            index=index,
        )
    verifier_index_utc(index)

    debut = debut_de_barre(index, timeframe)
    debut_ny = debut.tz_convert(config.reference_tz)

    # Jour de semaine (NY)
    day_of_week = np.asarray(debut_ny.dayofweek, dtype=int)

    # Rollover
    if config.rollover is not None:
        _, _, min_ny = heures_minutes(debut, config.reference_tz)
        d = minutes_depuis_minuit(config.rollover["debut"])
        f = minutes_depuis_minuit(config.rollover["fin"])
        is_rollover = masque_intervalle(min_ny, d, f)
    else:
        is_rollover = np.zeros(len(index), dtype=bool)

    # Fériés (comparaison sur le jour UTC normalisé du début de barre)
    if holidays is not None and len(holidays) > 0:
        jour = debut.tz_convert("UTC").normalize()
        is_holiday = np.asarray(jour.isin(holidays), dtype=bool)
    else:
        is_holiday = np.zeros(len(index), dtype=bool)

    return pd.DataFrame(
        {
            "ctx_day_of_week": day_of_week,
            "ctx_is_rollover": is_rollover,
            "ctx_is_holiday": is_holiday,
        },
        index=index,
    )


def enrichir_calendrier(
    df: pd.DataFrame,
    timeframe: str,
    config: "ContextConfig",
    holidays: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Ajoute les colonnes de calendrier à un DataFrame."""
    ctx = calculer_calendrier(df.index, timeframe, config, holidays)
    return pd.concat([df, ctx], axis=1)
