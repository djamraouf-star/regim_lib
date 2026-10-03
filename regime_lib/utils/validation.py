"""
Validation d'entrée OHLCV.

Vérifie :
  - présence des colonnes open, high, low, close, volume ;
  - index DatetimeIndex nommé (ou renommable) `timestamp` ;
  - index trié strictement croissant, sans doublons ;
  - absence de NaN sur les colonnes OHLC ; les NaN de volume sont tolérés
    et représentent une donnée de volume indisponible.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

OHLCV_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Valide un DataFrame OHLCV. Lève ValueError en cas de problème.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas.DataFrame.")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes : {missing}.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'index doit être un DatetimeIndex.")

    if df.index.has_duplicates:
        raise ValueError("L'index temporel contient des doublons.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("L'index temporel doit être trié par ordre croissant.")

    # NaN sur OHLC = fatal.
    ohlc = ["open", "high", "low", "close"]
    if df[ohlc].isna().any().any():
        raise ValueError("NaN détecté dans les colonnes OHLC.")

    # NaN dans volume indique que le volume de transactions n'est pas fourni.
