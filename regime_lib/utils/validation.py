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
from functools import wraps

import pandas as pd
import numpy as np

OHLCV_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]


def validate_time_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("L'index doit être un DatetimeIndex.")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Index temporel invalide : NaT, doublons ou ordre non croissant.")
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("L'index temporel doit être explicitement en UTC.")


def validate_alignment(left: pd.Index, right: pd.Index) -> None:
    validate_time_index(left)
    validate_time_index(right)
    if not left.equals(right):
        raise ValueError("Alignement strict requis : les index temporels diffèrent.")


def validity_mask(df: pd.DataFrame) -> pd.Series:
    """Validité publiée, sans interprétation numérique des labels ou des scores."""
    valid = pd.Series(True, index=df.index)
    for col in ("is_partial", "is_warmup"):
        if col in df:
            if not df[col].dropna().isin([True, False]).all():
                raise ValueError(f"{col} doit être booléen.")
            valid &= ~df[col].fillna(True).astype(bool)
    if "is_valid" in df:
        if not df["is_valid"].dropna().isin([True, False]).all():
            raise ValueError("is_valid doit être booléen.")
        valid &= df["is_valid"].fillna(False).astype(bool)
    if "regime" in df:
        valid &= df["regime"].notna() & df["regime"].ne("INCONNU")
    return valid


def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Valide un DataFrame OHLCV. Lève ValueError en cas de problème.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas.DataFrame.")

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes OHLCV manquantes : {missing}.")

    validate_time_index(df.index)
    if df.columns.has_duplicates:
        raise ValueError("Colonnes OHLCV dupliquées.")

    # NaN sur OHLC = fatal.
    ohlc = ["open", "high", "low", "close"]
    for col in ohlc + [c for c in ("volume", "bid_volume", "ask_volume") if c in df]:
        if not df.empty and not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"La colonne {col} doit être numérique.")
    if df[ohlc].isna().any().any():
        raise ValueError("NaN détecté dans les colonnes OHLC.")
    values = df[ohlc].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Les prix OHLC doivent être finis et strictement positifs.")
    if ((df["high"] < df[["open", "close", "low"]].max(axis=1)) |
            (df["low"] > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Relations OHLC incohérentes.")
    for col in ("volume", "bid_volume", "ask_volume"):
        if col in df:
            available = df[col].dropna().to_numpy(dtype=float)
            if not np.isfinite(available).all() or (available < 0).any():
                raise ValueError(f"{col} doit être fini et non négatif lorsqu'il est fourni.")
    validity_mask(df)


def validated_detector(method):
    """Contrat commun pour les appels directs et le CLI, sans changer les calculs."""
    @wraps(method)
    def wrapped(self, df):
        data = df.copy()
        data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
        if data.empty and isinstance(data.index, pd.DatetimeIndex) and data.index.tz is None:
            data.index = data.index.tz_localize("UTC")
        validate_ohlcv(data)
        if data.attrs.get("timeframe") and self.timeframe and data.attrs["timeframe"] != self.timeframe:
            raise ValueError("Timeframe du détecteur incompatible avec les données.")
        self.verifier_colonnes(data)
        for col in self.REQUIRES_COLUMNS:
            if col in ("volume", "bid_volume", "ask_volume") and data[col].isna().any():
                raise ValueError(f"{self.name} requiert {col} disponible sur chaque barre.")
        result = method(self, data)
        if data.empty and result.empty:
            result.index = data.index
        validate_alignment(data.index, result.index)
        for col in ("is_partial", "is_warmup", "is_valid", "coverage",
                    "source_count", "expected_count"):
            if col in data:
                if col in result and col in ("is_partial", "is_warmup"):
                    result[col] = result[col].fillna(True) | data[col].fillna(True)
                elif col in result and col == "is_valid":
                    result[col] = result[col].fillna(False) & data[col].fillna(False)
                else:
                    result[col] = data[col]
        result.attrs.update(data.attrs)
        result.attrs.update(self.interpretation())
        return result
    return wrapped
