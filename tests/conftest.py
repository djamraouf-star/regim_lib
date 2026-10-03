"""
Fixtures partagées pour les tests.

Fournit un DataFrame OHLCV synthétique déterministe, indexé à l'heure,
pour éviter de dépendre d'un parquet externe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ohlcv_h1() -> pd.DataFrame:
    """
    OHLCV synthétique H1, 2000 barres, avec une marche aléatoire reproductible.

    - open/close : marche aléatoire gaussienne.
    - high/low   : extensions autour de open/close.
    - volume     : uniforme.
    - index      : DatetimeIndex horaire nommé `timestamp`, UTC.
    """
    rng = np.random.default_rng(seed=42)
    n = 2000

    index = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )

    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0002, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0002, n))
    volume = rng.integers(100, 1000, n).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def ohlcv_columns_uppercase(ohlcv_h1: pd.DataFrame) -> pd.DataFrame:
    """Même DataFrame avec colonnes capitalisées (Open, High, ...)."""
    df = ohlcv_h1.copy()
    df.columns = [c.capitalize() for c in df.columns]
    return df
