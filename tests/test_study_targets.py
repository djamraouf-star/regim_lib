"""Tests des cibles d'étude."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.study.targets import (
    calculer_cible,
    liste_cibles,
    parser_cible,
)


@pytest.fixture
def ohlcv_simple():
    """OHLCV synthétique sur 100 barres."""
    n = 100
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    return pd.DataFrame(
        {"open": close, "high": close, "low": close,
         "close": close, "volume": np.ones(n)},
        index=idx,
    )


def test_parser_valide():
    assert parser_cible("ret_fwd_5") == ("ret", 5)
    assert parser_cible("vol_fwd_20") == ("vol", 20)
    assert parser_cible("dd_fwd_1") == ("dd", 1)


def test_parser_invalide():
    with pytest.raises(ValueError, match="Format attendu"):
        parser_cible("ret_5")
    with pytest.raises(ValueError, match="Famille"):
        parser_cible("xxx_fwd_5")
    with pytest.raises(ValueError, match="K doit"):
        parser_cible("ret_fwd_0")


def test_ret_fwd_calcul(ohlcv_simple):
    c = calculer_cible(ohlcv_simple, "ret_fwd_5")
    # Valeurs attendues : log(close[t+5]/close[t])
    attendu = np.log(ohlcv_simple["close"].shift(-5) / ohlcv_simple["close"])
    pd.testing.assert_series_equal(c, attendu, check_names=False)
    # Les 5 dernières sont NaN
    assert c.iloc[-5:].isna().all()


def test_vol_fwd_calcul(ohlcv_simple):
    c = calculer_cible(ohlcv_simple, "vol_fwd_20")
    assert c.name == "vol_fwd_20"
    # Non-négatif partout où défini
    assert (c.dropna() >= 0).all()
    # Les 20 dernières sont NaN
    assert c.iloc[-20:].isna().all()


def test_dd_fwd_valeurs(ohlcv_simple):
    c = calculer_cible(ohlcv_simple, "dd_fwd_5")
    # Drawdown ≤ 0
    assert (c.dropna() <= 0).all()


def test_ru_fwd_valeurs(ohlcv_simple):
    c = calculer_cible(ohlcv_simple, "ru_fwd_5")
    # Run-up ≥ 0
    assert (c.dropna() >= 0).all()


def test_liste_cibles():
    noms = liste_cibles(horizons=[5], familles=["ret"])
    assert noms == ["ret_fwd_5"]
    noms = liste_cibles(horizons=[1, 5], familles=["ret", "vol"])
    assert "ret_fwd_1" in noms
    assert "vol_fwd_5" in noms
    assert len(noms) == 3
    assert "vol_fwd_1" not in noms
