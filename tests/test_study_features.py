"""Tests de l'extraction des features."""

from __future__ import annotations

import pandas as pd
import pytest

from regime_lib.study.features import (
    extraire_features,
    lister_features_disponibles,
)


@pytest.fixture
def df_regimes():
    """Parquet long minimal avec 2 méthodes."""
    ts = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": list(ts) + list(ts),
        "regime": ["RANGE"] * 10 + ["CHOP"] * 10,
        "method": ["adx"] * 10 + ["shannon"] * 10,
        "timeframe": ["H1"] * 20,
        "asset": ["EURUSD"] * 20,
        "confidence": [1.0] * 20,
        "regime_id": [0] * 10 + [1] * 10,
        "params_hash": ["h"] * 20,
        "ctx_session": ["NY"] * 20,
    })


def test_lister_features(df_regimes):
    inv = lister_features_disponibles(df_regimes)
    assert "regime" in inv["categorielles"]
    assert "ctx_session" in inv["categorielles"]
    assert "confidence" in inv["numeriques"]


def test_extraire_features(df_regimes):
    out = extraire_features(
        df_regimes, methodes=["adx", "shannon"], features=["regime"],
    )
    assert "adx__regime" in out.columns
    assert "shannon__regime" in out.columns
    assert len(out) == 10  # 10 timestamps


def test_extraire_rejette_offline(df_regimes):
    """Les méthodes offline doivent être filtrées par défaut."""
    df_avec_offline = df_regimes.copy()
    df_avec_offline.loc[0:9, "method"] = "hmm_gaussian"
    out = extraire_features(
        df_avec_offline, features=["regime"],
        methodes_causales_only=True,
    )
    assert "hmm_gaussian__regime" not in out.columns


def test_extraire_feature_absente(df_regimes):
    with pytest.raises(ValueError, match="absentes"):
        extraire_features(
            df_regimes, features=["ctx_is_rollover"],
        )
