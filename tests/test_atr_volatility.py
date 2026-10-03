"""
Tests spécifiques au détecteur ATR.

Vérifie :
  - l'uniformité de la sortie ;
  - l'absence de lookahead (test de troncature) ;
  - le marquage `is_partial` ;
  - la cohérence de la confidence ;
  - les bornes du REGIME_MAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.volatility.atr_volatility import ATRVolatilityDetector


def _detector(**kwargs) -> ATRVolatilityDetector:
    return ATRVolatilityDetector(timeframe="H1", **kwargs)


def test_fit_predict_returns_regime_and_confidence(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert set(out.columns) == {"regime", "confidence"}
    assert len(out) == len(ohlcv_h1)
    assert out["confidence"].between(0.0, 1.0).all()
    assert out["confidence"].notna().all()


def test_unknown_confidence_is_zero(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "INCONNU"
    assert (out.loc[mask, "confidence"] == 0.0).all()


def test_chop_confidence_is_half(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "CHOP"
    if mask.any():
        assert (out.loc[mask, "confidence"] == 0.5).all()


def test_no_lookahead_by_truncation(ohlcv_h1):
    """Le calcul sur les k premières barres doit être identique au calcul
    sur la série complète restreint aux k premières barres."""
    det = _detector()
    full = det.fit_predict(ohlcv_h1)["regime"]
    k = len(ohlcv_h1) // 2
    truncated = det.fit_predict(ohlcv_h1.iloc[:k])["regime"]
    pd.testing.assert_series_equal(
        full.iloc[:k].reset_index(drop=True),
        truncated.reset_index(drop=True),
        check_names=False,
    )


def test_last_bar_is_unknown_when_partial(ohlcv_h1):
    df = ohlcv_h1.copy()
    df["is_partial"] = False
    df.iloc[-1, df.columns.get_loc("is_partial")] = True

    det = _detector()
    out = det.fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"
    assert out["confidence"].iloc[-1] == 0.0


def test_unaligned_timestamps_are_unknown(ohlcv_h1):
    """Si on ne fournit pas is_partial, les barres hors grille H1
    (minutes != 0) doivent être classées INCONNU."""
    df = ohlcv_h1.copy()
    # On décale une barre à 30 minutes → non alignée sur la grille H1.
    new_index = df.index.copy()
    new_index = new_index[:5].union(
        pd.DatetimeIndex(["2024-01-01 05:30:00+00:00"])
    )
    df = df.iloc[:6].copy()
    df.index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-01 02:00:00+00:00",
            "2024-01-01 03:00:00+00:00",
            "2024-01-01 04:00:00+00:00",
            "2024-01-01 05:30:00+00:00",  # hors grille
        ],
        name="timestamp",
    )
    det = _detector()
    out = det.fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"


def test_empty_dataframe():
    det = _detector()
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="timestamp"),
    )
    out = det.fit_predict(empty)
    assert out.empty
    assert set(out.columns) == {"regime", "confidence"}


def test_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match="q_low < q_high"):
        ATRVolatilityDetector(timeframe="H1", q_low=0.7, q_high=0.3)


def test_regime_map_covers_all_labels(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert set(out["regime"].unique()).issubset(set(det.REGIME_MAP.keys()))
