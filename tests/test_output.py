"""Tests du format uniforme."""

from __future__ import annotations

import pandas as pd
import pytest

from regime_lib.core.output import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    to_uniform,
)


def _df_regime(index, labels):
    return pd.DataFrame({"regime": labels}, index=index)


def test_to_uniform_prefix_order(ohlcv_h1):
    idx = ohlcv_h1.index[:5]
    df = _df_regime(idx, ["A", "A", "B", "B", "A"])

    out = to_uniform(
        df,
        method="m",
        timeframe="H1",
        asset="EURUSD",
        regime_map={"A": 0, "B": 1},
    )
    assert out.columns.tolist()[: len(REQUIRED_COLUMNS)] == REQUIRED_COLUMNS
    assert out["regime_id"].tolist() == [0, 0, 1, 1, 0]
    assert (out["confidence"] == 1.0).all()


def test_to_uniform_rejects_unknown_label(ohlcv_h1):
    df = _df_regime(ohlcv_h1.index[:3], ["A", "B", "Z"])
    with pytest.raises(ValueError, match="Labels de régime inconnus"):
        to_uniform(
            df, method="m", timeframe="H1", asset="EURUSD",
            regime_map={"A": 0, "B": 1},
        )


def test_to_uniform_rejects_nan_confidence(ohlcv_h1):
    df = _df_regime(ohlcv_h1.index[:3], ["A", "A", "A"])
    df["confidence"] = [0.5, float("nan"), 0.5]
    with pytest.raises(ValueError, match="NaN"):
        to_uniform(
            df, method="m", timeframe="H1", asset="EURUSD",
            regime_map={"A": 0},
        )


def test_to_uniform_rejects_out_of_range_confidence(ohlcv_h1):
    df = _df_regime(ohlcv_h1.index[:3], ["A", "A", "A"])
    df["confidence"] = [0.5, 1.5, 0.5]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        to_uniform(
            df, method="m", timeframe="H1", asset="EURUSD",
            regime_map={"A": 0},
        )


def test_to_uniform_materializes_timestamp(ohlcv_h1):
    df = _df_regime(ohlcv_h1.index[:3], ["A", "A", "A"])
    out = to_uniform(
        df, method="m", timeframe="H1", asset="EURUSD",
        regime_map={"A": 0},
    )
    assert "timestamp" in out.columns
    assert isinstance(out.index, pd.RangeIndex)
