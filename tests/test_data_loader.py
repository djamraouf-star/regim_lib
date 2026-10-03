"""Tests du chargement et du resampling."""

from __future__ import annotations

import pandas as pd
import pytest

from regime_lib.core.data_loader import (
    agreger_tick,
    charger_tick,
    infer_asset_from_url,
    load_parquet,
    resample,
)


def test_infer_asset_from_url():
    assert infer_asset_from_url("EURUSD_M1_2025.parquet") == "EURUSD"
    assert infer_asset_from_url("/tmp/GBPJPY_H1.parquet") == "GBPJPY"
    assert infer_asset_from_url("no_underscore.parquet") == "no"


def test_load_parquet_normalizes_columns(tmp_path, ohlcv_columns_uppercase):
    p = tmp_path / "EURUSD_H1.parquet"
    ohlcv_columns_uppercase.to_parquet(p)

    df, asset = load_parquet(str(p))
    assert asset == "EURUSD"
    assert df.columns.tolist() == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "timestamp"


def test_load_parquet_rejects_missing_columns(tmp_path):
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9]},  # close/volume manquants
        index=pd.DatetimeIndex(["2025-01-01"], name="timestamp"),
    )
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="Colonnes OHLCV manquantes"):
        load_parquet(str(p), asset="EURUSD")


def _write_quote_ticks(path):
    ticks = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                [
                    "2026-01-02 01:00:10",
                    "2026-01-02 01:00:20",
                    "2026-01-02 01:00:20",
                    "2026-01-02 01:00:40",
                    "2026-01-02 01:01:00",
                    "2026-01-02 01:01:20",
                ]
            ),
            "Bid Price": [9.0, 10.0, 10.5, 8.0, 10.0, 12.0],
            "Ask Price": [10.0, 12.0, 12.5, 9.0, 11.0, 13.0],
            "Bid Volume": [0.1] * 6,
            "Ask Volume": [0.1] * 6,
        }
    )
    ticks.to_parquet(path)
    return ticks


def test_load_parquet_aggregates_quote_ticks_using_bid_and_enriches_bars(
    tmp_path,
):
    path = tmp_path / "XAUUSD_ticks.parquet"
    ticks = _write_quote_ticks(path)

    bars, asset = load_parquet(str(path))

    assert asset == "XAUUSD"
    assert bars.attrs["source_type"] == "ticks"
    assert bars.attrs["source_rows"] == len(ticks)
    assert bars.index.tz is not None
    assert str(bars.index.tz) == "UTC"
    assert bars.index.name == "timestamp"
    assert bars.attrs["price_side"] == "bid"
    assert bars.columns.tolist() == [
        "open", "high", "low", "close", "volume",
        "bid_volume", "ask_volume",
        "spread_mean", "spread_max", "spread_close", "ask_close",
        "tick_count", "is_partial",
    ]
    assert bars.loc[pd.Timestamp("2026-01-02 01:01:00", tz="UTC"),
                    ["open", "high", "low", "close"]].tolist() == [
        9.0, 10.5, 8.0, 8.0
    ]
    assert bars.loc[pd.Timestamp("2026-01-02 01:02:00", tz="UTC"),
                    ["open", "high", "low", "close"]].tolist() == [
        10.0, 12.0, 10.0, 12.0
    ]
    first_minute = bars.loc[pd.Timestamp("2026-01-02 01:01:00", tz="UTC")]
    assert first_minute["volume"] == pytest.approx(0.8)
    assert first_minute["bid_volume"] == pytest.approx(0.4)
    assert first_minute["ask_volume"] == pytest.approx(0.4)
    assert first_minute["spread_mean"] == pytest.approx(1.5)
    assert first_minute["spread_max"] == 2.0
    assert first_minute["spread_close"] == 1.0
    assert first_minute["ask_close"] == 9.0
    assert first_minute["tick_count"] == 4
    assert bars["is_partial"].tolist() == [True, True]

    hourly = resample(bars, "H1")
    assert hourly.loc[pd.Timestamp("2026-01-02 02:00:00", tz="UTC"),
                      "volume"] == pytest.approx(1.2)
    assert hourly.loc[pd.Timestamp("2026-01-02 02:00:00", tz="UTC"),
                      "bid_volume"] == pytest.approx(0.6)
    assert hourly.loc[pd.Timestamp("2026-01-02 02:00:00", tz="UTC"),
                      "ask_volume"] == pytest.approx(0.6)
    assert hourly.loc[pd.Timestamp("2026-01-02 02:00:00", tz="UTC"),
                      "close"] == 12.0
    assert hourly["is_partial"].tolist() == [True]
    assert "spread_mean" in hourly.columns


@pytest.mark.parametrize(
    ("price_side", "expected"),
    [
        ("bid", [9.0, 10.5, 8.0, 8.0]),
        ("ask", [10.0, 12.5, 9.0, 9.0]),
        ("mid", [9.5, 11.5, 8.5, 8.5]),
    ],
)
def test_agreger_tick_supports_bid_ask_and_mid(tmp_path, price_side, expected):
    path = tmp_path / "XAUUSD_ticks.parquet"
    _write_quote_ticks(path)

    ticks = charger_tick(str(path))
    bars = agreger_tick(ticks, "M1", prix=price_side)

    assert bars.loc[
        pd.Timestamp("2026-01-02 01:01:00", tz="UTC"),
        ["open", "high", "low", "close"],
    ].tolist() == expected


def test_load_parquet_can_stream_ticks_in_batches(tmp_path):
    path = tmp_path / "XAUUSD_ticks.parquet"
    _write_quote_ticks(path)

    full, _ = load_parquet(str(path))
    streamed, _ = load_parquet(str(path), batch_size=2)

    pd.testing.assert_frame_equal(streamed, full)
    assert streamed.attrs == full.attrs


def test_load_parquet_rejects_ambiguous_normalized_column_names(tmp_path):
    path = tmp_path / "ambiguous.parquet"
    df = pd.DataFrame(
        [[1.0, 1.0, 1.1, 0.9, 1.0, 10.0]],
        columns=["timestamp", "Open", " open ", "high", "low", "close"],
    )
    df["volume"] = 1.0
    df["timestamp"] = pd.to_datetime(["2025-01-01"])
    df.to_parquet(path)

    with pytest.raises(ValueError, match="(?i)noms de colonnes ambigus"):
        load_parquet(str(path), asset="EURUSD")


def test_resample_preserves_ohlcv_logic(ohlcv_h1):
    """
    Vérifie les règles d'agrégation OHLCV du resample :
        open   = première barre de la fenêtre
        high   = max
        low    = min
        close  = dernière barre de la fenêtre
        volume = somme

    On identifie une fenêtre H4 par sa borne droite (label='right'),
    puis on compare aux barres H1 de (borne - 4h, borne].
    """
    df = resample(ohlcv_h1, "H4")
    assert df.index.name == "timestamp"

    # Prend une fenêtre H4 intermédiaire, bien après le début.
    ts = df.index[10]
    window_start = ts - pd.Timedelta(hours=4)

    src = ohlcv_h1.loc[
        (ohlcv_h1.index > window_start) & (ohlcv_h1.index <= ts)
    ]
    assert len(src) == 4, f"Fenêtre attendue de 4 barres, trouvée {len(src)}"

    row = df.loc[ts]
    assert row["open"] == src["open"].iloc[0]
    assert row["high"] == src["high"].max()
    assert row["low"] == src["low"].min()
    assert row["close"] == src["close"].iloc[-1]
    assert row["volume"] == src["volume"].sum()


def test_resample_marks_last_bar_partial(ohlcv_h1):
    df = resample(ohlcv_h1, "H4")
    assert df["is_partial"].iloc[-1] is True or df["is_partial"].iloc[-1] == True
    assert not df["is_partial"].iloc[:-1].any()


def test_resample_rejects_unknown_timeframe(ohlcv_h1):
    with pytest.raises(ValueError, match="Timeframe inconnu"):
        resample(ohlcv_h1, "Z9")
