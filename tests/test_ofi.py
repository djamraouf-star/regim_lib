"""
Tests du détecteur Order Flow Imbalance.

Vérifie :
  - le refus si les colonnes requises sont absentes ;
  - l'uniformité de sortie (regime + confidence + mesure) ;
  - la cohérence mathématique (ofi_value ∈ [-1, 1]) ;
  - la classification correcte par signe ;
  - la causalité (non-fuite du futur) ;
  - le marquage `is_partial` et hors grille ;
  - les cas dégénérés ;
  - les validations de paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.volume.ofi import OrderFlowImbalanceDetector


def _detector(**kwargs) -> OrderFlowImbalanceDetector:
    return OrderFlowImbalanceDetector(timeframe="H1", **kwargs)


def _make_ohlcv_volume(
    close: np.ndarray,
    bid_volume: np.ndarray | None = None,
    ask_volume: np.ndarray | None = None,
) -> pd.DataFrame:
    """OHLCV avec volumes bid/ask (nécessaires à l'OFI)."""
    n = len(close)
    idx = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    if bid_volume is None:
        bid_volume = np.full(n, 100.0)
    if ask_volume is None:
        ask_volume = np.full(n, 100.0)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close),
        "low": np.minimum(open_, close),
        "close": close,
        "volume": bid_volume + ask_volume,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
    }, index=idx)


# ---------------------------------------------------------------------------
# Colonnes requises
# ---------------------------------------------------------------------------
def test_refus_si_colonnes_absentes(ohlcv_h1):
    """Le détecteur refuse un DataFrame sans bid_volume/ask_volume."""
    det = _detector()
    with pytest.raises(ValueError, match="bid_volume"):
        det.fit_predict(ohlcv_h1)


def test_accepte_avec_colonnes(ohlcv_h1):
    df = ohlcv_h1.copy()
    df["bid_volume"] = 100.0
    df["ask_volume"] = 100.0
    det = _detector()
    out = det.fit_predict(df)
    assert "ofi_value" in out.columns


# ---------------------------------------------------------------------------
# Uniformité
# ---------------------------------------------------------------------------
def test_fit_predict_columns():
    n = 100
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv_volume(close)
    det = _detector()
    out = det.fit_predict(df)
    assert set(out.columns) == {"regime", "confidence", "ofi_value"}
    assert len(out) == n


def test_regime_map_couvre_labels():
    n = 100
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv_volume(close)
    out = _detector().fit_predict(df)
    assert set(out["regime"].unique()).issubset(
        set(_detector().REGIME_MAP.keys())
    )


def test_confidence_bounds():
    n = 200
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-4, n))
    bid_vol = rng.uniform(50, 200, n)
    ask_vol = rng.uniform(50, 200, n)
    df = _make_ohlcv_volume(close, bid_vol, ask_vol)
    out = _detector().fit_predict(df)
    assert out["confidence"].between(0.0, 1.0).all()
    assert out["confidence"].notna().all()


def test_inconnu_confidence_zero():
    n = 100
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv_volume(close)
    out = _detector().fit_predict(df)
    mask = out["regime"] == "INCONNU"
    assert (out.loc[mask, "confidence"] == 0.0).all()


# ---------------------------------------------------------------------------
# Cohérence mathématique
# ---------------------------------------------------------------------------
def test_ofi_value_bounds():
    n = 500
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-4, n))
    bid_vol = rng.uniform(10, 500, n)
    ask_vol = rng.uniform(10, 500, n)
    df = _make_ohlcv_volume(close, bid_vol, ask_vol)
    out = _detector().fit_predict(df)
    valid = out["ofi_value"].dropna()
    assert (valid >= -1.0).all()
    assert (valid <= 1.0).all()


def test_ofi_positif_pression_acheteuse():
    """askVolume >> bidVolume → OFI positif → PRESSION_ACHETEUSE."""
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    bid_vol = np.full(n, 100.0)
    ask_vol = np.full(n, 300.0)  # 3× plus d'ask
    df = _make_ohlcv_volume(close, bid_vol, ask_vol)
    det = _detector(fenetre=20, seuil_pression=0.1)
    out = det.fit_predict(df)

    # Après chauffe : toutes les barres en PRESSION_ACHETEUSE
    tail = out.iloc[50:]
    assert (tail["regime"] == "PRESSION_ACHETEUSE").all()


def test_ofi_negatif_pression_vendeuse():
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    bid_vol = np.full(n, 300.0)
    ask_vol = np.full(n, 100.0)
    df = _make_ohlcv_volume(close, bid_vol, ask_vol)
    det = _detector(fenetre=20, seuil_pression=0.1)
    out = det.fit_predict(df)

    tail = out.iloc[50:]
    assert (tail["regime"] == "PRESSION_VENDEUSE").all()


def test_ofi_equilibre():
    """bidVolume = askVolume → OFI = 0 → EQUILIBRE."""
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    bid_vol = np.full(n, 100.0)
    ask_vol = np.full(n, 100.0)
    df = _make_ohlcv_volume(close, bid_vol, ask_vol)
    det = _detector(fenetre=20, seuil_pression=0.1)
    out = det.fit_predict(df)

    tail = out.iloc[50:]
    assert (tail["regime"] == "EQUILIBRE").all()
    assert (tail["ofi_value"].abs() < 1e-10).all()


# ---------------------------------------------------------------------------
# Causalité
# ---------------------------------------------------------------------------
def test_no_future_leak_by_truncation():
    n = 500
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-4, n))
    bid_vol = rng.uniform(50, 200, n)
    ask_vol = rng.uniform(50, 200, n)
    df = _make_ohlcv_volume(close, bid_vol, ask_vol)

    det = _detector()
    full = det.fit_predict(df)["regime"]
    k = n // 2
    truncated = det.fit_predict(df.iloc[:k])["regime"]

    pd.testing.assert_series_equal(
        full.iloc[:k].reset_index(drop=True),
        truncated.reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Bougies non clôturées
# ---------------------------------------------------------------------------
def test_last_bar_unknown_si_partial():
    n = 100
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv_volume(close)
    df["is_partial"] = False
    df.iloc[-1, df.columns.get_loc("is_partial")] = True

    out = _detector().fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"
    assert out["confidence"].iloc[-1] == 0.0


def test_unaligned_timestamps_unknown():
    """Barres hors grille H1 → INCONNU en l'absence de `is_partial`."""
    index = pd.DatetimeIndex([
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 01:00:00+00:00",
        "2024-01-01 02:00:00+00:00",
        "2024-01-01 03:00:00+00:00",
        "2024-01-01 04:00:00+00:00",
        "2024-01-01 05:30:00+00:00",
    ], name="timestamp")
    n = len(index)
    rng = np.random.default_rng(1)
    close = 1.10 + np.cumsum(rng.normal(0, 5e-5, n))
    df = _make_ohlcv_volume(
        close,
        np.full(n, 100.0),
        np.full(n, 100.0),
    )
    df.index = index

    out = _detector().fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"


# ---------------------------------------------------------------------------
# Cas dégénérés
# ---------------------------------------------------------------------------
def test_empty_dataframe():
    det = _detector()
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume",
                 "bid_volume", "ask_volume"],
        index=pd.DatetimeIndex([], name="timestamp"),
    )
    out = det.fit_predict(empty)
    assert out.empty
    assert set(out.columns) == {"regime", "confidence", "ofi_value"}


def test_zero_volume_renvoie_nan():
    """Si bid_volume = ask_volume = 0, OFI = NaN → INCONNU."""
    n = 100
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv_volume(close, np.zeros(n), np.zeros(n))
    out = _detector().fit_predict(df)
    # Tous les ofi_value doivent être NaN (division 0/0)
    assert out["ofi_value"].isna().all()


# ---------------------------------------------------------------------------
# Validations paramètres
# ---------------------------------------------------------------------------
def test_rejects_invalid_fenetre():
    with pytest.raises(ValueError, match="fenetre"):
        OrderFlowImbalanceDetector(timeframe="H1", fenetre=1)


def test_rejects_invalid_seuil():
    with pytest.raises(ValueError, match="seuil_pression"):
        OrderFlowImbalanceDetector(timeframe="H1", seuil_pression=0.0)
    with pytest.raises(ValueError, match="seuil_pression"):
        OrderFlowImbalanceDetector(timeframe="H1", seuil_pression=1.5)


# ---------------------------------------------------------------------------
# Non-régression masque unaligned
# ---------------------------------------------------------------------------
def test_unaligned_mask_agnostic_to_resolution():
    n = 100
    for unit in ("s", "ms", "us", "ns"):
        index = pd.date_range(
            "2024-01-01", periods=n, freq="1h", tz="UTC"
        )
        try:
            index = index.as_unit(unit, round_ok=True)
        except (AttributeError, TypeError):
            pytest.skip(f"as_unit({unit}) non supporté")
        index.name = "timestamp"
        mask = OrderFlowImbalanceDetector._unaligned_mask(index, "H1")
        assert not mask.any(), (
            f"Résolution {unit} : {int(mask.sum())} barres hors grille"
        )
