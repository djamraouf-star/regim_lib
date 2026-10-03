"""
Tests spécifiques au détecteur ADX (Average Directional Index de Wilder).

Vérifie :
  - l'uniformité de la sortie (colonnes, types, index) ;
  - l'absence de lookahead (test de troncature) ;
  - les calculs des composantes internes (TR, ±DM, lissage Wilder, ADX) ;
  - la classification sur tendances franches et zones de range ;
  - les bornes et la logique de confidence ;
  - le marquage `is_partial` et hors grille ;
  - la gestion des cas dégénérés (DataFrame vide, série courte) ;
  - la validation des hyperparamètres ;
  - l'insensibilité de la détection hors grille à la résolution de l'index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.trend.adx import (
    ADXDetector,
    _compute_adx,
    _directional_movement,
    _true_range,
    _wilder_smooth,
)


def _detector(**kwargs) -> ADXDetector:
    return ADXDetector(timeframe="H1", **kwargs)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------
def test_true_range():
    high = np.array([10.0, 12.0, 11.0])
    low = np.array([8.0, 9.0, 10.0])
    close = np.array([9.0, 11.0, 10.5])

    tr = _true_range(high, low, close)
    assert len(tr) == 3
    # t=0: high[0] - low[0] = 2.0
    assert np.isclose(tr[0], 2.0)
    # t=1: max(12-9=3, |12-9|=3, |9-9|=0) = 3.0
    assert np.isclose(tr[1], 3.0)
    # t=2: max(11-10=1, |11-11|=0, |10-11|=1) = 1.0
    assert np.isclose(tr[2], 1.0)


def test_directional_movement():
    high = np.array([10.0, 12.0, 11.0, 13.0])
    low = np.array([5.0, 6.0, 4.0, 5.0])

    pdm, mdm = _directional_movement(high, low)
    assert len(pdm) == 4
    assert len(mdm) == 4
    # t=0: toujours 0
    assert pdm[0] == 0.0 and mdm[0] == 0.0

    # t=1: up = 12-10 = 2, down = 5-6 = -1 -> up > down and up > 0 -> +DM=2, -DM=0
    assert pdm[1] == 2.0 and mdm[1] == 0.0

    # t=2: up = 11-12 = -1, down = 6-4 = 2 -> down > up and down > 0 -> +DM=0, -DM=2
    assert pdm[2] == 0.0 and mdm[2] == 2.0

    # t=3: up = 13-11 = 2, down = 4-5 = -1 -> +DM=2, -DM=0
    assert pdm[3] == 2.0 and mdm[3] == 0.0


def test_directional_movement_short():
    high = np.array([10.0])
    low = np.array([5.0])
    pdm, mdm = _directional_movement(high, low)
    assert len(pdm) == 1 and pdm[0] == 0.0
    assert len(mdm) == 1 and mdm[0] == 0.0


def test_wilder_smooth():
    values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    smoothed = _wilder_smooth(values, n=3)
    # Les n-1 premières valeurs doivent être NaN
    assert np.isnan(smoothed[0])
    assert np.isnan(smoothed[1])
    assert not np.isnan(smoothed[2])


def test_compute_adx_warmup():
    n = 50
    high = np.linspace(10, 20, n)
    low = high - 1.0
    close = high - 0.5
    plus_di, minus_di, adx = _compute_adx(high, low, close, n=14)
    # Warmup pour ADX: 2*n - 1 = 27 barres
    assert np.isnan(adx[:27]).all()
    assert not np.isnan(adx[27:]).any()


# ---------------------------------------------------------------------------
# Contrat de sortie et régimes
# ---------------------------------------------------------------------------
def test_fit_predict_returns_regime_and_confidence(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert set(out.columns) == {"regime", "confidence"}
    assert len(out) == len(ohlcv_h1)
    assert out["confidence"].between(0.0, 1.0).all()
    assert out["confidence"].notna().all()


def test_regime_map_covers_all_labels(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert set(out["regime"].unique()).issubset(set(det.REGIME_MAP.keys()))


def test_no_lookahead_by_truncation(ohlcv_h1):
    """ADX est causal : le résultat sur un préfixe doit être identique."""
    det = _detector()
    full = det.fit_predict(ohlcv_h1)["regime"]
    k = len(ohlcv_h1) // 2
    truncated = det.fit_predict(ohlcv_h1.iloc[:k])["regime"]
    pd.testing.assert_series_equal(
        full.iloc[:k].reset_index(drop=True),
        truncated.reset_index(drop=True),
        check_names=False,
    )


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


def test_strong_uptrend_produces_tendance_haussiere():
    """Tendance haussière franche et continue -> TENDANCE_HAUSSIERE."""
    n = 100
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    # Hausse régulière sans pullback
    close = np.linspace(1.10, 1.30, n)
    high = close + 0.002
    low = close - 0.001
    open_ = close - 0.001
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=index,
    )
    det = _detector(n_adx=14, seuil_tendance=25.0)
    out = det.fit_predict(df)

    # Après la phase de chauffe, la tendance doit être détectée
    regimes_post_warmup = out["regime"].iloc[40:]
    assert "TENDANCE_HAUSSIERE" in regimes_post_warmup.values
    assert (regimes_post_warmup == "TENDANCE_HAUSSIERE").sum() > 20


def test_strong_downtrend_produces_tendance_baissiere():
    """Tendance baissière franche et continue -> TENDANCE_BAISSIERE."""
    n = 100
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    close = np.linspace(1.30, 1.10, n)
    high = close + 0.001
    low = close - 0.002
    open_ = close + 0.001
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=index,
    )
    det = _detector(n_adx=14, seuil_tendance=25.0)
    out = det.fit_predict(df)

    regimes_post_warmup = out["regime"].iloc[40:]
    assert "TENDANCE_BAISSIERE" in regimes_post_warmup.values
    assert (regimes_post_warmup == "TENDANCE_BAISSIERE").sum() > 20


def test_flat_range_produces_range():
    """Marché oscillant sans direction -> +DM et -DM s'annulent -> ADX bas -> RANGE."""
    n = 100
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    # Oscillation rapide haute fréquence (pas de direction dominante)
    close = 1.10 + 0.0005 * (np.arange(n) % 2 * 2 - 1)
    high = close + 0.0005
    low = close - 0.0005
    open_ = close
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 100.0},
        index=index,
    )
    det = _detector(n_adx=14, seuil_range=20.0)
    out = det.fit_predict(df)

    regimes_post_warmup = out["regime"].iloc[40:]
    assert "RANGE" in regimes_post_warmup.values
    assert (regimes_post_warmup == "RANGE").sum() > 20


# ---------------------------------------------------------------------------
# Bougies non clôturées & alignement
# ---------------------------------------------------------------------------
def test_last_bar_is_unknown_when_partial(ohlcv_h1):
    df = ohlcv_h1.copy()
    df["is_partial"] = False
    df.iloc[-1, df.columns.get_loc("is_partial")] = True

    det = _detector()
    out = det.fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"
    assert out["confidence"].iloc[-1] == 0.0


def test_unaligned_timestamps_are_unknown(ohlcv_h1):
    df = ohlcv_h1.iloc[:6].copy()
    df.index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-01 02:00:00+00:00",
            "2024-01-01 03:00:00+00:00",
            "2024-01-01 04:00:00+00:00",
            "2024-01-01 05:30:00+00:00",  # hors grille H1
        ],
        name="timestamp",
    )
    det = _detector()
    out = det.fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"
    assert out["confidence"].iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# Cas limites & validations
# ---------------------------------------------------------------------------
def test_empty_dataframe():
    det = _detector()
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="timestamp"),
    )
    out = det.fit_predict(empty)
    assert out.empty
    assert set(out.columns) == {"regime", "confidence"}


def test_rejects_invalid_n_adx():
    with pytest.raises(ValueError, match="n_adx doit être >= 2"):
        ADXDetector(n_adx=1)


def test_rejects_invalid_thresholds():
    # seuil_range >= seuil_tendance
    with pytest.raises(ValueError, match="seuil_range < seuil_tendance"):
        ADXDetector(seuil_range=30.0, seuil_tendance=25.0)

    # seuil_range <= 0
    with pytest.raises(ValueError, match="seuil_range < seuil_tendance"):
        ADXDetector(seuil_range=-5.0, seuil_tendance=25.0)

    # seuil_tendance >= 100
    with pytest.raises(ValueError, match="seuil_range < seuil_tendance"):
        ADXDetector(seuil_range=20.0, seuil_tendance=105.0)


def test_unaligned_mask_agnostic_to_index_resolution():
    n = 100
    for unit in ("s", "ms", "us", "ns"):
        index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        try:
            index = index.as_unit(unit, round_ok=True)
        except (AttributeError, TypeError):
            pytest.skip(f"as_unit({unit}) non supporté sur cette version")
        index.name = "timestamp"
        mask = ADXDetector._unaligned_mask(index, "H1")
        assert not mask.any(), f"Résolution {unit} : fausses barres hors grille détectées."
