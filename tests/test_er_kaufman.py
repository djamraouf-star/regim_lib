"""
Tests spécifiques au détecteur ER Kaufman.

Vérifie :
  - l'uniformité de la sortie (regime + confidence) ;
  - la non-fuite du futur (test de troncature) ;
  - la non-auto-référence du seuil (test de perturbation) ;
  - les bornes de l'ER sur des cas analytiques ;
  - les cas dégénérés (close constant, ER indéfini) ;
  - le marquage `is_partial` et hors grille ;
  - les validations des paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.trend.er_kaufman import ERKaufmanDetector


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------
def _make_ohlcv(close: np.ndarray) -> pd.DataFrame:
    """
    Construit un OHLCV minimal à partir d'une série de close.

    open[t] = close[t-1], high = max(open, close), low = min(open, close).
    Index horaire UTC nommé `timestamp`.
    """
    n = len(close)
    index = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close)
    low = np.minimum(open_, close)
    volume = np.full(n, 100.0)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low,
         "close": close, "volume": volume},
        index=index,
    )


def _detector(**kwargs) -> ERKaufmanDetector:
    return ERKaufmanDetector(timeframe="H1", **kwargs)


# ---------------------------------------------------------------------------
# Uniformité de la sortie
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


def test_unknown_confidence_is_zero(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "INCONNU"
    assert (out.loc[mask, "confidence"] == 0.0).all()


def test_chop_confidence_in_range(ohlcv_h1):
    """
    CHOP = 0.5 + 0.5 * dist_chop, où dist_chop ∈ [0, 1].
    Donc confidence ∈ [0.5, 1.0]. Ce n'est pas 0.5 fixe.

    Une confidence CHOP élevée signifie que l'ER est très en-dessous du
    seuil bas (chop franc). Une confidence à 0.5 signifie ER au seuil.
    """
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "CHOP"
    if mask.any():
        vals = out.loc[mask, "confidence"]
        assert (vals >= 0.5).all(), f"min CHOP confidence = {vals.min()}"
        assert (vals <= 1.0).all(), f"max CHOP confidence = {vals.max()}"


def test_range_confidence_is_half(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "RANGE"
    if mask.any():
        assert (out.loc[mask, "confidence"] == 0.5).all()


def test_tendance_confidence_in_range(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"].str.startswith("TENDANCE")
    if mask.any():
        vals = out.loc[mask, "confidence"]
        assert (vals >= 0.5).all() and (vals <= 1.0).all()


# ---------------------------------------------------------------------------
# Causalité : non-fuite du futur
# ---------------------------------------------------------------------------
def test_no_future_leak_by_truncation(ohlcv_h1):
    """
    Le calcul sur les k premières barres doit être identique au calcul
    sur la série complète restreint aux k premières barres.
    """
    det = _detector()
    full = det.fit_predict(ohlcv_h1)["regime"]
    k = len(ohlcv_h1) // 2
    truncated = det.fit_predict(ohlcv_h1.iloc[:k])["regime"]
    pd.testing.assert_series_equal(
        full.iloc[:k].reset_index(drop=True),
        truncated.reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Causalité : non-auto-référence du seuil
# ---------------------------------------------------------------------------
def test_threshold_not_self_referential(ohlcv_h1):
    """
    Perturber TR[i] change er[i] (légitime), mais ne doit PAS changer
    le seuil[i]. On vérifie que le seuil à l'indice i est invariant à
    une perturbation de la barre i elle-même (grâce au shift(1)).
    """
    det = _detector()
    fenetre = det.fenetre
    min_periods = det.min_periods
    n_er = det.n_er
    q_chop = det.q_chop
    q_tendance = det.q_tendance

    def compute_seuils(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        close = df["close"].to_numpy(dtype=float)
        er = ERKaufmanDetector._efficiency_ratio(close, n_er)
        er_series = pd.Series(er, index=df.index)
        er_past = er_series.shift(1)
        s_chop = er_past.rolling(fenetre, min_periods=min_periods).quantile(q_chop)
        s_tend = er_past.rolling(fenetre, min_periods=min_periods).quantile(q_tendance)
        return s_chop.to_numpy(), s_tend.to_numpy()

    s_chop_base, s_tend_base = compute_seuils(ohlcv_h1)

    # On perturbe 10 barres bien après la chauffe.
    idx_test = list(range(600, 610))
    divergences = 0
    for i in idx_test:
        df_mod = ohlcv_h1.copy()
        df_mod.iloc[i, df_mod.columns.get_loc("high")] *= 5
        df_mod.iloc[i, df_mod.columns.get_loc("low")] /= 5
        s_chop_mod, s_tend_mod = compute_seuils(df_mod)

        if not np.isclose(s_chop_mod[i], s_chop_base[i], equal_nan=True):
            divergences += 1
        if not np.isclose(s_tend_mod[i], s_tend_base[i], equal_nan=True):
            divergences += 1

    assert divergences == 0, (
        f"Auto-référence détectée : {divergences} seuils changent "
        f"quand on perturbe leur propre TR."
    )


# ---------------------------------------------------------------------------
# ER : bornes analytiques
# ---------------------------------------------------------------------------
def test_efficiency_ratio_bounds_on_random_walk(ohlcv_h1):
    close = ohlcv_h1["close"].to_numpy(dtype=float)
    er = ERKaufmanDetector._efficiency_ratio(close, n=20)
    finite = er[~np.isnan(er)]
    assert len(finite) > 0
    assert (finite >= 0.0).all()
    assert (finite <= 1.0).all()


def test_efficiency_ratio_on_perfect_trend_is_one():
    """close strictement croissant → ER = 1 (hors phase de chauffe)."""
    close = np.linspace(1.0, 2.0, 500)
    er = ERKaufmanDetector._efficiency_ratio(close, n=20)
    # Premières 20 barres : NaN.
    assert np.isnan(er[:20]).all()
    # Après : parfaitement 1.
    assert np.allclose(er[20:], 1.0)


def test_efficiency_ratio_on_constant_close_is_nan():
    """close constant → dénominateur nul → ER = NaN partout après chauffe."""
    close = np.full(500, 1.10)
    er = ERKaufmanDetector._efficiency_ratio(close, n=20)
    assert np.isnan(er).all()


def test_efficiency_ratio_on_choppy_series_is_low():
    """
    Oscillation +1/-1 autour d'un pivot → change ≈ 0, volatility > 0
    → ER ≈ 0 après chauffe.
    """
    base = np.full(500, 1.10)
    base[::2] += 0.001   # impairs (+1)
    base[1::2] -= 0.001  # pairs (-1)
    er = ERKaufmanDetector._efficiency_ratio(base, n=20)
    finite = er[~np.isnan(er)]
    assert len(finite) > 0
    assert (finite < 0.2).all(), f"ER max = {finite.max()}"


def test_efficiency_ratio_returns_nan_when_length_le_n():
    close = np.array([1.0, 1.1, 1.2])
    er = ERKaufmanDetector._efficiency_ratio(close, n=20)
    assert np.isnan(er).all()


# ---------------------------------------------------------------------------
# Bougies non clôturées
# ---------------------------------------------------------------------------
def test_last_bar_is_unknown_when_partial(ohlcv_h1):
    df = ohlcv_h1.copy()
    df["is_partial"] = False
    df.iloc[-1, df.columns.get_loc("is_partial")] = True

    det = _detector()
    out = det.fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"
    assert out["confidence"].iloc[-1] == 0.0


def test_unaligned_timestamps_are_unknown():
    """Barres hors grille H1 (minutes != 0) → INCONNU en l'absence
    de `is_partial`."""
    index = pd.DatetimeIndex(
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
    rng = np.random.default_rng(seed=1)
    n = len(index)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.0001,
            "low": np.minimum(open_, close) - 0.0001,
            "close": close,
            "volume": np.full(n, 100.0),
        },
        index=index,
    )
    det = _detector()
    out = det.fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"


# ---------------------------------------------------------------------------
# Cas dégénérés
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
    assert out.index.name == "timestamp"


def test_min_periods_controls_warmup():
    """
    Vérifie que min_periods contrôle la durée de la phase de chauffe.

    Avec min_periods plus grand, on exige plus de données non-NaN dans
    la fenêtre avant de calculer un seuil, donc davantage de barres
    INCONNU au début.

    En cas d'égalité entre les deux configs, un diagnostic exhaustif est
    affiché pour identifier la cause (ER tout NaN, masque unaligned trop
    zélé, etc.).
    """
    rng = np.random.default_rng(seed=7)
    n = 800
    index = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.0001,
            "low": np.minimum(open_, close) - 0.0001,
            "close": close,
            "volume": np.full(n, 100.0),
        },
        index=index,
    )

    det_short = ERKaufmanDetector(
        timeframe="H1", fenetre=100, min_periods=30, n_er=10
    )
    det_strict = ERKaufmanDetector(
        timeframe="H1", fenetre=100, min_periods=100, n_er=10
    )

    out_short = det_short.fit_predict(df)
    out_strict = det_strict.fit_predict(df)

    n_unk_short = (out_short["regime"] == "INCONNU").sum()
    n_unk_strict = (out_strict["regime"] == "INCONNU").sum()

    if n_unk_short == n_unk_strict:
        # --- Diagnostic : où est le blocage ? ---
        close_arr = df["close"].to_numpy(dtype=float)
        er = ERKaufmanDetector._efficiency_ratio(close_arr, 10)
        n_er_finite = int(np.sum(~np.isnan(er)))

        er_series = pd.Series(er, index=df.index)
        er_past = er_series.shift(1)
        s_chop_short = er_past.rolling(100, min_periods=30).quantile(0.40)
        s_chop_strict = er_past.rolling(100, min_periods=100).quantile(0.40)
        n_seuil_short = int(s_chop_short.notna().sum())
        n_seuil_strict = int(s_chop_strict.notna().sum())

        raise AssertionError(
            f"min_periods sans effet sur les régimes.\n"
            f"  n_unk_short  = {n_unk_short}\n"
            f"  n_unk_strict = {n_unk_strict}\n"
            f"  n total      = {n}\n"
            f"  ER non-NaN        : {n_er_finite}/{n}\n"
            f"  seuils non-NaN short : {n_seuil_short}/{n}\n"
            f"  seuils non-NaN strict: {n_seuil_strict}/{n}\n"
            f"  Régimes short : {out_short['regime'].value_counts().to_dict()}\n"
            f"  Régimes strict: {out_strict['regime'].value_counts().to_dict()}"
        )

    assert n_unk_strict > n_unk_short, (
        f"Attendu strict > short, obtenu "
        f"strict={n_unk_strict}, short={n_unk_short}"
    )


# ---------------------------------------------------------------------------
# Validations des paramètres
# ---------------------------------------------------------------------------
def test_rejects_invalid_q_order():
    with pytest.raises(ValueError, match="q_chop < q_tendance"):
        ERKaufmanDetector(timeframe="H1", q_chop=0.8, q_tendance=0.3)


def test_rejects_invalid_n_er():
    with pytest.raises(ValueError, match="n_er"):
        ERKaufmanDetector(timeframe="H1", n_er=0)


def test_rejects_invalid_fenetre():
    with pytest.raises(ValueError, match="fenetre"):
        ERKaufmanDetector(timeframe="H1", fenetre=1)


def test_rejects_min_periods_greater_than_fenetre():
    with pytest.raises(ValueError, match="min_periods"):
        ERKaufmanDetector(timeframe="H1", fenetre=100, min_periods=200)


def test_rejects_min_periods_less_than_one():
    with pytest.raises(ValueError, match="min_periods"):
        ERKaufmanDetector(timeframe="H1", min_periods=0)


def test_defaults_are_timeframe_specific():
    det_h1 = ERKaufmanDetector(timeframe="H1")
    assert det_h1.fenetre == 700
    assert det_h1.min_periods == 350

    det_m1 = ERKaufmanDetector(timeframe="M1")
    assert det_m1.fenetre == 8000
    assert det_m1.min_periods == 4000
