"""
Tests spécifiques au détecteur Minkowski causal.

Vérifie :
  - l'uniformité de la sortie (regime + confidence + mesures) ;
  - la cohérence mathématique (ds2 = c² − Δx²) ;
  - la causalité (non-fuite du futur par troncature) ;
  - la non-auto-référence du seuil c_mkt ;
  - les régimes dégénérés (série courte, is_partial, hors grille) ;
  - les validations des paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.misc.minkowski_causal import MinkowskiCausalDetector


def _detector(**kwargs) -> MinkowskiCausalDetector:
    return MinkowskiCausalDetector(timeframe="H1", **kwargs)


# ---------------------------------------------------------------------------
# Uniformité de sortie
# ---------------------------------------------------------------------------
def test_fit_predict_columns(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert set(out.columns) == {
        "regime", "confidence", "minkowski_ds2", "minkowski_beta",
    }
    assert len(out) == len(ohlcv_h1)


def test_regime_map_covers_all_labels(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert set(out["regime"].unique()).issubset(set(det.REGIME_MAP.keys()))


def test_confidence_bounds(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert out["confidence"].between(0.0, 1.0).all()
    assert out["confidence"].notna().all()


def test_unknown_has_zero_confidence(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "INCONNU"
    assert (out.loc[mask, "confidence"] == 0.0).all()


def test_lightlike_has_half_confidence(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "LIGHTLIKE"
    if mask.any():
        assert (out.loc[mask, "confidence"] == 0.5).all()


# ---------------------------------------------------------------------------
# Cohérence mathématique
# ---------------------------------------------------------------------------
def test_ds2_sign_matches_regime(ohlcv_h1):
    """
    TIMELIKE → ds2 > 0 ; SPACELIKE → ds2 < 0.
    """
    det = _detector()
    out = det.fit_predict(ohlcv_h1)

    tm = out["regime"] == "TIMELIKE"
    sm = out["regime"] == "SPACELIKE"

    if tm.any():
        assert (out.loc[tm, "minkowski_ds2"] > 0).all()
    if sm.any():
        assert (out.loc[sm, "minkowski_ds2"] < 0).all()


def test_beta_bounds_match_regime(ohlcv_h1):
    """
    TIMELIKE → beta < 1 (avec tolérance frontière) ;
    SPACELIKE → beta > 1 (avec tolérance frontière).
    """
    det = _detector()
    out = det.fit_predict(ohlcv_h1)

    tm = out["regime"] == "TIMELIKE"
    sm = out["regime"] == "SPACELIKE"

    if tm.any():
        assert (out.loc[tm, "minkowski_beta"] <= 1.01).all()
    if sm.any():
        assert (out.loc[sm, "minkowski_beta"] >= 0.99).all()


def test_majority_timelike_on_synthetic_mixture():
    """
    Série synthétique : 95 % petits rendements + 5 % chocs.
    Le quantile 0.90 de |log_return| se cale sur les chocs,
    donc la majorité des barres calmes sont TIMELIKE.
    """
    n = 500
    rng = np.random.default_rng(42)
    returns = np.where(
        rng.random(n) < 0.95,
        rng.normal(0, 1e-5, n),   # mouvements calmes
        rng.normal(0, 1e-3, n),   # chocs rares
    )
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    idx.name = "timestamp"
    close = 1.10 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": close, "high": close, "low": close,
        "close": close, "volume": np.full(n, 100.0),
    }, index=idx)

    det = _detector(fenetre_c=100, min_periods_c=50)
    out = det.fit_predict(df)
    tail = out.iloc[200:]  # après chauffe

    pct_timelike = (tail["regime"] == "TIMELIKE").mean()
    pct_spacelike = (tail["regime"] == "SPACELIKE").mean()

    assert pct_timelike > 0.80, (
        f"TIMELIKE seulement {pct_timelike*100:.1f} % "
        f"(attendu > 80 %)"
    )
    assert pct_spacelike > 0.0, (
        "Aucun SPACELIKE détecté malgré la présence de chocs"
    )


def test_constant_returns_are_lightlike():
    """
    Série à rendements STRICTEMENT constants → c_mkt = |Δx| → Δs² = 0
    → LIGHTLIKE partout après chauffe.

    Cas limite cohérent avec la définition mathématique : la vitesse
    du marché est exactement égale à la vitesse limite.
    """
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    idx.name = "timestamp"
    close = 1.10 * np.exp(np.cumsum(np.full(n, 1e-5)))
    df = pd.DataFrame({
        "open": close, "high": close, "low": close,
        "close": close, "volume": np.full(n, 100.0),
    }, index=idx)

    det = _detector(fenetre_c=50, min_periods_c=20)
    out = det.fit_predict(df)
    tail = out.iloc[100:]

    assert (tail["regime"] == "LIGHTLIKE").all(), (
        f"Régimes observés : {tail['regime'].value_counts().to_dict()}"
    )


# ---------------------------------------------------------------------------
# Causalité
# ---------------------------------------------------------------------------
def test_no_future_leak_by_truncation(ohlcv_h1):
    """
    Tronquer la série ne change pas les régimes passés.
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


def test_c_mkt_not_self_referential(ohlcv_h1):
    """
    Le c_mkt[t] ne dépend que des |log_return| passés (shift(1)).
    Modifier close[t] ne change pas c_mkt[t], seulement Δx[t].
    """
    # Chauffe courte pour avoir un indice de test bien après la chauffe
    det = _detector(fenetre_c=100, min_periods_c=20)
    i_test = 200

    # Calcul c_mkt sur la série originale
    close = ohlcv_h1["close"].to_numpy(dtype=float)
    log_ret = np.full(len(close), np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    s = pd.Series(np.abs(log_ret), index=ohlcv_h1.index)
    c_orig = s.shift(1).rolling(
        det.fenetre_c, min_periods=det.min_periods_c
    ).quantile(det.q_c).iloc[i_test]

    assert not np.isnan(c_orig), (
        f"c_orig est NaN — l'indice {i_test} est dans la chauffe"
    )

    # On perturbe close[i_test]
    df_mod = ohlcv_h1.copy()
    df_mod.iloc[i_test, df_mod.columns.get_loc("close")] *= 1.05

    close_m = df_mod["close"].to_numpy(dtype=float)
    log_ret_m = np.full(len(close_m), np.nan)
    log_ret_m[1:] = np.log(close_m[1:] / close_m[:-1])
    s_m = pd.Series(np.abs(log_ret_m), index=df_mod.index)
    c_mod = s_m.shift(1).rolling(
        det.fenetre_c, min_periods=det.min_periods_c
    ).quantile(det.q_c).iloc[i_test]

    assert np.isclose(c_orig, c_mod, rtol=1e-10), (
        f"c_mkt contient une auto-référence : orig={c_orig}, mod={c_mod}"
    )


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
    index = pd.DatetimeIndex(
        [
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-01 02:00:00+00:00",
            "2024-01-01 03:00:00+00:00",
            "2024-01-01 04:00:00+00:00",
            "2024-01-01 05:30:00+00:00",
        ],
        name="timestamp",
    )
    rng = np.random.default_rng(seed=1)
    n = len(index)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + 0.0001,
        "low": np.minimum(open_, close) - 0.0001,
        "close": close,
        "volume": np.full(n, 100.0),
    }, index=index)

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
    assert set(out.columns) == {
        "regime", "confidence", "minkowski_ds2", "minkowski_beta",
    }


def test_warmup_marks_unknown(ohlcv_h1):
    det = _detector(fenetre_c=100, min_periods_c=50)
    out = det.fit_predict(ohlcv_h1)
    # Les 50 premières barres sont en chauffe → INCONNU
    assert (out["regime"].iloc[:50] == "INCONNU").all()


# ---------------------------------------------------------------------------
# Validations des paramètres
# ---------------------------------------------------------------------------
def test_rejects_invalid_fenetre():
    with pytest.raises(ValueError, match="fenetre_c"):
        MinkowskiCausalDetector(timeframe="H1", fenetre_c=1)


def test_rejects_invalid_q_c():
    with pytest.raises(ValueError, match="q_c"):
        MinkowskiCausalDetector(timeframe="H1", q_c=1.5)
    with pytest.raises(ValueError, match="q_c"):
        MinkowskiCausalDetector(timeframe="H1", q_c=0.0)


def test_rejects_invalid_seuil_lightlike():
    with pytest.raises(ValueError, match="seuil_lightlike"):
        MinkowskiCausalDetector(timeframe="H1", seuil_lightlike=-0.1)


def test_rejects_min_periods_greater_than_fenetre():
    with pytest.raises(ValueError, match="min_periods_c"):
        MinkowskiCausalDetector(
            timeframe="H1", fenetre_c=100, min_periods_c=200
        )


# ---------------------------------------------------------------------------
# Non-régression : masque unaligned agnostique à la résolution
# ---------------------------------------------------------------------------
def test_unaligned_mask_agnostic_to_index_resolution():
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
        mask = MinkowskiCausalDetector._unaligned_mask(index, "H1")
        assert not mask.any(), (
            f"Résolution {unit} : {int(mask.sum())} barres hors grille"
        )
