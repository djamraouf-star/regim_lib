"""
Tests spécifiques au détecteur d'entropie de Shannon.

Vérifie :
  - l'uniformité de sortie (regime + confidence + mesure) ;
  - la cohérence mathématique (shannon_value ∈ [0, 1]) ;
  - le comportement attendu sur séries contrôlées ;
  - la causalité (troncature, non-fuite du futur) ;
  - le marquage `is_partial` et hors grille ;
  - les cas dégénérés ;
  - les validations de paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.entropy.shannon import (
    ShannonEntropyDetector,
    _shannon_windowed,
)


def _detector(**kwargs) -> ShannonEntropyDetector:
    return ShannonEntropyDetector(timeframe="H1", **kwargs)


def _make_ohlcv(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close),
        "low": np.minimum(open_, close),
        "close": close,
        "volume": np.full(n, 100.0),
    }, index=idx)


# ---------------------------------------------------------------------------
# Uniformité de sortie
# ---------------------------------------------------------------------------
def test_fit_predict_columns(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert set(out.columns) == {"regime", "confidence", "shannon_value"}
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


def test_neutre_has_half_confidence(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "NEUTRE"
    if mask.any():
        assert (out.loc[mask, "confidence"] == 0.5).all()


# ---------------------------------------------------------------------------
# Cohérence de la mesure
# ---------------------------------------------------------------------------
def test_shannon_value_bounds(ohlcv_h1):
    """La mesure est normalisée dans [0, 1]."""
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    valid = out["shannon_value"].dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()


def test_shannon_value_nan_on_unknown(ohlcv_h1):
    """Les barres INCONNU ont shannon_value = NaN."""
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "INCONNU"
    assert out.loc[mask, "shannon_value"].isna().all()


# ---------------------------------------------------------------------------
# Comportement sur séries contrôlées
# ---------------------------------------------------------------------------
def test_shannon_low_on_perfect_trend():
    """
    Rendements constants (avec bruit de flottant) → distribution
    dégénérée → entropie ≈ 0 grâce à la tolérance numérique.
    """
    n = 500
    close = 1.10 * np.exp(np.cumsum(np.full(n, 1e-4)))
    df = _make_ohlcv(close)

    det = _detector(fenetre=50, n_bins=10)
    out = det.fit_predict(df)

    valid = out["shannon_value"].iloc[100:].dropna()
    assert len(valid) > 0
    assert valid.mean() < 0.01, (
        f"Entropie moyenne = {valid.mean():.6f} (attendu < 0.01)"
    )


def test_shannon_high_on_random_returns():
    """
    Rendements gaussiens i.i.d. → distribution proche de l'uniforme
    après discrétisation → entropie élevée.
    """
    rng = np.random.default_rng(42)
    n = 1000
    returns = rng.normal(0, 0.001, n)
    close = 1.10 * np.exp(np.cumsum(returns))
    df = _make_ohlcv(close)

    det = _detector(fenetre=100, n_bins=10)
    out = det.fit_predict(df)

    valid = out["shannon_value"].iloc[200:].dropna()
    assert len(valid) > 0
    assert valid.mean() > 0.5, (
        f"Entropie moyenne = {valid.mean():.3f} (attendu > 0.5)"
    )


def test_shannon_ordered_on_clean_trend():
    """
    Contraste net : rendements constants (entropie ≈ 0) vs rendements
    uniformes (entropie ≈ 1). Teste la capacité discriminante de la
    mesure sur deux cas extrêmes.

    Note : deux gaussiennes de variances différentes ont des entropies
    normalisées proches. L'entropie normalisée mesure la forme de la
    distribution, pas son échelle.
    """
    rng = np.random.default_rng(0)
    n = 800

    # Série parfaitement ordonnée : rendements constants
    returns_calm = np.full(n, 1e-4)
    close_calm = 1.10 * np.exp(np.cumsum(returns_calm))

    # Série parfaitement chaotique : rendements uniformes
    returns_wild = rng.uniform(-5e-4, 5e-4, n)
    close_wild = 1.10 * np.exp(np.cumsum(returns_wild))

    det = _detector(fenetre=100, n_bins=10)

    out_calm = det.fit_predict(_make_ohlcv(close_calm))
    out_wild = det.fit_predict(_make_ohlcv(close_wild))

    h_calm = out_calm["shannon_value"].iloc[200:].dropna().mean()
    h_wild = out_wild["shannon_value"].iloc[200:].dropna().mean()

    assert h_calm < h_wild, (
        f"Entropie calme ({h_calm:.4f}) devrait être < "
        f"entropie chaotique ({h_wild:.4f})"
    )
    assert h_wild - h_calm > 0.5, (
        f"Écart trop faible : {h_wild - h_calm:.4f} (attendu > 0.5)"
    )
    assert h_calm < 0.05, (
        f"Entropie calme = {h_calm:.4f} (attendu < 0.05)"
    )
    assert h_wild > 0.85, (
        f"Entropie chaotique = {h_wild:.4f} (attendu > 0.85)"
    )


# ---------------------------------------------------------------------------
# Causalité
# ---------------------------------------------------------------------------
def test_no_future_leak_by_truncation(ohlcv_h1):
    """Tronquer la série ne change pas les régimes passés."""
    det = _detector()
    full = det.fit_predict(ohlcv_h1)["regime"]
    k = len(ohlcv_h1) // 2
    truncated = det.fit_predict(ohlcv_h1.iloc[:k])["regime"]
    pd.testing.assert_series_equal(
        full.iloc[:k].reset_index(drop=True),
        truncated.reset_index(drop=True),
        check_names=False,
    )


def test_window_is_strictly_past():
    """
    Le shannon_value[t] ne dépend que des rendements [t-fenetre+1, t].
    Modifier un rendement en t+1 ne doit pas changer shannon_value[t].
    """
    n = 500
    rng = np.random.default_rng(7)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    df = _make_ohlcv(close)

    i_test = 300
    det = _detector(fenetre=50)
    out_orig = det.fit_predict(df)
    val_orig = out_orig["shannon_value"].iloc[i_test]

    # Perturber close[i_test + 1] (futur immédiat)
    df_mod = df.copy()
    df_mod.iloc[i_test + 1, df_mod.columns.get_loc("close")] *= 1.10
    # La perturbation reste une barre OHLC cohérente.
    df_mod.iloc[i_test + 1, df_mod.columns.get_loc("high")] = max(
        df_mod["high"].iloc[i_test + 1], df_mod["close"].iloc[i_test + 1]
    )
    out_mod = det.fit_predict(df_mod)
    val_mod = out_mod["shannon_value"].iloc[i_test]

    assert np.isclose(val_orig, val_mod, rtol=1e-10), (
        f"shannon_value[{i_test}] change malgré perturbation du futur : "
        f"orig={val_orig}, mod={val_mod}"
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
    assert pd.isna(out["shannon_value"].iloc[-1])


def test_unaligned_timestamps_are_unknown():
    index = pd.DatetimeIndex([
        "2024-01-01 00:00:00+00:00",
        "2024-01-01 01:00:00+00:00",
        "2024-01-01 02:00:00+00:00",
        "2024-01-01 03:00:00+00:00",
        "2024-01-01 04:00:00+00:00",
        "2024-01-01 05:30:00+00:00",
    ], name="timestamp")
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
    assert set(out.columns) == {"regime", "confidence", "shannon_value"}


def test_warmup_marks_unknown(ohlcv_h1):
    """
    La chauffe dure `fenetre - 1` barres (indices 0 à fenetre-2).
    """
    det = _detector(fenetre=100)
    out = det.fit_predict(ohlcv_h1)
    assert (out["regime"].iloc[:99] == "INCONNU").all()
    assert out["regime"].iloc[99] != "INCONNU", (
        "La barre 99 devrait déjà être classifiée"
    )


def test_constant_close_gives_zero_entropy():
    """Série strictement plate → entropie = 0 (distribution dégénérée)."""
    n = 200
    close = np.full(n, 1.10)
    df = _make_ohlcv(close)
    det = _detector(fenetre=50)
    out = det.fit_predict(df)
    valid = out["shannon_value"].iloc[100:].dropna()
    if len(valid) > 0:
        assert (valid == 0.0).all(), (
            f"Entropie non nulle sur série plate : {valid.unique()[:5]}"
        )


# ---------------------------------------------------------------------------
# Fonction interne _shannon_windowed
# ---------------------------------------------------------------------------
def test_shannon_windowed_basic():
    """Test direct de la fonction de calcul."""
    n = 200
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.001, n)

    h = _shannon_windowed(returns, fenetre=50, n_bins=10)
    # NaN sur les 49 premières
    assert np.isnan(h[:49]).all()
    assert not np.isnan(h[49:]).any()
    # Bornes
    assert (h[49:] >= 0).all()
    assert (h[49:] <= 1).all()


# ---------------------------------------------------------------------------
# Validations des paramètres
# ---------------------------------------------------------------------------
def test_rejects_invalid_fenetre():
    with pytest.raises(ValueError, match="fenetre"):
        ShannonEntropyDetector(timeframe="H1", fenetre=5)


def test_rejects_invalid_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        ShannonEntropyDetector(timeframe="H1", n_bins=1)


def test_rejects_invalid_seuils():
    with pytest.raises(ValueError, match="seuil_ordonne < seuil_desordre"):
        ShannonEntropyDetector(
            timeframe="H1", seuil_ordonne=0.9, seuil_desordre=0.5
        )


def test_rejects_seuils_out_of_range():
    with pytest.raises(ValueError, match="seuil_ordonne < seuil_desordre"):
        ShannonEntropyDetector(
            timeframe="H1", seuil_ordonne=0.0, seuil_desordre=0.5
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
        mask = ShannonEntropyDetector._unaligned_mask(index, "H1")
        assert not mask.any(), (
            f"Résolution {unit} : {int(mask.sum())} barres hors grille"
        )
