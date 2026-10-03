"""
Tests spécifiques au détecteur PCA axis.

Vérifie :
  - l'uniformité de sortie (regime + confidence + mesures) ;
  - la cohérence mathématique (direction ∈ [-1, 1], ratios ∈ [0, 1]) ;
  - le comportement sur séries contrôlées (tendance → HAUSSIER/BAISSIER) ;
  - la causalité (la fenêtre ne contient que le passé) ;
  - le marquage `is_partial` et hors grille ;
  - les cas dégénérés ;
  - les validations de paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.vector.pca_axis import (
    PCAAxisDetector,
    _construire_features,
    _pca_windowed,
)


def _detector(**kwargs) -> PCAAxisDetector:
    return PCAAxisDetector(timeframe="H1", **kwargs)


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
    assert set(out.columns) == {
        "regime", "confidence",
        "pca_direction", "pca_eigen_ratio", "pca_variance_ratio",
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
# Cohérence des mesures
# ---------------------------------------------------------------------------
def test_direction_bounds(ohlcv_h1):
    """pca_direction ∈ [-1, 1] par construction."""
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    valid = out["pca_direction"].dropna()
    assert (valid >= -1.0).all()
    assert (valid <= 1.0).all()


def test_eigen_ratio_bounds(ohlcv_h1):
    """λ1 / Σλ ∈ [0.5, 1] pour deux dimensions."""
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    valid = out["pca_eigen_ratio"].dropna()
    assert (valid >= 0.0).all()
    assert (valid <= 1.0).all()


def test_variance_ratio_bounds(ohlcv_h1):
    """Pour deux dimensions, ce ratio égale pca_eigen_ratio."""
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    valid = out["pca_variance_ratio"].dropna()
    assert (valid >= 0.5 - 1e-10).all(), (
        f"variance_ratio min = {valid.min():.6f} (attendu >= 0.5)"
    )
    assert (valid <= 1.0).all()
    np.testing.assert_allclose(
        valid.to_numpy(),
        out.loc[valid.index, "pca_eigen_ratio"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Comportement sur séries contrôlées
# ---------------------------------------------------------------------------
def test_uptrend_yields_more_haussier():
    """
    Série bruitée avec drift positif → HAUSSIER > BAISSIER.

    Note : une série à rendements strictement constants rend la matrice
    de covariance singulière (vol = 0), ce qui produit une classification
    arbitraire. On utilise donc une série bruitée réaliste.
    """
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(2e-4, 1e-4, n)  # drift positif + bruit
    close = 1.10 * np.exp(np.cumsum(returns))
    df = _make_ohlcv(close)

    det = _detector(
        fenetre=100, seuil_direction=0.3, seuil_eigen_ratio=0.3,
    )
    out = det.fit_predict(df)

    tail = out.iloc[200:]
    pct_haussier = (tail["regime"] == "HAUSSIER").mean()
    pct_baissier = (tail["regime"] == "BAISSIER").mean()

    assert pct_haussier > pct_baissier, (
        f"HAUSSIER ({pct_haussier*100:.1f} %) devrait être > "
        f"BAISSIER ({pct_baissier*100:.1f} %)"
    )
    assert pct_haussier > 0.4, (
        f"HAUSSIER = {pct_haussier*100:.1f} % (attendu > 40 %)"
    )


def test_price_drift_determines_direction_not_candle_features():
    """Un drift net doit orienter la PCA même avec des bougies identiques."""
    n = 300
    close = 1.10 * np.exp(np.linspace(0.0, 0.1, n))
    df = _make_ohlcv(close)

    out = _detector(fenetre=50).fit_predict(df)
    tail = out.iloc[100:]

    assert (tail["regime"] == "HAUSSIER").all()
    assert (tail["pca_direction"] > 0.99).all()


def test_downtrend_yields_more_baissier():
    """
    Série bruitée avec drift négatif → BAISSIER > HAUSSIER.
    """
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(-2e-4, 1e-4, n)  # drift négatif + bruit
    close = 1.10 * np.exp(np.cumsum(returns))
    df = _make_ohlcv(close)

    det = _detector(
        fenetre=100, seuil_direction=0.3, seuil_eigen_ratio=0.3,
    )
    out = det.fit_predict(df)

    tail = out.iloc[200:]
    pct_haussier = (tail["regime"] == "HAUSSIER").mean()
    pct_baissier = (tail["regime"] == "BAISSIER").mean()

    assert pct_baissier > pct_haussier, (
        f"BAISSIER ({pct_baissier*100:.1f} %) devrait être > "
        f"HAUSSIER ({pct_haussier*100:.1f} %)"
    )
    assert pct_baissier > 0.4, (
        f"BAISSIER = {pct_baissier*100:.1f} % (attendu > 40 %)"
    )


def test_flat_series_yields_neutre_or_inconnu():
    """
    Série plate → NEUTRE ou INCONNU (matrice de covariance singulière).
    """
    n = 500
    close = np.full(n, 1.10)
    df = _make_ohlcv(close)

    det = _detector(fenetre=100, seuil_direction=0.3)
    out = det.fit_predict(df)

    tail = out.iloc[200:]
    pct_neutre = (tail["regime"] == "NEUTRE").mean()
    pct_inconnu = (tail["regime"] == "INCONNU").mean()
    assert pct_neutre + pct_inconnu > 0.8, (
        f"NEUTRE+INCONNU = {(pct_neutre+pct_inconnu)*100:.1f} % "
        f"(attendu > 80 %)"
    )


# ---------------------------------------------------------------------------
# Causalité
# ---------------------------------------------------------------------------
def test_no_future_leak_by_truncation(ohlcv_h1):
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
    La fenêtre à t contient les barres [t-fenetre, t-1].
    Modifier x_{t+1} ne doit rien changer pour la classification à t.
    """
    n = 500
    rng = np.random.default_rng(7)
    close = 1.10 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    df = _make_ohlcv(close)

    det = _detector(fenetre=50)
    out_orig = det.fit_predict(df)

    i_test = 300
    dir_orig = out_orig["pca_direction"].iloc[i_test]
    eig_orig = out_orig["pca_eigen_ratio"].iloc[i_test]

    # Perturber close[i_test + 1]
    df_mod = df.copy()
    df_mod.iloc[i_test + 1, df_mod.columns.get_loc("close")] *= 1.20
    out_mod = det.fit_predict(df_mod)

    dir_mod = out_mod["pca_direction"].iloc[i_test]
    eig_mod = out_mod["pca_eigen_ratio"].iloc[i_test]

    assert np.isclose(dir_orig, dir_mod, rtol=1e-8, equal_nan=True), (
        f"pca_direction[{i_test}] change : orig={dir_orig}, mod={dir_mod}"
    )
    assert np.isclose(eig_orig, eig_mod, rtol=1e-8, equal_nan=True), (
        f"pca_eigen_ratio[{i_test}] change : "
        f"orig={eig_orig}, mod={eig_mod}"
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
    assert pd.isna(out["pca_direction"].iloc[-1])


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
    assert set(out.columns) == {
        "regime", "confidence",
        "pca_direction", "pca_eigen_ratio", "pca_variance_ratio",
    }


def test_warmup_marks_unknown(ohlcv_h1):
    """
    La chauffe PCA dure `fenetre - 1` barres (indices 0 à fenetre-2).
    Les features ont leur propre chauffe (vol rolling 20) mais elle est
    incluse dans la fenêtre PCA.
    """
    det = _detector(fenetre=100)
    out = det.fit_predict(ohlcv_h1)
    # Chauffe : fenetre - 1 = 99 barres
    assert (out["regime"].iloc[:99] == "INCONNU").all()
    # À partir de l'indice 99, des classifications apparaissent
    assert (out["regime"].iloc[99:120] != "INCONNU").any(), (
        "Aucune classification après la chauffe attendue"
    )


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------
def test_pca_windowed_basic():
    """Test direct de _pca_windowed."""
    n, d = 200, 2
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n, d))

    direction, eig_ratio, var_ratio = _pca_windowed(X, fenetre=50)
    # NaN sur les 50 premières
    assert np.isnan(direction[:50]).all()
    # Après : valeurs cohérentes
    valid_d = direction[50:][~np.isnan(direction[50:])]
    assert (valid_d >= -1.0).all() and (valid_d <= 1.0).all()


def test_construire_features_shape(ohlcv_h1):
    """_construire_features retourne les coordonnées temps et log-prix."""
    X = _construire_features(ohlcv_h1)
    assert X.shape == (len(ohlcv_h1), 2)
    np.testing.assert_array_equal(X[:, 0], np.arange(len(ohlcv_h1)))
    np.testing.assert_allclose(X[:, 1], np.log(ohlcv_h1["close"]))


# ---------------------------------------------------------------------------
# Validations des paramètres
# ---------------------------------------------------------------------------
def test_rejects_invalid_fenetre():
    with pytest.raises(ValueError, match="fenetre"):
        PCAAxisDetector(timeframe="H1", fenetre=5)


def test_rejects_invalid_seuil_direction():
    with pytest.raises(ValueError, match="seuil_direction"):
        PCAAxisDetector(timeframe="H1", seuil_direction=1.5)
    with pytest.raises(ValueError, match="seuil_direction"):
        PCAAxisDetector(timeframe="H1", seuil_direction=0.0)


def test_rejects_invalid_seuil_eigen_ratio():
    with pytest.raises(ValueError, match="seuil_eigen_ratio"):
        PCAAxisDetector(timeframe="H1", seuil_eigen_ratio=1.5)
    with pytest.raises(ValueError, match="seuil_eigen_ratio"):
        PCAAxisDetector(timeframe="H1", seuil_eigen_ratio=0.0)


# ---------------------------------------------------------------------------
# Non-régression : masque unaligned
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
        mask = PCAAxisDetector._unaligned_mask(index, "H1")
        assert not mask.any(), (
            f"Résolution {unit} : {int(mask.sum())} barres hors grille"
        )
