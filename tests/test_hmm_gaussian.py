"""
Tests spécifiques au détecteur HMM gaussien (hmmlearn).

Vérifie :
  - le caractère offline (requires_lookahead = True) ;
  - la construction dynamique de REGIME_MAP selon n_states ;
  - l'uniformité de la sortie (colonnes, types, index) ;
  - le déterminisme des états et de la confidence (seed) ;
  - le comportement des features et de la normalisation ;
  - la formule de confidence à partir des posteriors ;
  - le marquage `is_partial` et hors grille ;
  - les cas limites (DataFrame vide, série trop courte) ;
  - les validations des hyperparamètres ;
  - l'insensibilité de la détection hors grille à la résolution de l'index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.statistical.hmm_gaussian import (
    HMMGaussianDetector,
    _confidence_from_posteriors,
    _construire_features,
    _normaliser,
)


def _detector(**kwargs) -> HMMGaussianDetector:
    """Instancie un HMMGaussianDetector avec allow_lookahead=True."""
    return HMMGaussianDetector(timeframe="H1", allow_lookahead=True, **kwargs)


# ---------------------------------------------------------------------------
# Caractère offline & lookahead
# ---------------------------------------------------------------------------
def test_requires_lookahead_refuses_without_flag():
    """Le HMM doit refuser de s'instancier sans allow_lookahead=True."""
    with pytest.raises(ValueError, match="exige le lookahead"):
        HMMGaussianDetector(timeframe="H1")


def test_requires_lookahead_accepts_with_flag():
    det = HMMGaussianDetector(timeframe="H1", allow_lookahead=True)
    assert det.requires_lookahead is True
    assert det.allow_lookahead is True


# ---------------------------------------------------------------------------
# Construction dynamique de REGIME_MAP
# ---------------------------------------------------------------------------
def test_regime_map_dynamic_two_states():
    det = _detector(n_states=2)
    assert det.REGIME_MAP == {
        "STATE_0": 0,
        "STATE_1": 1,
        "INCONNU": 2,
    }


def test_regime_map_dynamic_three_states():
    det = _detector(n_states=3)
    assert det.REGIME_MAP == {
        "STATE_0": 0,
        "STATE_1": 1,
        "STATE_2": 2,
        "INCONNU": 3,
    }


def test_regime_map_dynamic_four_states():
    det = _detector(n_states=4)
    assert det.REGIME_MAP == {
        "STATE_0": 0,
        "STATE_1": 1,
        "STATE_2": 2,
        "STATE_3": 3,
        "INCONNU": 4,
    }


# ---------------------------------------------------------------------------
# Helpers internes (features, normalisation, confidence)
# ---------------------------------------------------------------------------
def test_construire_features(ohlcv_h1):
    feats = _construire_features(ohlcv_h1)
    expected_cols = {"log_return", "vol_realisee", "range_norm", "close_position"}
    assert set(feats.columns) == expected_cols
    assert not feats.isna().any().any()
    # close_position doit être borné dans [0, 1]
    assert feats["close_position"].between(0.0, 1.0).all()


def test_normaliser(ohlcv_h1):
    feats = _construire_features(ohlcv_h1)
    norm = _normaliser(feats)
    # Les colonnes normalisées doivent avoir une moyenne proche de 0 et std proche de 1
    for col in norm.columns:
        assert np.isclose(norm[col].mean(), 0.0, atol=1e-7)
        assert np.isclose(norm[col].std(), 1.0, atol=1e-7)


def test_confidence_from_posteriors():
    # Cas 1: 2 états, équiprobable (0.5, 0.5) -> confidence minimale = 0.5
    post_uniform = np.array([[0.5, 0.5]])
    conf = _confidence_from_posteriors(post_uniform, n_states=2)
    assert np.isclose(conf[0], 0.5)

    # Cas 2: 2 états, certitude absolue (1.0, 0.0) -> confidence maximale = 1.0
    post_certain = np.array([[1.0, 0.0]])
    conf = _confidence_from_posteriors(post_certain, n_states=2)
    assert np.isclose(conf[0], 1.0)

    # Cas 3: 3 états, équiprobable (1/3, 1/3, 1/3) -> 0.5
    post_3_uniform = np.array([[1.0 / 3, 1.0 / 3, 1.0 / 3]])
    conf = _confidence_from_posteriors(post_3_uniform, n_states=3)
    assert np.isclose(conf[0], 0.5)

    # Cas 4: 1 état (cas dégénéré) -> 1.0
    post_1 = np.array([[1.0]])
    conf = _confidence_from_posteriors(post_1, n_states=1)
    assert np.isclose(conf[0], 1.0)


# ---------------------------------------------------------------------------
# Contrat de sortie et régimes
# ---------------------------------------------------------------------------
def test_fit_predict_returns_regime_and_confidence(ohlcv_h1):
    det = _detector(n_states=2)
    out = det.fit_predict(ohlcv_h1)
    assert set(out.columns) == {"regime", "confidence"}
    assert len(out) == len(ohlcv_h1)
    assert out["confidence"].between(0.0, 1.0).all()
    assert out["confidence"].notna().all()


def test_regime_map_covers_all_labels(ohlcv_h1):
    det = _detector(n_states=2)
    out = det.fit_predict(ohlcv_h1)
    unique_regimes = set(out["regime"].unique())
    assert unique_regimes.issubset(set(det.REGIME_MAP.keys()))
    # En pratique, les états découverts doivent être présents
    assert "STATE_0" in unique_regimes or "STATE_1" in unique_regimes


def test_warmup_produces_unknown(ohlcv_h1):
    """Les barres de warmup pour les features doivent être INCONNU avec confidence 0.0."""
    det = _detector(n_states=2)
    out = det.fit_predict(ohlcv_h1)
    # Les premières barres (~20 barres de rolling) sont INCONNU
    assert out["regime"].iloc[0] == "INCONNU"
    assert out["confidence"].iloc[0] == 0.0


def test_unknown_confidence_is_zero(ohlcv_h1):
    det = _detector(n_states=2)
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"] == "INCONNU"
    assert (out.loc[mask, "confidence"] == 0.0).all()


def test_valid_states_confidence_bounds(ohlcv_h1):
    det = _detector(n_states=2)
    out = det.fit_predict(ohlcv_h1)
    valid_mask = out["regime"] != "INCONNU"
    # Les états classés ont une confidence >= 0.5
    assert (out.loc[valid_mask, "confidence"] >= 0.5).all()


def test_determinism_with_fixed_seed(ohlcv_h1):
    """Deux exécutions avec la même graine doivent donner un résultat identique."""
    det1 = _detector(n_states=2, seed=42)
    det2 = _detector(n_states=2, seed=42)

    out1 = det1.fit_predict(ohlcv_h1)
    out2 = det2.fit_predict(ohlcv_h1)

    pd.testing.assert_series_equal(out1["regime"], out2["regime"])
    pd.testing.assert_series_equal(out1["confidence"], out2["confidence"])


# ---------------------------------------------------------------------------
# Bougies non clôturées & alignement
# ---------------------------------------------------------------------------
def test_last_bar_is_unknown_when_partial(ohlcv_h1):
    df = ohlcv_h1.copy()
    df["is_partial"] = False
    df.iloc[-1, df.columns.get_loc("is_partial")] = True

    det = _detector(n_states=2)
    out = det.fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"
    assert out["confidence"].iloc[-1] == 0.0


def test_unaligned_timestamps_are_unknown(ohlcv_h1):
    df = ohlcv_h1.copy()
    new_index = list(df.index[:100])
    new_index[-1] = pd.Timestamp("2024-01-05 03:30:00+00:00")  # minute 30 hors grille H1
    df = df.iloc[:100].copy()
    df.index = pd.DatetimeIndex(new_index, name="timestamp")

    det = _detector(n_states=2)
    out = det.fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"
    assert out["confidence"].iloc[-1] == 0.0


# ---------------------------------------------------------------------------
# Cas limites & validations
# ---------------------------------------------------------------------------
def test_empty_dataframe():
    det = _detector(n_states=2)
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="timestamp"),
    )
    out = det.fit_predict(empty)
    assert out.empty
    assert set(out.columns) == {"regime", "confidence"}


def test_short_series_produces_all_unknown():
    """Série trop courte pour calculer les features (ex: 5 barres) -> tout INCONNU."""
    n = 5
    index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp")
    df = pd.DataFrame(
        {
            "open": np.full(n, 1.10),
            "high": np.full(n, 1.11),
            "low": np.full(n, 1.09),
            "close": np.full(n, 1.10),
            "volume": np.full(n, 100.0),
        },
        index=index,
    )
    det = _detector(n_states=2)
    out = det.fit_predict(df)
    assert (out["regime"] == "INCONNU").all()
    assert (out["confidence"] == 0.0).all()


def test_rejects_invalid_n_states():
    with pytest.raises(ValueError, match="n_states"):
        HMMGaussianDetector(allow_lookahead=True, n_states=0)

    with pytest.raises(ValueError, match="n_states"):
        HMMGaussianDetector(allow_lookahead=True, n_states=-2)

    with pytest.raises(ValueError, match="n_states"):
        HMMGaussianDetector(allow_lookahead=True, n_states="2")  # type: ignore


def test_rejects_invalid_n_iter():
    with pytest.raises(ValueError, match="n_iter"):
        HMMGaussianDetector(allow_lookahead=True, n_iter=0)

    with pytest.raises(ValueError, match="n_iter"):
        HMMGaussianDetector(allow_lookahead=True, n_iter=-10)


def test_unaligned_mask_agnostic_to_index_resolution():
    n = 100
    for unit in ("s", "ms", "us", "ns"):
        index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        try:
            index = index.as_unit(unit, round_ok=True)
        except (AttributeError, TypeError):
            pytest.skip(f"as_unit({unit}) non supporté sur cette version")
        index.name = "timestamp"
        mask = HMMGaussianDetector._unaligned_mask(index, "H1")
        assert not mask.any(), f"Résolution {unit} : fausses barres hors grille détectées."
