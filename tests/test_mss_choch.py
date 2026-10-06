"""
Tests spécifiques au détecteur MSS/CHOCH.

Vérifie :
  - le caractère offline (requires_lookahead = True) ;
  - la preuve du lookahead par perturbation du futur ;
  - la cohérence de la hiérarchie CHOCH < MSS (confidence) ;
  - le comportement sur séries contrôlées (uptrend/downtrend) ;
  - l'uniformité de sortie (regime + confidence) ;
  - le marquage `is_partial` et hors grille ;
  - les cas dégénérés ;
  - les validations des paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.price.mss_choch import (
    MSSCHOCHDetector,
    _build_alternating_pivots,
    _classify_mss_choch,
    _detect_fractals,
)


def _detector(**kwargs) -> MSSCHOCHDetector:
    return MSSCHOCHDetector(
        timeframe="H1", allow_lookahead=True, **kwargs
    )


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
# Caractère offline
# ---------------------------------------------------------------------------
def test_requires_lookahead_refuses_without_flag():
    with pytest.raises(ValueError, match="exige le lookahead"):
        MSSCHOCHDetector(timeframe="H1")


def test_requires_lookahead_accepts_with_flag():
    det = MSSCHOCHDetector(timeframe="H1", allow_lookahead=True)
    assert det.requires_lookahead is True
    assert det.allow_lookahead is True


def test_future_perturbation_changes_past_classification():
    """
    Preuve DIRECTE du lookahead : modifier une barre FUTURE (i + n) change
    la détection de fractale à i, donc la classification passée.
    """
    # Série contrôlée avec un sommet net à l'indice 5
    high = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 3.0, 2.0, 1.5, 1.0])
    low = high - 0.5
    n_fractale = 2

    is_sommet_orig, _ = _detect_fractals(high, low, n_fractale)
    assert is_sommet_orig[5], "high[5]=4.0 doit être un sommet"

    # Perturber une barre FUTURE (indice 6, qui est à droite de 5)
    high_mod = high.copy()
    high_mod[6] = 5.0  # dépasse maintenant high[5]=4.0

    is_sommet_mod, _ = _detect_fractals(high_mod, low, n_fractale)
    assert not is_sommet_mod[5], (
        "high[5] ne doit plus être un sommet après perturbation de high[6]"
    )


# ---------------------------------------------------------------------------
# Uniformité de sortie
# ---------------------------------------------------------------------------
def test_fit_predict_columns(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    assert set(out.columns) == {"regime", "confidence"}
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
# Hiérarchie sémantique
# ---------------------------------------------------------------------------
def test_choch_confidence_is_point_six(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"].isin(["CHOCH_HAUSSIER", "CHOCH_BAISSIER"])
    if mask.any():
        assert (out.loc[mask, "confidence"] == 0.6).all()


def test_mss_confidence_is_point_eight(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    mask = out["regime"].isin(["MSS_HAUSSIER", "MSS_BAISSIER"])
    if mask.any():
        assert (out.loc[mask, "confidence"] == 0.8).all()


def test_mss_more_confident_than_choch(ohlcv_h1):
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    conf_choch = out.loc[
        out["regime"].str.startswith("CHOCH"), "confidence"
    ]
    conf_mss = out.loc[
        out["regime"].str.startswith("MSS"), "confidence"
    ]
    if len(conf_choch) > 0 and len(conf_mss) > 0:
        assert conf_mss.min() > conf_choch.max(), (
            "MSS doit avoir une confidence strictement supérieure à CHOCH"
        )


# ---------------------------------------------------------------------------
# Comportement sur séries contrôlées
# ---------------------------------------------------------------------------
def test_uptrend_pure_yields_neutre():
    """
    Uptrend strictement monotone → tous les pivots sont alternés sans
    cassure → NEUTRE partout (après chauffe).
    """
    n = 200
    close = 1.10 + 0.001 * np.arange(n)
    df = _make_ohlcv(close)
    det = _detector(n_fractale=3)
    out = det.fit_predict(df)

    # Sur une série monotone, aucun retournement → pas de CHOCH/MSS
    events = out["regime"].str.startswith(("MSS", "CHOCH"))
    assert events.sum() == 0, (
        f"Uptrend monotone ne doit produire aucun événement, "
        f"trouvé : {out.loc[events, 'regime'].value_counts().to_dict()}"
    )


def test_majority_neutre_on_real_data(ohlcv_h1):
    """
    Sur données réelles, la majorité des barres doit être NEUTRE.
    Un retournement est un événement rare.
    """
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    pct_neutre = (out["regime"] == "NEUTRE").mean()
    assert pct_neutre > 0.80, (
        f"NEUTRE = {pct_neutre*100:.1f} % (attendu > 80 %)"
    )


def test_direction_is_symmetric_under_price_reflection(ohlcv_h1):
    """
    Réfléchir les prix autour d'un niveau constant doit inverser les
    labels directionnels sans modifier les autres régimes.
    """
    rng = np.random.default_rng(17)
    returns = rng.integers(-5, 6, size=500) / 256.0
    close = 2.0 + np.cumsum(returns)
    data = _make_ohlcv(close)
    det = _detector()
    original = det.fit_predict(data)["regime"]

    reflected_df = data.copy()
    center = 8.0
    reflected_df["open"] = center - data["open"]
    reflected_df["high"] = center - data["low"]
    reflected_df["low"] = center - data["high"]
    reflected_df["close"] = center - data["close"]
    reflected = det.fit_predict(reflected_df)["regime"]

    reflected_labels = {
        "MSS_HAUSSIER": "MSS_BAISSIER",
        "MSS_BAISSIER": "MSS_HAUSSIER",
        "CHOCH_HAUSSIER": "CHOCH_BAISSIER",
        "CHOCH_BAISSIER": "CHOCH_HAUSSIER",
        "NEUTRE": "NEUTRE",
        "INCONNU": "INCONNU",
    }
    expected = original.map(reflected_labels)
    pd.testing.assert_series_equal(
        reflected.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
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
    assert set(out.columns) == {"regime", "confidence"}


def test_short_series_yields_unknown():
    """
    Série plus courte que la fenêtre de fractale → INCONNU partout.
    """
    n = 5
    close = 1.10 + 0.001 * np.arange(n)
    df = _make_ohlcv(close)
    det = _detector(n_fractale=5)  # besoin de 11 barres minimum
    out = det.fit_predict(df)
    assert (out["regime"] == "INCONNU").all()


# ---------------------------------------------------------------------------
# Fonctions internes
# ---------------------------------------------------------------------------
def test_detect_fractals_simple():
    high = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    low = high - 0.5
    s, c = _detect_fractals(high, low, n_fractale=2)
    assert s[2], "high[2]=3.0 doit être un sommet"
    assert s[6], "high[6]=3.0 doit être un sommet"


def test_build_alternating_pivots():
    high = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0])
    low = high - 0.5
    is_sommet = np.array([False, True, False, True, False, True, False])
    is_creux = np.array([False, False, True, False, True, False, True])
    pivots = _build_alternating_pivots(is_sommet, is_creux, high, low)
    # Doit alterner S/C
    types = [p[1] for p in pivots]
    for i in range(1, len(types)):
        assert types[i] != types[i - 1], (
            f"Pivots non alternés : {types}"
        )


def test_classify_mss_choch_basic():
    """Test direct sur un cas construit."""
    # Uptrend puis cassure du dernier creux
    high = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5])
    low = high - 0.3
    close = np.array([1.0, 2.0, 3.0, 2.5, 2.0, 1.5, 0.8, 0.3])
    # Pivots : S à 2 (3.0), C à 5 (1.7) - hmm structure
    is_sommet = np.array([False, False, True, False, False, False, False, False])
    is_creux = np.array([False, False, False, False, False, True, False, False])
    pivots = _build_alternating_pivots(is_sommet, is_creux, high, low)
    # Ajouter manuellement des pivots pour tester la classification
    # Structure simple : on vérifie juste que la fonction retourne un array
    regime = _classify_mss_choch(pivots, close, len(high))
    assert len(regime) == len(high)


# ---------------------------------------------------------------------------
# Validations des paramètres
# ---------------------------------------------------------------------------
def test_rejects_invalid_n_fractale():
    with pytest.raises(ValueError, match="n_fractale"):
        MSSCHOCHDetector(
            timeframe="H1", allow_lookahead=True, n_fractale=0
        )


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
        mask = MSSCHOCHDetector._unaligned_mask(index, "H1")
        assert not mask.any(), (
            f"Résolution {unit} : {int(mask.sum())} barres hors grille"
        )
