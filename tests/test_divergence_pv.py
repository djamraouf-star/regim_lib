"""
Tests du détecteur de divergence prix / volume.

Vérifie :
  - le refus si les colonnes requises sont absentes ;
  - l'uniformité de sortie ;
  - la cohérence des nouveaux extrêmes (is_new_high, is_new_low) ;
  - la classification correcte selon volume fort/faible ;
  - la causalité (non-fuite du futur) ;
  - le marquage `is_partial` ;
  - les cas dégénérés ;
  - les validations de paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.volume.divergence_pv import DivergencePVDetector


def _detector(**kwargs) -> DivergencePVDetector:
    return DivergencePVDetector(timeframe="H1", **kwargs)


def _make_ohlcv(
    close: np.ndarray,
    volume: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(close)
    idx = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    if volume is None:
        volume = np.full(n, 100.0)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close),
        "low": np.minimum(open_, close),
        "close": close,
        "volume": volume,
    }, index=idx)


# ---------------------------------------------------------------------------
# Colonnes requises
# ---------------------------------------------------------------------------
def test_refus_si_colonnes_absentes():
    """Sans close ni volume, le détecteur refuse."""
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.0]},
        index=pd.DatetimeIndex(["2024-01-01"], name="timestamp"),
    )
    with pytest.raises(ValueError, match="volume"):
        _detector().fit_predict(df)


# ---------------------------------------------------------------------------
# Uniformité
# ---------------------------------------------------------------------------
def test_fit_predict_columns():
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv(close)
    out = _detector().fit_predict(df)
    assert set(out.columns) == {
        "regime", "confidence", "volume_rel",
        "is_new_high", "is_new_low",
    }
    assert len(out) == n


def test_regime_map_couvre_labels():
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv(close)
    out = _detector().fit_predict(df)
    assert set(out["regime"].unique()).issubset(
        set(_detector().REGIME_MAP.keys())
    )


def test_confidence_bounds():
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv(close)
    out = _detector().fit_predict(df)
    assert out["confidence"].between(0.0, 1.0).all()
    assert out["confidence"].notna().all()


# ---------------------------------------------------------------------------
# Comportement sur séries contrôlées
# ---------------------------------------------------------------------------
def test_no_extremes_donnent_neutre():
    """
    Une série plate : ni nouveaux hauts, ni nouveaux bas
    (inégalités strictes). Toutes les barres après chauffe sont NEUTRE.
    """
    n = 200
    close = np.full(n, 1.10)
    df = _make_ohlcv(close)
    out = _detector().fit_predict(df)

    tail = out.iloc[50:]
    assert (tail["regime"] == "NEUTRE").all()


def test_nouveau_haut_volume_fort_confirmation():
    """
    Nouveau haut + volume fort → CONFIRMATION_HAUSSIERE.

    Le volume fort est défini **relativement à la médiane passée**.
    Pour qu'il soit détecté comme fort, la fenêtre médiane doit
    contenir majoritairement des valeurs faibles. On utilise donc :
      - une longue phase calme (150 barres à 30) ;
      - une courte montée (50 barres à 500) ;
      - une fenêtre médiane de 100 barres.

    Les premières barres de la montée ont une médiane tirée par les
    barres calmes encore présentes dans la fenêtre.
    """
    n = 200
    close = np.concatenate([
        np.full(150, 1.10),                    # calme
        np.linspace(1.10, 1.15, 50),           # montée
    ])
    volume = np.concatenate([
        np.full(150, 30.0),                    # volume faible
        np.full(50, 500.0),                    # volume fort
    ])
    df = _make_ohlcv(close, volume)
    out = _detector(
        fenetre_extreme=10,
        fenetre_median_vol=100,
        ratio_volume_fort=1.5,
    ).fit_predict(df)

    n_conf = (out["regime"] == "CONFIRMATION_HAUSSIERE").sum()
    assert n_conf >= 5, f"Attendu >=5 confirmations, reçu {n_conf}"


def test_nouveau_haut_volume_faible_divergence():
    """
    Nouveau haut + volume faible → DIVERGENCE_HAUSSIERE.

    Symétrique du test précédent : phase initiale à volume FORT,
    puis montée à volume FAIBLE. La médiane reste haute pendant
    les premières barres de la montée, donc le volume faible y est
    détecté comme faible.
    """
    n = 200
    close = np.concatenate([
        np.full(150, 1.10),
        np.linspace(1.10, 1.15, 50),
    ])
    volume = np.concatenate([
        np.full(150, 500.0),                   # volume fort
        np.full(50, 30.0),                     # volume faible
    ])
    df = _make_ohlcv(close, volume)
    out = _detector(
        fenetre_extreme=10,
        fenetre_median_vol=100,
        ratio_volume_fort=1.5,
    ).fit_predict(df)

    n_div = (out["regime"] == "DIVERGENCE_HAUSSIERE").sum()
    assert n_div >= 5, f"Attendu >=5 divergences, reçu {n_div}"


def test_nouveau_bas_volume_faible_divergence():
    """Nouveau bas + volume faible → DIVERGENCE_BAISSIERE."""
    n = 200
    close = np.concatenate([
        np.full(150, 1.15),
        np.linspace(1.15, 1.10, 50),           # descente
    ])
    volume = np.concatenate([
        np.full(150, 500.0),
        np.full(50, 30.0),
    ])
    df = _make_ohlcv(close, volume)
    out = _detector(
        fenetre_extreme=10,
        fenetre_median_vol=100,
        ratio_volume_fort=1.5,
    ).fit_predict(df)

    n_div = (out["regime"] == "DIVERGENCE_BAISSIERE").sum()
    assert n_div >= 5, f"Attendu >=5 divergences, reçu {n_div}"


# ---------------------------------------------------------------------------
# Cohérence des mesures
# ---------------------------------------------------------------------------
def test_is_new_high_low_coherents():
    """
    Une barre ne peut pas être simultanément nouveau haut et nouveau bas.
    """
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv(close)
    out = _detector().fit_predict(df)
    both = out["is_new_high"] & out["is_new_low"]
    assert not both.any()


def test_volume_rel_positif():
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv(close)
    out = _detector().fit_predict(df)
    valid = out["volume_rel"].dropna()
    assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# Causalité
# ---------------------------------------------------------------------------
def test_no_future_leak_by_truncation():
    n = 500
    rng = np.random.default_rng(0)
    close = 1.10 + np.cumsum(rng.normal(0, 1e-4, n))
    volume = rng.uniform(50, 200, n)
    df = _make_ohlcv(close, volume)

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
    df = _make_ohlcv(close)
    df["is_partial"] = False
    df.iloc[-1, df.columns.get_loc("is_partial")] = True

    out = _detector().fit_predict(df)
    assert out["regime"].iloc[-1] == "INCONNU"
    assert out["confidence"].iloc[-1] == 0.0


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
        "regime", "confidence", "volume_rel",
        "is_new_high", "is_new_low",
    }


def test_warmup_marque_inconnu():
    """Les premières barres (chauffe) sont INCONNU."""
    n = 200
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    df = _make_ohlcv(close)
    det = _detector(fenetre_extreme=20, fenetre_median_vol=50)
    out = det.fit_predict(df)
    # Au moins les fenetre_median_vol - 1 premières barres sont INCONNU
    assert (out["regime"].iloc[:25] == "INCONNU").all()


# ---------------------------------------------------------------------------
# Validations paramètres
# ---------------------------------------------------------------------------
def test_rejects_invalid_fenetre_extreme():
    with pytest.raises(ValueError, match="fenetre_extreme"):
        DivergencePVDetector(timeframe="H1", fenetre_extreme=1)


def test_rejects_invalid_fenetre_median_vol():
    with pytest.raises(ValueError, match="fenetre_median_vol"):
        DivergencePVDetector(timeframe="H1", fenetre_median_vol=1)


def test_rejects_invalid_ratio():
    with pytest.raises(ValueError, match="ratio_volume_fort"):
        DivergencePVDetector(timeframe="H1", ratio_volume_fort=0.0)
    with pytest.raises(ValueError, match="ratio_volume_fort"):
        DivergencePVDetector(timeframe="H1", ratio_volume_fort=-1.0)


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
        mask = DivergencePVDetector._unaligned_mask(index, "H1")
        assert not mask.any(), (
            f"Résolution {unit} : {int(mask.sum())} barres hors grille"
        )
