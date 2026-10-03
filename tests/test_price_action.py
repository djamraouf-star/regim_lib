"""
Tests spécifiques au détecteur Price Action.

Vérifie :
  - le caractère offline (requires_lookahead = True) ;
  - la dépendance structurelle au futur (test de perturbation) ;
  - la stabilité des classifications passées sous troncature (documente
    le caractère "lookahead de latence" et non "de valeurs") ;
  - l'équivalence stricte de la vectorisation numpy avec la boucle de
    référence (non-régression) ;
  - les cas dégénérés (série courte, plateau, pic isolé) ;
  - l'uniformité de sortie (regime + confidence, [0, 1]) ;
  - le marquage `is_partial` et hors grille ;
  - les validations des paramètres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.price.price_action import (
    PriceActionDetector,
    _detect_fractals,
)


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------
def _detector(**kwargs) -> PriceActionDetector:
    """Instancie un détecteur en mode offline (lookahead autorisé)."""
    return PriceActionDetector(
        timeframe="H1", allow_lookahead=True, **kwargs
    )


# ---------------------------------------------------------------------------
# Caractère offline
# ---------------------------------------------------------------------------
def test_requires_lookahead_refuses_without_flag():
    """La méthode doit refuser de s'instancier sans lookahead explicite."""
    with pytest.raises(ValueError, match="exige le lookahead"):
        PriceActionDetector(timeframe="H1")


def test_requires_lookahead_accepts_with_flag():
    """Elle s'instancie avec allow_lookahead=True."""
    det = PriceActionDetector(timeframe="H1", allow_lookahead=True)
    assert det.requires_lookahead is True
    assert det.allow_lookahead is True


# ---------------------------------------------------------------------------
# Preuve du lookahead
# ---------------------------------------------------------------------------
def test_fractals_depend_on_future():
    """
    Preuve DIRECTE du lookahead structurel.

    On vérifie que la détection de fractale à l'indice i dépend d'une
    barre FUTURE (i+1). C'est exactement la définition du lookahead.

    Ce test remplace un ancien test de troncature qui était
    conceptuellement faux : le lookahead de price_action est un lookahead
    de LATENCE (il faut attendre n_fractale barres pour confirmer un
    pivot), pas un lookahead de VALEURS (il ne modifie pas la
    classification du passé).
    """
    # Série contrôlée : sommet net à l'indice 2.
    high = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    low = high - 0.5
    n_fractale = 2

    s_before, _ = _detect_fractals(high, low, n_fractale)
    assert s_before[2], "high[2]=3.0 devait être un sommet initial"

    # On perturbe une barre FUTURE (i + 1 = 3) pour invalider le sommet.
    high_mod = high.copy()
    high_mod[3] = 4.0  # high[3] dépasse maintenant high[2]

    s_after, _ = _detect_fractals(high_mod, low, n_fractale)
    assert not s_after[2], (
        "Après perturbation de high[3], high[2] ne doit plus être un "
        "sommet (high[3] le dépasse)"
    )

    # Conclusion : la détection à i=2 dépend de la barre i+1=3 → lookahead.
    assert s_before[2] != s_after[2], (
        "La détection de fractale à i=2 n'a pas changé malgré la "
        "modification de la barre future i+1=3 → le lookahead structurel "
        "n'est pas effectif."
    )


def test_truncation_does_not_change_past_classification(ohlcv_h1):
    """
    Caractérisation du lookahead de price_action.

    Contrairement aux méthodes causales, on N'ATTEND PAS de divergence
    sur les valeurs passées après troncature — le lookahead est un
    lookahead de LATENCE, pas de VALEURS. Ce test documente cette
    propriété : les classifications passées sont stables.

    Si un jour ce test échoue, c'est que l'implémentation a changé de
    nature (peut-être une version causale a été introduite par erreur).
    """
    det = _detector()
    full = det.fit_predict(ohlcv_h1)["regime"]
    k = len(ohlcv_h1) // 2
    truncated = det.fit_predict(ohlcv_h1.iloc[:k])["regime"]

    # Sur la plage [0, k - n_fractale - 1], les classifications doivent
    # être identiques : elles ne dépendent que des barres 0..k-1.
    safe_end = k - det.n_fractale - 1
    pd.testing.assert_series_equal(
        full.iloc[:safe_end].reset_index(drop=True),
        truncated.iloc[:safe_end].reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Détection de fractales — non-régression vectorisation
# ---------------------------------------------------------------------------
def test_fractals_vectorized_equals_loop():
    """
    La version vectorisée (sliding_window_view) doit produire exactement
    le même résultat que la boucle de référence.
    """
    rng = np.random.default_rng(seed=42)
    n = 500
    high = np.cumsum(rng.normal(0, 1.0, n)) + 100
    low = high - np.abs(rng.normal(0, 0.5, n))

    n_fractale = 5

    # --- Référence : boucle ---
    is_sommet_ref = np.zeros(n, dtype=bool)
    is_creux_ref = np.zeros(n, dtype=bool)
    for i in range(n_fractale, n - n_fractale):
        if (high[i] > high[i - n_fractale:i].max() and
                high[i] > high[i + 1:i + n_fractale + 1].max()):
            is_sommet_ref[i] = True
        if (low[i] < low[i - n_fractale:i].min() and
                low[i] < low[i + 1:i + n_fractale + 1].min()):
            is_creux_ref[i] = True

    # --- Version vectorisée ---
    is_sommet_vec, is_creux_vec = _detect_fractals(high, low, n_fractale)

    np.testing.assert_array_equal(is_sommet_vec, is_sommet_ref)
    np.testing.assert_array_equal(is_creux_vec, is_creux_ref)


def test_fractals_handles_short_series():
    """Séries plus courtes que la fenêtre → tout False."""
    for n in (0, 1, 5, 10):
        high = np.arange(n, dtype=float) + 100
        low = high - 1.0
        s, c = _detect_fractals(high, low, n_fractale=5)
        assert not s.any()
        assert not c.any()


def test_fractals_detects_known_peak():
    """Pic artificiel isolé → détecté comme sommet."""
    high = np.linspace(100, 101, 50)
    high[25] = 105.0
    low = high - 0.5
    s, c = _detect_fractals(high, low, n_fractale=5)
    assert s[25], "Le pic artificiel n'a pas été détecté"


def test_fractals_rejects_plateau():
    """
    Un plateau de valeurs égales ne doit PAS être un sommet, car
    l'inégalité est STRICTE (> et non >=).
    """
    high = np.full(50, 100.0)
    high[20:30] = 105.0  # plateau
    low = high - 0.5
    s, c = _detect_fractals(high, low, n_fractale=5)
    assert not s[20:30].any(), (
        "Un plateau ne doit pas être détecté comme sommet (inégalité "
        "stricte attendue)."
    )


def test_fractals_handles_n_fractale_one():
    """
    Cas dégénéré : n_fractale=1 → comparaison stricte aux voisins
    immédiats. Les indices de bord 0 et n-1 ne sont pas classifiables
    (il manque un voisin d'un côté).
    """
    # high : sommets nets à 1 et 3.
    # low  : creux nets à 2.
    high = np.array([1.0, 3.0, 1.5, 3.0, 1.0])
    low = np.array([0.5, 2.5, 0.5, 2.5, 0.5])

    s, c = _detect_fractals(high, low, n_fractale=1)

    # --- Sommets ---
    assert s[1], "high[1]=3.0 devait être un sommet"
    assert s[3], "high[3]=3.0 devait être un sommet"
    assert not s[0], "high[0] est un bord, non classifiable"
    assert not s[2], "high[2]=1.5 n'est pas un pic local"
    assert not s[4], "high[4] est un bord, non classifiable"

    # --- Creux ---
    assert c[2], "low[2]=0.5 devait être un creux"
    assert not c[0], "low[0] est un bord, non classifiable"
    assert not c[1], "low[1]=2.5 n'est pas un creux local"
    assert not c[3], "low[3]=2.5 n'est pas un creux local"
    assert not c[4], "low[4] est un bord, non classifiable"


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


def test_warmup_produces_unknown(ohlcv_h1):
    """
    Les premières barres (avant 2*n_fractale) doivent être INCONNU.
    """
    det = _detector()
    out = det.fit_predict(ohlcv_h1)
    warmup = 2 * det.n_fractale
    assert (out["regime"].iloc[:warmup] == "INCONNU").all()


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
    """Barres hors grille H1 → INCONNU en l'absence de `is_partial`."""
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


def test_short_series_produces_all_unknown():
    """Série plus courte que 2*n_fractale + 4 → tout INCONNU."""
    n = 10
    index = pd.date_range(
        "2024-01-01", periods=n, freq="1h", tz="UTC", name="timestamp"
    )
    rng = np.random.default_rng(seed=5)
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
    assert (out["regime"] == "INCONNU").all()


# ---------------------------------------------------------------------------
# Validations des paramètres
# ---------------------------------------------------------------------------
def test_rejects_invalid_n_fractale():
    with pytest.raises(ValueError, match="n_fractale"):
        PriceActionDetector(
            timeframe="H1", allow_lookahead=True, n_fractale=0
        )


def test_rejects_invalid_k_atr():
    with pytest.raises(ValueError, match="k_atr"):
        PriceActionDetector(
            timeframe="H1", allow_lookahead=True, k_atr=0.0
        )


def test_rejects_invalid_n_atr_tol():
    with pytest.raises(ValueError, match="n_atr_tol"):
        PriceActionDetector(
            timeframe="H1", allow_lookahead=True, n_atr_tol=0
        )


# ---------------------------------------------------------------------------
# Non-régression : masque unaligned agnostique à la résolution
# ---------------------------------------------------------------------------
def test_unaligned_mask_agnostic_to_index_resolution():
    """
    Non-régression : le masque doit fonctionner quelle que soit la
    résolution interne du DatetimeIndex (s, ms, us, ns).
    """
    n = 100
    for unit in ("s", "ms", "us", "ns"):
        index = pd.date_range(
            "2024-01-01", periods=n, freq="1h", tz="UTC"
        )
        try:
            index = index.as_unit(unit, round_ok=True)
        except (AttributeError, TypeError):
            pytest.skip(f"as_unit({unit}) non supporté sur cette version")
        index.name = "timestamp"
        mask = PriceActionDetector._unaligned_mask(index, "H1")
        assert not mask.any(), (
            f"Résolution {unit} : {int(mask.sum())} barres marquées "
            f"hors grille alors qu'elles sont alignées."
        )
