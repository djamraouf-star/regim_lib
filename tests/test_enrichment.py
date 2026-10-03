"""
Tests unitaires du module `context.enrichment`.

Vérifie :
  - l'orchestration complète (`enrichir_contexte`) ;
  - la présence des 8 colonnes `ctx_*` ;
  - la préservation des colonnes originales ;
  - la cohérence avec les modules individuels ;
  - le cas où le profil n'a ni fériés ni événements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.config import load_profile
from regime_lib.context.enrichment import enrichir_contexte


COLS_CTX = [
    "ctx_session",
    "ctx_hour_ny",
    "ctx_hour_utc",
    "ctx_day_of_week",
    "ctx_is_rollover",
    "ctx_is_holiday",
    "ctx_in_event",
    "ctx_event_type",
]


@pytest.fixture
def config_forex():
    """Configuration du contexte du profil default."""
    return load_profile("default").context


@pytest.fixture
def df_h1_simple():
    """DataFrame OHLCV minimal sur 48h."""
    idx = pd.date_range("2026-01-15 00:00", periods=48, freq="1h", tz="UTC")
    n = len(idx)
    close = 1.10 + np.cumsum(np.random.default_rng(0).normal(0, 1e-4, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1e-4,
            "low": close - 1e-4,
            "close": close,
            "volume": np.full(n, 100.0),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def test_enrichir_contexte_8_colonnes(config_forex, df_h1_simple):
    """Les 8 colonnes ctx_* sont présentes dans la sortie."""
    out = enrichir_contexte(df_h1_simple, "H1", config_forex)
    for col in COLS_CTX:
        assert col in out.columns, f"Colonne manquante : {col}"


def test_enrichir_contexte_preserve_colonnes(config_forex, df_h1_simple):
    """Les colonnes originales sont préservées."""
    out = enrichir_contexte(df_h1_simple, "H1", config_forex)
    for col in df_h1_simple.columns:
        assert col in out.columns
    assert len(out) == len(df_h1_simple)
    pd.testing.assert_index_equal(out.index, df_h1_simple.index)


def test_enrichir_contexte_ne_modifie_pas_entree(config_forex, df_h1_simple):
    """Le DataFrame en entrée n'est pas modifié."""
    df_avant = df_h1_simple.copy()
    _ = enrichir_contexte(df_h1_simple, "H1", config_forex)
    pd.testing.assert_frame_equal(df_h1_simple, df_avant)


def test_enrichir_contexte_sans_fichiers(df_h1_simple):
    """Un profil sans fériés ni événements fonctionne quand même."""
    from regime_lib.config.schema import ContextConfig

    config = ContextConfig(
        reference_tz="America/New_York",
        sessions={
            "TOUJOURS": {"debut": "00:00", "fin": "23:59"},
        },
        rollover=None,
        events={},
        holidays={},
    )
    out = enrichir_contexte(df_h1_simple, "H1", config)
    assert out["ctx_is_holiday"].sum() == 0
    assert out["ctx_in_event"].sum() == 0
    assert (out["ctx_session"] == "TOUJOURS").all()


def test_avertit_si_donnees_apres_le_dernier_evenement(config_forex):
    """Le calendrier incomplet est signalé si les barres le dépassent."""
    idx = pd.date_range("2026-10-03", periods=2, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": [1.0, 1.0]}, index=idx)

    with pytest.warns(UserWarning, match="dépassent la date du dernier événement"):
        enrichir_contexte(df, "H1", config_forex)


# ---------------------------------------------------------------------------
# Cohérence avec les modules individuels
# ---------------------------------------------------------------------------
def test_session_coherente(config_forex, df_h1_simple):
    """
    La colonne `ctx_session` doit être cohérente avec le calcul
    individuel du module `sessions`.
    """
    from regime_lib.context import calculer_sessions

    out = enrichir_contexte(df_h1_simple, "H1", config_forex)
    sessions = calculer_sessions(df_h1_simple.index, "H1", config_forex)

    pd.testing.assert_series_equal(
        out["ctx_session"], sessions["ctx_session"],
        check_names=False,
    )


def test_calendrier_coherent(config_forex, df_h1_simple):
    """
    Les colonnes de calendrier sont cohérentes avec le module
    `calendar`.
    """
    from regime_lib.context.calendar import (
        calculer_calendrier,
        charger_holidays,
    )

    holidays = charger_holidays(config_forex.holidays["fichier"])
    cal = calculer_calendrier(
        df_h1_simple.index, "H1", config_forex, holidays=holidays,
    )
    out = enrichir_contexte(df_h1_simple, "H1", config_forex)

    for col in ("ctx_day_of_week", "ctx_is_rollover", "ctx_is_holiday"):
        pd.testing.assert_series_equal(
            out[col], cal[col], check_names=False,
        )


def test_events_coherents(config_forex, df_h1_simple):
    """
    Les colonnes d'événements sont cohérentes avec le module `events`.
    """
    from regime_lib.context.events import (
        calculer_events,
        charger_events,
    )

    events = charger_events(config_forex.events["fichier"])
    ctx = calculer_events(
        df_h1_simple.index, "H1", config_forex, events=events,
    )
    out = enrichir_contexte(df_h1_simple, "H1", config_forex)

    for col in ("ctx_in_event", "ctx_event_type"):
        pd.testing.assert_series_equal(
            out[col], ctx[col], check_names=False,
        )


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
def test_types_colonnes(config_forex, df_h1_simple):
    """
    Vérifie les types des colonnes ctx_*.

    Note : pandas 2.x peut produire un dtype `StringDtype` pour les
    colonnes de chaînes plutôt que `object`. On accepte les deux.
    """
    from pandas.api.types import is_string_dtype, is_integer_dtype, is_bool_dtype

    out = enrichir_contexte(df_h1_simple, "H1", config_forex)

    # Colonnes string (dtype object ou StringDtype selon pandas)
    assert is_string_dtype(out["ctx_session"])
    assert is_string_dtype(out["ctx_event_type"])

    # Colonnes entières
    assert is_integer_dtype(out["ctx_hour_ny"])
    assert is_integer_dtype(out["ctx_hour_utc"])
    assert is_integer_dtype(out["ctx_day_of_week"])

    # Colonnes booléennes
    assert is_bool_dtype(out["ctx_is_rollover"])
    assert is_bool_dtype(out["ctx_is_holiday"])
    assert is_bool_dtype(out["ctx_in_event"])


def test_deterministe(config_forex, df_h1_simple):
    """Deux appels produisent le même résultat."""
    out1 = enrichir_contexte(df_h1_simple, "H1", config_forex)
    out2 = enrichir_contexte(df_h1_simple, "H1", config_forex)
    pd.testing.assert_frame_equal(out1, out2)
