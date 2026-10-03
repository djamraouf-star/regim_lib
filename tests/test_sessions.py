"""
Tests unitaires du module `context.sessions`.

Vérifie :
  - la correspondance heure UTC ↔ heure NY en hiver et en été ;
  - la règle d'affectation « barre classée selon son début » ;
  - la gestion du chevauchement minuit (ASIE 19:00 → 03:00) ;
  - le cas frontière aux bornes de session ;
  - la valeur HORS_SESSION pour la fenêtre non couverte ;
  - les erreurs (index sans tz, timeframe inconnu) ;
  - l'enrichissement (colonnes ajoutées, originales préservées).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.config import load_profile
from regime_lib.context import calculer_sessions, enrichir_sessions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def config_forex():
    """Configuration du contexte du profil default (forex, NY)."""
    return load_profile("default").context


@pytest.fixture
def index_hiver_24h():
    """24 barres H1 sur une journée d'hiver (NY = UTC-5)."""
    return pd.date_range(
        "2026-01-15 00:00", periods=24, freq="1h", tz="UTC"
    )


@pytest.fixture
def index_ete_24h():
    """24 barres H1 sur une journée d'été (NY = UTC-4)."""
    return pd.date_range(
        "2026-07-15 00:00", periods=24, freq="1h", tz="UTC"
    )


# ---------------------------------------------------------------------------
# Correspondance heure UTC ↔ heure NY
# ---------------------------------------------------------------------------
def test_hiver_utc_vers_ny(index_hiver_24h, config_forex):
    """En janvier, NY = UTC-5."""
    ctx = calculer_sessions(index_hiver_24h, "H1", config_forex)
    # La barre étiquetée 00:00 UTC couvre 23:00 → 00:00 UTC, soit
    # 18:00 → 19:00 NY.
    assert ctx.iloc[0]["ctx_hour_utc"] == 23
    assert ctx.iloc[0]["ctx_hour_ny"] == 18
    # La barre étiquetée 05:00 UTC → début 04:00 UTC → 23:00 NY (J-1).
    assert ctx.iloc[5]["ctx_hour_utc"] == 4
    assert ctx.iloc[5]["ctx_hour_ny"] == 23


def test_ete_utc_vers_ny(index_ete_24h, config_forex):
    """En juillet, NY = UTC-4."""
    ctx = calculer_sessions(index_ete_24h, "H1", config_forex)
    # La barre étiquetée 00:00 UTC couvre 23:00 → 00:00 UTC, soit
    # 19:00 → 20:00 NY (décalage -4).
    assert ctx.iloc[0]["ctx_hour_utc"] == 23
    assert ctx.iloc[0]["ctx_hour_ny"] == 19
    # La barre étiquetée 05:00 UTC → début 04:00 UTC → 00:00 NY.
    assert ctx.iloc[5]["ctx_hour_utc"] == 4
    assert ctx.iloc[5]["ctx_hour_ny"] == 0


# ---------------------------------------------------------------------------
# Affectation par session
# ---------------------------------------------------------------------------
def test_repartition_hiver(index_hiver_24h, config_forex):
    """
    Répartition attendue sur 24h en hiver :
    ASIE 8, LONDRES 5, OVERLAP 4, NY 5, HORS_SESSION 2.
    """
    ctx = calculer_sessions(index_hiver_24h, "H1", config_forex)
    counts = ctx["ctx_session"].value_counts().to_dict()
    assert counts == {
        "ASIE": 8,
        "LONDRES": 5,
        "OVERLAP": 4,
        "NY": 5,
        "HORS_SESSION": 2,
    }


def test_repartition_ete(index_ete_24h, config_forex):
    """
    La répartition en été doit être identique : les bornes sont
    exprimées en heure NY, stables par construction.
    """
    ctx = calculer_sessions(index_ete_24h, "H1", config_forex)
    counts = ctx["ctx_session"].value_counts().to_dict()
    assert counts == {
        "ASIE": 8,
        "LONDRES": 5,
        "OVERLAP": 4,
        "NY": 5,
        "HORS_SESSION": 2,
    }


def test_transitions_hiver(index_hiver_24h, config_forex):
    """Vérifie les transitions aux bornes exactes en hiver."""
    ctx = calculer_sessions(index_hiver_24h, "H1", config_forex)
    sessions = ctx["ctx_session"].tolist()

    # Index et heure NY du début de la barre (étiquette - 1h).
    # Barre étiquetée 09:00 UTC → début 08:00 UTC → 03:00 NY → LONDRES.
    assert sessions[9] == "LONDRES"
    # Barre étiquetée 14:00 UTC → début 13:00 UTC → 08:00 NY → OVERLAP.
    assert sessions[14] == "OVERLAP"
    # Barre étiquetée 18:00 UTC → début 17:00 UTC → 12:00 NY → NY.
    assert sessions[18] == "NY"
    # Barre étiquetée 23:00 UTC → début 22:00 UTC → 17:00 NY → HORS_SESSION.
    assert sessions[23] == "HORS_SESSION"


# ---------------------------------------------------------------------------
# Chevauchement minuit (ASIE)
# ---------------------------------------------------------------------------
def test_asie_chevauche_minuit(config_forex):
    """
    La session ASIE (19:00 → 03:00 NY) doit être continue de part et
    d'autre de minuit NY.
    """
    # 4 barres H1 : 23:00, 00:00, 01:00, 02:00 NY (début)
    # En hiver : début 04:00, 05:00, 06:00, 07:00 UTC.
    idx = pd.date_range(
        "2026-01-15 05:00", periods=4, freq="1h", tz="UTC"
    )
    ctx = calculer_sessions(idx, "H1", config_forex)
    assert all(ctx["ctx_session"] == "ASIE"), ctx["ctx_session"].tolist()


def test_asie_fin_avant_minuit(config_forex):
    """Une barre à 02:00 NY (début) est en ASIE, à 03:00 NY en LONDRES."""
    # En hiver : début 02:00 NY → 07:00 UTC (étiquette 08:00 UTC).
    # Début 03:00 NY → 08:00 UTC (étiquette 09:00 UTC).
    idx = pd.DatetimeIndex([
        "2026-01-15 08:00:00+00:00",   # début 07:00 UTC = 02:00 NY
        "2026-01-15 09:00:00+00:00",   # début 08:00 UTC = 03:00 NY
    ])
    ctx = calculer_sessions(idx, "H1", config_forex)
    assert ctx.iloc[0]["ctx_session"] == "ASIE"
    assert ctx.iloc[1]["ctx_session"] == "LONDRES"


# ---------------------------------------------------------------------------
# Cas frontière : barre débutant exactement à une borne
# ---------------------------------------------------------------------------
def test_borne_inferieure_incluse(config_forex):
    """
    Une barre dont le début est à 03:00 NY appartient à LONDRES
    (borne inférieure incluse).
    """
    # 03:00 NY en hiver → 08:00 UTC. Barre étiquetée 09:00 UTC.
    idx = pd.DatetimeIndex(["2026-01-15 09:00:00+00:00"])
    ctx = calculer_sessions(idx, "H1", config_forex)
    assert ctx.iloc[0]["ctx_session"] == "LONDRES"
    assert ctx.iloc[0]["ctx_hour_ny"] == 3


def test_borne_superieure_exclue(config_forex):
    """
    Une barre dont le début est à 08:00 NY (fin de LONDRES) est en
    OVERLAP (borne supérieure exclue).
    """
    # 08:00 NY en hiver → 13:00 UTC. Barre étiquetée 14:00 UTC.
    idx = pd.DatetimeIndex(["2026-01-15 14:00:00+00:00"])
    ctx = calculer_sessions(idx, "H1", config_forex)
    assert ctx.iloc[0]["ctx_session"] == "OVERLAP"
    assert ctx.iloc[0]["ctx_hour_ny"] == 8


# ---------------------------------------------------------------------------
# HORS_SESSION
# ---------------------------------------------------------------------------
def test_hors_session_fenetre_17h19h_ny(config_forex):
    """La fenêtre 17:00–19:00 NY n'est couverte par aucune session."""
    # 17:00 NY en hiver → 22:00 UTC. Barre étiquetée 23:00 UTC.
    # 18:00 NY en hiver → 23:00 UTC. Barre étiquetée 00:00 UTC (J+1).
    idx = pd.DatetimeIndex([
        "2026-01-15 23:00:00+00:00",   # début 22:00 UTC = 17:00 NY
        "2026-01-16 00:00:00+00:00",   # début 23:00 UTC = 18:00 NY
    ])
    ctx = calculer_sessions(idx, "H1", config_forex)
    assert all(ctx["ctx_session"] == "HORS_SESSION")


# ---------------------------------------------------------------------------
# Cas dégénérés
# ---------------------------------------------------------------------------
def test_index_vide(config_forex):
    """Un index vide retourne un DataFrame vide avec les bonnes colonnes."""
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    ctx = calculer_sessions(idx, "H1", config_forex)
    assert ctx.empty
    assert set(ctx.columns) == {"ctx_session", "ctx_hour_ny", "ctx_hour_utc"}


def test_index_sans_timezone(config_forex):
    """Un index naïf (sans tz) doit lever une erreur explicite."""
    idx = pd.date_range("2026-01-15", periods=5, freq="1h")
    with pytest.raises(ValueError, match="tz-aware"):
        calculer_sessions(idx, "H1", config_forex)


def test_timeframe_inconnu(config_forex):
    """Un timeframe hors TIMEFRAME_RULES doit lever une erreur."""
    idx = pd.date_range("2026-01-15", periods=5, freq="1h", tz="UTC")
    with pytest.raises(ValueError, match="Timeframe inconnu"):
        calculer_sessions(idx, "M7", config_forex)


# ---------------------------------------------------------------------------
# Différents timeframes
# ---------------------------------------------------------------------------
def test_timeframe_m15_debut_barre(config_forex):
    """
    En M15, une barre étiquetée 00:15 UTC couvre 00:00 → 00:15 UTC.
    Son début est donc à 00:00 UTC.
    """
    idx = pd.DatetimeIndex(["2026-01-15 00:15:00+00:00"])
    ctx = calculer_sessions(idx, "M15", config_forex)
    assert ctx.iloc[0]["ctx_hour_utc"] == 0


def test_timeframe_h4_debut_barre(config_forex):
    """
    En H4, une barre étiquetée 04:00 UTC couvre 00:00 → 04:00 UTC.
    Son début est à 00:00 UTC.
    """
    idx = pd.DatetimeIndex(["2026-01-15 04:00:00+00:00"])
    ctx = calculer_sessions(idx, "H4", config_forex)
    assert ctx.iloc[0]["ctx_hour_utc"] == 0


# ---------------------------------------------------------------------------
# enrichir_sessions
# ---------------------------------------------------------------------------
def test_enrichir_sessions_preserve_colonnes(index_hiver_24h, config_forex):
    """Les colonnes originales sont préservées, les nouvelles sont ajoutées."""
    df = pd.DataFrame({
        "open": np.arange(24, dtype=float),
        "close": np.arange(24, dtype=float) + 0.5,
    }, index=index_hiver_24h)

    out = enrichir_sessions(df, "H1", config_forex)
    assert "open" in out.columns
    assert "close" in out.columns
    assert "ctx_session" in out.columns
    assert "ctx_hour_ny" in out.columns
    assert "ctx_hour_utc" in out.columns
    assert len(out) == len(df)
    pd.testing.assert_index_equal(out.index, df.index)


def test_enrichir_sessions_ne_modifie_pas_entree(index_hiver_24h, config_forex):
    """La fonction ne doit pas modifier le DataFrame en entrée."""
    df = pd.DataFrame({"open": np.arange(24, dtype=float)}, index=index_hiver_24h)
    df_avant = df.copy()

    _ = enrichir_sessions(df, "H1", config_forex)
    pd.testing.assert_frame_equal(df, df_avant)


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
def test_resultat_deterministe(index_hiver_24h, config_forex):
    """Deux appels successifs produisent exactement le même résultat."""
    ctx1 = calculer_sessions(index_hiver_24h, "H1", config_forex)
    ctx2 = calculer_sessions(index_hiver_24h, "H1", config_forex)
    pd.testing.assert_frame_equal(ctx1, ctx2)
