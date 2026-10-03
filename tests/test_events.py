"""
Tests unitaires du module `context.events`.

Vérifie :
  - le chargement CSV (format NY → UTC) ;
  - le calcul des fenêtres (avant/après) ;
  - la préservation du type d'événement ;
  - le fallback par défaut pour un type inconnu ;
  - les cas dégénérés (pas d'événements, index vide, sans tz) ;
  - l'enrichissement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.config import load_profile
from regime_lib.context.events import (
    calculer_events,
    charger_events,
    enrichir_events,
)


@pytest.fixture
def config_forex():
    """Configuration du contexte du profil default."""
    return load_profile("default").context


@pytest.fixture
def csv_events(tmp_path):
    """Petit CSV d'événements pour les tests (heure NY)."""
    p = tmp_path / "events_test.csv"
    p.write_text(
        "datetime_ny,event_type\n"
        "2026-01-09 08:30,NFP\n"
        "2026-01-14 08:30,CPI\n"
        "2026-01-28 14:00,FOMC\n"
    )
    return p


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def test_charger_events_format_ok(csv_events):
    """Un CSV valide est chargé avec conversion NY → UTC."""
    df = charger_events(str(csv_events))
    assert set(df.columns) == {"datetime_utc", "event_type"}
    assert len(df) == 3
    assert df["datetime_utc"].dt.tz is not None
    # Vérifie le premier : 09 janvier 08:30 NY = 13:30 UTC (hiver)
    assert df["datetime_utc"].iloc[0].hour == 13
    assert df["datetime_utc"].iloc[0].minute == 30


def test_charger_events_fichier_absent(tmp_path):
    """Fichier inexistant → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        charger_events(str(tmp_path / "nope.csv"))


def test_charger_events_colonne_manquante(tmp_path):
    """CSV sans colonnes requises → ValueError."""
    p = tmp_path / "bad.csv"
    p.write_text("date,event\n2026-01-01,X\n")
    with pytest.raises(ValueError, match="colonnes"):
        charger_events(str(p))


# ---------------------------------------------------------------------------
# Calcul des événements
# ---------------------------------------------------------------------------
def test_calculer_events_colonnes(config_forex, csv_events):
    """Les 2 colonnes attendues sont produites."""
    events = charger_events(str(csv_events))
    idx = pd.date_range("2026-01-09", periods=24, freq="1h", tz="UTC")
    ctx = calculer_events(idx, "H1", config_forex, events)
    assert set(ctx.columns) == {"ctx_in_event", "ctx_event_type"}
    assert len(ctx) == 24


def test_calculer_events_index_vide(config_forex, csv_events):
    """Index vide → DataFrame vide."""
    events = charger_events(str(csv_events))
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    ctx = calculer_events(idx, "H1", config_forex, events)
    assert ctx.empty
    assert set(ctx.columns) == {"ctx_in_event", "ctx_event_type"}


def test_calculer_events_sans_events(config_forex):
    """Si events est None, aucune barre n'est marquée."""
    idx = pd.date_range("2026-01-09", periods=24, freq="1h", tz="UTC")
    ctx = calculer_events(idx, "H1", config_forex, events=None)
    assert ctx["ctx_in_event"].sum() == 0
    assert (ctx["ctx_event_type"] == "").all()


def test_calculer_events_sans_tz(config_forex, csv_events):
    """Index naïf → ValueError."""
    events = charger_events(str(csv_events))
    idx = pd.date_range("2026-01-09", periods=24, freq="1h")
    with pytest.raises(ValueError, match="tz-aware"):
        calculer_events(idx, "H1", config_forex, events)


def test_calculer_events_timeframe_inconnu(config_forex, csv_events):
    """Timeframe invalide → ValueError."""
    events = charger_events(str(csv_events))
    idx = pd.date_range("2026-01-09", periods=24, freq="1h", tz="UTC")
    with pytest.raises(ValueError, match="Timeframe inconnu"):
        calculer_events(idx, "M7", config_forex, events)


# ---------------------------------------------------------------------------
# Fenêtres
# ---------------------------------------------------------------------------
def test_fenetre_nfp(config_forex, csv_events):
    """
    NFP du 09 janvier 2026 à 08:30 NY = 13:30 UTC.
    Fenêtre : [13:25, 14:30) UTC → 1 barre H1 dans la fenêtre
    (celle qui commence à 14:00 UTC, étiquette 15:00 UTC).
    """
    events = charger_events(str(csv_events))
    # Index H1 sur 3h autour de l'événement
    idx = pd.date_range("2026-01-09 12:00", periods=6, freq="1h", tz="UTC")
    ctx = calculer_events(idx, "H1", config_forex, events)

    # Une seule barre doit être dans la fenêtre NFP
    assert ctx["ctx_in_event"].sum() == 1
    assert (ctx.loc[ctx["ctx_in_event"], "ctx_event_type"] == "NFP").all()


def test_fenetre_fomc_plus_large(config_forex, csv_events):
    """
    FOMC du 28 janvier 2026 à 14:00 NY = 19:00 UTC.
    Fenêtre : [18:55, 20:30) UTC → 2 barres H1
    (celles qui commencent à 19:00 et 20:00 UTC).
    """
    events = charger_events(str(csv_events))
    idx = pd.date_range("2026-01-28 17:00", periods=6, freq="1h", tz="UTC")
    ctx = calculer_events(idx, "H1", config_forex, events)

    n_fomc = (ctx["ctx_event_type"] == "FOMC").sum()
    assert n_fomc == 2, f"Attendu 2 barres FOMC, reçu {n_fomc}"


def test_fenetre_borne_droite_exclusive(config_forex):
    """Une barre qui commence à la fin de la fenêtre n'est pas marquée."""
    events = pd.DataFrame({
        "datetime_utc": pd.DatetimeIndex(["2026-01-09 13:30:00+00:00"]),
        "event_type": ["NFP"],
    })
    idx = pd.date_range(
        "2026-01-09 14:30", periods=2, freq="1min", tz="UTC"
    )

    ctx = calculer_events(idx, "M1", config_forex, events)

    assert ctx["ctx_in_event"].tolist() == [True, False]
    assert ctx["ctx_event_type"].tolist() == ["NFP", ""]


def test_type_preserve(config_forex, csv_events):
    """Le type d'événement est correctement associé."""
    events = charger_events(str(csv_events))
    idx = pd.date_range("2026-01-14 12:00", periods=6, freq="1h", tz="UTC")
    ctx = calculer_events(idx, "H1", config_forex, events)

    types = ctx.loc[ctx["ctx_in_event"], "ctx_event_type"].unique()
    assert "CPI" in types


def test_defaut_si_type_inconnu():
    """Un type d'événement absent de config.fenetres reçoit un défaut."""
    from regime_lib.config.schema import ContextConfig

    config = ContextConfig(
        reference_tz="America/New_York",
        events={"fenetres": {}},  # aucune config de fenêtre
    )
    # Créer un événement manuellement
    events = pd.DataFrame({
        "datetime_utc": pd.DatetimeIndex(
            ["2026-01-09 13:30:00+00:00"]
        ),
        "event_type": ["INCONNU_TYPE"],
    })
    idx = pd.date_range("2026-01-09 12:00", periods=6, freq="1h", tz="UTC")
    ctx = calculer_events(idx, "H1", config, events)

    # Le défaut [avant=5, après=60] doit s'appliquer
    assert ctx["ctx_in_event"].sum() >= 1
    assert (ctx.loc[ctx["ctx_in_event"], "ctx_event_type"]
            == "INCONNU_TYPE").all()


# ---------------------------------------------------------------------------
# enrichir_events
# ---------------------------------------------------------------------------
def test_enrichir_events_preserve_colonnes(config_forex, csv_events):
    """Les colonnes originales sont préservées."""
    events = charger_events(str(csv_events))
    idx = pd.date_range("2026-01-09", periods=24, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.arange(24.0)}, index=idx)

    out = enrichir_events(df, "H1", config_forex, events)
    assert "open" in out.columns
    assert "ctx_in_event" in out.columns
    assert "ctx_event_type" in out.columns
    assert len(out) == len(df)


def test_enrichir_events_ne_modifie_pas_entree(config_forex, csv_events):
    """Le DataFrame en entrée n'est pas modifié."""
    events = charger_events(str(csv_events))
    idx = pd.date_range("2026-01-09", periods=24, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.arange(24.0)}, index=idx)
    df_avant = df.copy()

    _ = enrichir_events(df, "H1", config_forex, events)
    pd.testing.assert_frame_equal(df, df_avant)
