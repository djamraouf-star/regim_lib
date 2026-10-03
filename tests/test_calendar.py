"""
Tests unitaires du module `context.calendar`.

Vérifie :
  - le calcul des jours de semaine en heure NY ;
  - la détection du rollover (fenêtre en heure NY) ;
  - la détection des jours fériés depuis un CSV ;
  - le chargement des CSV (format, erreurs) ;
  - les cas dégénérés (index vide, sans tz, timeframe inconnu) ;
  - l'enrichissement (colonnes ajoutées, originales préservées).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime_lib.config import load_profile
from regime_lib.context.calendar import (
    calculer_calendrier,
    charger_holidays,
    enrichir_calendrier,
)


@pytest.fixture
def config_forex():
    """Configuration du contexte du profil default."""
    return load_profile("default").context


@pytest.fixture
def csv_holidays(tmp_path):
    """Petit CSV de fériés pour les tests."""
    p = tmp_path / "holidays_test.csv"
    p.write_text(
        "date,label\n"
        "2026-01-01,Nouvel An\n"
        "2026-12-25,Noël\n"
    )
    return p


# ---------------------------------------------------------------------------
# Calcul du calendrier
# ---------------------------------------------------------------------------
def test_calculer_calendrier_colonnes(config_forex):
    """Les 3 colonnes attendues sont produites."""
    idx = pd.date_range("2026-01-15", periods=5, freq="1h", tz="UTC")
    cal = calculer_calendrier(idx, "H1", config_forex)
    assert set(cal.columns) == {
        "ctx_day_of_week", "ctx_is_rollover", "ctx_is_holiday",
    }
    assert len(cal) == 5


def test_calculer_calendrier_index_vide(config_forex):
    """Index vide → DataFrame vide avec les bonnes colonnes."""
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    cal = calculer_calendrier(idx, "H1", config_forex)
    assert cal.empty
    assert set(cal.columns) == {
        "ctx_day_of_week", "ctx_is_rollover", "ctx_is_holiday",
    }


def test_calculer_calendrier_sans_tz(config_forex):
    """Index naïf → ValueError."""
    idx = pd.date_range("2026-01-15", periods=5, freq="1h")
    with pytest.raises(ValueError, match="tz-aware"):
        calculer_calendrier(idx, "H1", config_forex)


def test_calculer_calendrier_timeframe_inconnu(config_forex):
    """Timeframe hors DUREES → ValueError."""
    idx = pd.date_range("2026-01-15", periods=5, freq="1h", tz="UTC")
    with pytest.raises(ValueError, match="Timeframe inconnu"):
        calculer_calendrier(idx, "M7", config_forex)


# ---------------------------------------------------------------------------
# Jour de semaine (NY)
# ---------------------------------------------------------------------------
def test_day_of_week_hiver(config_forex):
    """
    Le 15 janvier 2026 est un jeudi.
    En hiver (NY = UTC−5), la barre étiquetée 00:00 UTC a un début à
    23:00 UTC le 14 → 18:00 NY le 14 (mercredi).
    """
    idx = pd.date_range("2026-01-15 00:00", periods=24, freq="1h", tz="UTC")
    cal = calculer_calendrier(idx, "H1", config_forex)

    # Barre 00:00 UTC → début 23:00 UTC le 14 → 18:00 NY le 14 (mercredi = 2)
    assert cal.iloc[0]["ctx_day_of_week"] == 2
    # Barre 06:00 UTC → début 05:00 UTC = 00:00 NY le 15 (jeudi = 3)
    assert cal.iloc[6]["ctx_day_of_week"] == 3


def test_day_of_week_ete(config_forex):
    """
    Le 15 juillet 2026 est un mercredi.
    En été (NY = UTC−4), la barre étiquetée 00:00 UTC a un début à
    23:00 UTC le 14 → 19:00 NY le 14 (mardi).
    """
    idx = pd.date_range("2026-07-15 00:00", periods=24, freq="1h", tz="UTC")
    cal = calculer_calendrier(idx, "H1", config_forex)

    # Barre 00:00 UTC → début 23:00 UTC le 14 → 19:00 NY le 14 (mardi = 1)
    assert cal.iloc[0]["ctx_day_of_week"] == 1
    # Barre 05:00 UTC → début 04:00 UTC = 00:00 NY le 15 (mercredi = 2)
    assert cal.iloc[5]["ctx_day_of_week"] == 2


# ---------------------------------------------------------------------------
# Rollover
# ---------------------------------------------------------------------------
def test_rollover_detecte_hiver(config_forex):
    """
    En hiver, le rollover [17:00, 18:00) NY = [22:00, 23:00) UTC.
    Seule la barre dont le début est dans cette fenêtre est marquée.
    """
    idx = pd.date_range("2026-01-15 00:00", periods=24, freq="1h", tz="UTC")
    cal = calculer_calendrier(idx, "H1", config_forex)

    assert cal["ctx_is_rollover"].sum() == 1
    # Barre étiquetée 23:00 UTC → début 22:00 UTC = 17:00 NY → rollover.
    assert cal.iloc[23]["ctx_is_rollover"]


def test_rollover_absent_si_config_none():
    """Si config.rollover est None, aucun rollover n'est marqué."""
    from regime_lib.config.schema import ContextConfig
    config = ContextConfig(
        reference_tz="America/New_York",
        sessions={"TEST": {"debut": "00:00", "fin": "00:01"}},
        rollover=None,
    )
    idx = pd.date_range("2026-01-15 00:00", periods=24, freq="1h", tz="UTC")
    cal = calculer_calendrier(idx, "H1", config)
    assert cal["ctx_is_rollover"].sum() == 0


# ---------------------------------------------------------------------------
# Fériés
# ---------------------------------------------------------------------------
def test_holidays_detecte(config_forex, csv_holidays):
    """Un férié fourni est correctement détecté."""
    holidays = charger_holidays(str(csv_holidays))

    # 25 décembre 2026 en H1 : 24 barres
    idx = pd.date_range("2026-12-25 00:00", periods=24, freq="1h", tz="UTC")
    cal = calculer_calendrier(idx, "H1", config_forex, holidays=holidays)

    # Les barres dont le début (jour UTC) est le 25 décembre
    n_feries = cal["ctx_is_holiday"].sum()
    assert n_feries > 0, "Aucune barre fériée détectée"


def test_holidays_absent_si_none(config_forex):
    """Si holidays est None, aucun férié n'est marqué."""
    idx = pd.date_range("2026-12-25 00:00", periods=24, freq="1h", tz="UTC")
    cal = calculer_calendrier(idx, "H1", config_forex, holidays=None)
    assert cal["ctx_is_holiday"].sum() == 0


# ---------------------------------------------------------------------------
# Chargement des fériés
# ---------------------------------------------------------------------------
def test_charger_holidays_format_ok(csv_holidays):
    """Un CSV valide est chargé en DatetimeIndex UTC normalisé."""
    h = charger_holidays(str(csv_holidays))
    assert isinstance(h, pd.DatetimeIndex)
    assert len(h) == 2
    assert h[0].tz is not None
    assert h[0].hour == 0 and h[0].minute == 0


def test_charger_holidays_fichier_absent(tmp_path):
    """Fichier inexistant → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        charger_holidays(str(tmp_path / "nope.csv"))


def test_charger_holidays_colonne_manquante(tmp_path):
    """CSV sans colonne 'date' → ValueError."""
    p = tmp_path / "bad.csv"
    p.write_text("jour,label\n2026-01-01,X\n")
    with pytest.raises(ValueError, match="colonne 'date'"):
        charger_holidays(str(p))


def test_charger_holidays_doublons_supprimes(tmp_path):
    """Les dates dupliquées sont dédupliquées."""
    p = tmp_path / "dup.csv"
    p.write_text(
        "date,label\n"
        "2026-01-01,A\n"
        "2026-01-01,B\n"
    )
    h = charger_holidays(str(p))
    assert len(h) == 1


# ---------------------------------------------------------------------------
# enrichir_calendrier
# ---------------------------------------------------------------------------
def test_enrichir_calendrier_preserve_colonnes(config_forex):
    """Les colonnes originales sont préservées."""
    idx = pd.date_range("2026-01-15", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.arange(5.0)}, index=idx)

    out = enrichir_calendrier(df, "H1", config_forex)
    assert "open" in out.columns
    assert "ctx_day_of_week" in out.columns
    assert len(out) == len(df)


def test_enrichir_calendrier_ne_modifie_pas_entree(config_forex):
    """Le DataFrame en entrée n'est pas modifié."""
    idx = pd.date_range("2026-01-15", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": np.arange(5.0)}, index=idx)
    df_avant = df.copy()

    _ = enrichir_calendrier(df, "H1", config_forex)
    pd.testing.assert_frame_equal(df, df_avant)


def test_calendrier_deterministe(config_forex):
    """Deux appels produisent le même résultat."""
    idx = pd.date_range("2026-01-15", periods=24, freq="1h", tz="UTC")
    cal1 = calculer_calendrier(idx, "H1", config_forex)
    cal2 = calculer_calendrier(idx, "H1", config_forex)
    pd.testing.assert_frame_equal(cal1, cal2)
