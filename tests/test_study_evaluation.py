"""Tests des métriques et tests statistiques."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from regime_lib.study.evaluation import (
    hit_rate_directionnel,
    ic_spearman,
    stats_conditionnelles,
    mann_whitney_test,
    welch_test,
    z_proportions_test,
    batterie_tests,
    par_modalite_tests,
)


# ---------------------------------------------------------------------------
# IC Spearman
# ---------------------------------------------------------------------------
def test_ic_correlation_parfaite():
    x = pd.Series(np.arange(100, dtype=float))
    y = pd.Series(np.arange(100, dtype=float) * 2 + 1)
    res = ic_spearman(x, y)
    assert res["ic"] == pytest.approx(1.0, abs=1e-6)
    assert res["n"] == 100


def test_ic_correlation_inversee():
    x = pd.Series(np.arange(100, dtype=float))
    y = pd.Series(-np.arange(100, dtype=float))
    res = ic_spearman(x, y)
    assert res["ic"] == pytest.approx(-1.0, abs=1e-6)


def test_ic_independance():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 1, 1000))
    y = pd.Series(rng.normal(0, 1, 1000))
    res = ic_spearman(x, y)
    assert abs(res["ic"]) < 0.1


def test_ic_trop_peu_de_points():
    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([1.0, 2.0, 3.0])
    res = ic_spearman(x, y)
    assert np.isnan(res["ic"])


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------
def test_hit_rate_parfait():
    feature = pd.Series([1.0] * 50)
    cible = pd.Series([0.01] * 50)
    res = hit_rate_directionnel(feature, cible)
    assert res["hit_rate"] == 1.0


def test_hit_rate_catégoriel_renvoie_nan():
    feature = pd.Series(["A", "B"] * 50)
    cible = pd.Series([0.01, -0.01] * 50)
    res = hit_rate_directionnel(feature, cible)
    assert np.isnan(res["hit_rate"])


# ---------------------------------------------------------------------------
# Tests individuels
# ---------------------------------------------------------------------------
def test_welch_moyennes_egales():
    a = np.random.default_rng(0).normal(0, 1, 500)
    b = np.random.default_rng(1).normal(0, 1, 500)
    res = welch_test(a, b)
    assert res["pvalue"] > 0.05


def test_welch_moyennes_differentes():
    a = np.random.default_rng(0).normal(0, 1, 500)
    b = np.random.default_rng(1).normal(1, 1, 500)
    res = welch_test(a, b)
    assert res["pvalue"] < 1e-10


def test_welch_ne_perd_pas_de_precision_avec_un_decalage_commun():
    offset = 1e12
    a = offset + np.tile([0.0, 0.001, -0.001, 0.002], 100)
    b = offset + np.tile([0.001, 0.002, 0.0, 0.003], 100)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        res = welch_test(a, b)

    assert not captured
    assert np.isfinite(res["t_stat"])
    assert np.isfinite(res["pvalue"])
    assert res["pvalue"] == pytest.approx(0.0, abs=1e-12)


def test_welch_groupes_constants():
    identiques = welch_test(np.ones(10), np.ones(10))
    differents = welch_test(np.ones(10), np.full(10, 2.0))

    assert identiques["t_stat"] == 0.0
    assert identiques["pvalue"] == 1.0
    assert np.isinf(differents["t_stat"])
    assert differents["pvalue"] == 0.0


def test_mann_whitney_detecte_decalage():
    a = np.arange(500, dtype=float)
    b = np.arange(500, dtype=float) + 100
    res = mann_whitney_test(a, b)
    assert res["pvalue"] < 1e-10


def test_z_proportions_egales():
    res = z_proportions_test(500, 250, 500, 250)
    assert res["pvalue"] == pytest.approx(1.0)


def test_z_proportions_differentes():
    res = z_proportions_test(500, 250, 500, 400)
    assert res["pvalue"] < 1e-10


# ---------------------------------------------------------------------------
# Batterie
# ---------------------------------------------------------------------------
def test_batterie_coherente():
    """Les 3 tests doivent donner des signaux cohérents."""
    a = np.random.default_rng(0).normal(0.0, 1.0, 500)
    b = np.random.default_rng(1).normal(0.5, 1.0, 500)
    res = batterie_tests(a, b, nom_cible="ret_fwd_5")
    assert res["welch_p"] < 0.001
    assert res["mw_p"] < 0.001
    assert res["z_p"] < 0.001


def test_batterie_convention_vol():
    """
    Sur une cible de type vol_fwd, la convention de succès utilise la
    médiane globale. Les deux groupes doivent avoir des fréquences
    différentes.
    """
    rng = np.random.default_rng(0)
    a = rng.normal(0.001, 0.0001, 500)   # groupe calme
    b = rng.normal(0.002, 0.0001, 500)   # groupe agité
    res = batterie_tests(a, b, nom_cible="vol_fwd_20")
    # p_a_pos << p_b_pos (le groupe a est plus souvent sous la médiane)
    assert res["p_a_pos"] < res["p_b_pos"]
    assert res["z_p"] < 0.001


def test_batterie_convention_ret():
    """Sur ret_fwd, la convention de succès est cible > 0."""
    a = np.random.default_rng(0).normal(0.01, 0.01, 500)
    b = np.random.default_rng(1).normal(-0.01, 0.01, 500)
    res = batterie_tests(a, b, nom_cible="ret_fwd_5")
    assert res["p_a_pos"] > 0.7
    assert res["p_b_pos"] < 0.3


# ---------------------------------------------------------------------------
# Par modalité
# ---------------------------------------------------------------------------
def test_modalite_deux_groupes():
    rng = np.random.default_rng(0)
    feature = pd.Series(["A"] * 500 + ["B"] * 500)
    cible = pd.Series(np.concatenate([
        rng.normal(0.0, 1.0, 500),
        rng.normal(1.0, 1.0, 500),
    ]))
    tab = par_modalite_tests(feature, cible, nom_cible="ret_fwd_5")
    assert "A" in tab.index and "B" in tab.index
    assert tab.loc["A", "welch_p"] < 0.001
    assert tab.loc["B", "welch_p"] < 0.001


def test_stats_conditionnelles():
    feature = pd.Series(["A"] * 100 + ["B"] * 100)
    cible = pd.Series(np.concatenate([
        np.zeros(100), np.ones(100),
    ]))
    stats = stats_conditionnelles(feature, cible)
    assert stats.loc["A", "mean"] == 0.0
    assert stats.loc["B", "mean"] == 1.0
    assert stats.loc["A", "n"] == 100
