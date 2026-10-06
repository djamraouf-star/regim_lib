"""Même contrat d'entrée pour les études prédictives et descriptives."""

import numpy as np
import pandas as pd
import pytest

from regime_lib.study import Study
from regime_lib.study.features import extraire_features
from regime_lib.study.report import rapport_complet


@pytest.fixture
def inputs(tmp_path):
    index = pd.date_range("2026-01-01", periods=64, freq="h", tz="UTC", name="timestamp")
    close = 100 + np.sin(np.arange(64))
    prices = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": 1.}, index=index)
    prices.attrs.update(asset="EURUSD", timeframe="H1")
    regimes = pd.DataFrame({"timestamp": index, "asset": "EURUSD", "timeframe": "H1",
                           "method": "hmm_gaussian", "params_hash": "one",
                           "regime": ["STATE_0", "STATE_1"] * 32, "confidence": .8})
    return tmp_path, prices, regimes


def execute(inputs, mode):
    directory, prices, regimes = inputs
    regimes.to_parquet(directory / "regimes.parquet", index=False)
    prices.to_parquet(directory / "prices.parquet")
    if mode == "study":
        return Study(directory / "regimes.parquet", directory / "prices.parquet",
                     methodes_causales_only=False, tests_modalite=False).run()
    return rapport_complet(directory, str(directory / "prices.parquet"))


@pytest.mark.parametrize("mode", ["study", "report"])
@pytest.mark.parametrize("problem,match", [
    ("asset", "Actif OHLCV"), ("source", "source"),
    ("configuration", "Configurations ambiguës"), ("duplicate", "doublons"),
    ("missing_price", "timestamps"), ("naive_index", "UTC"),
    ("missing_identity", "Identité de série absente"),
    ("bad_flag", "booléen"),
])
def test_shared_rejections(inputs, mode, problem, match):
    directory, prices, regimes = inputs
    if problem == "asset":
        prices.attrs["asset"] = "GBPUSD"
    elif problem == "source":
        regimes["source"] = "feed-A"
        prices.attrs["source"] = "feed-B"
    elif problem == "configuration":
        regimes.loc[20, "params_hash"] = "two"
    elif problem == "duplicate":
        regimes = pd.concat([regimes, regimes.iloc[:1]])
    elif problem == "missing_price":
        prices = prices.drop(prices.index[30])
    elif problem == "naive_index":
        regimes["timestamp"] = regimes.timestamp.dt.tz_localize(None)
    elif problem == "missing_identity":
        regimes = regimes.drop(columns="asset")
    elif problem == "bad_flag":
        regimes["is_valid"] = "yes"
    with pytest.raises(ValueError, match=match):
        execute((directory, prices, regimes), mode)


def test_configuration_selection_precedes_identity_check(inputs):
    directory, prices, regimes = inputs
    other = regimes.assign(params_hash="two", source="feed-B")
    regimes["source"] = "feed-A"
    combined = pd.concat([regimes, other])
    extracted = extraire_features(combined, methodes_causales_only=False,
                                  configurations={"hmm_gaussian": "one"})
    assert extracted.attrs["series_identity"]["source"] == "feed-A"
    combined.to_parquet(directory / "regimes.parquet", index=False)
    prices.attrs["source"] = "feed-A"
    prices.to_parquet(directory / "prices.parquet")
    result = rapport_complet(directory, str(directory / "prices.parquet"), configuration="one")
    assert "RAPPORT D'ANALYSE" in result


@pytest.mark.parametrize("mode", ["study", "report"])
def test_empty_valid_support_is_explicit(inputs, mode):
    directory, prices, regimes = inputs
    regimes["is_valid"] = False
    result = execute(inputs, mode)
    if mode == "study":
        assert result.couverture.n_valide.tolist() == [0]
        assert result.couverture.n_exclu.tolist() == [64]
        assert result.resultats.n_test.tolist() == [0]
    else:
        assert "n_valide : 0" in result
        assert "n_exclu : 64" in result
        assert "Aucune estimation : support valide vide" in result


def test_report_excludes_invalid_prices(inputs):
    directory, prices, regimes = inputs
    prices["is_valid"] = False
    result = execute(inputs, "report")
    assert "n_support_descriptif : 0" in result
    assert "Aucune estimation : support valide vide" in result


def test_coverage_counts_overlapping_exclusions_once(inputs):
    _, _, regimes = inputs
    regimes["is_valid"] = True
    regimes["is_partial"] = False
    regimes.loc[30, ["is_valid", "is_partial"]] = [False, True]
    out = extraire_features(regimes, methodes_causales_only=False)
    coverage = out.attrs["exclusions"]["hmm_gaussian"]
    assert coverage["n_invalide"] == coverage["n_partiel"] == 1
    assert coverage["n_exclu"] == 1
    assert coverage["n_valide"] == 63
