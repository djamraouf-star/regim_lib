"""Calculs descriptifs contrôlés à la main, sans modèle statistique appris."""
import numpy as np
import pandas as pd
import pytest

from regime_lib.study.report import (
    aligner_sur_features, matrice_transition, persistance, rapport_complet,
)


@pytest.mark.parametrize("values", [[0, 0, -1, 0, 0], [0, 0, np.nan, 0, 0]])
def test_unknown_splits_episode_and_never_counts_as_state(values):
    states = pd.Series(values)
    episodes = persistance(states)
    assert episodes.index.tolist() == [0]
    assert episodes.loc[0, "episodes"] == 2
    assert episodes.loc[0, "duree_med"] == 2
    matrix = matrice_transition(states)
    assert matrix.attrs["counts"].iloc[0, 0] == 2
    assert matrix.iloc[0, 0] == 1


def test_gap_splits_episode_and_transition():
    index = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    states = pd.Series([0, 0, 0, 1, 1], index=index.delete(2))
    episodes = persistance(states, timeframe="H1")
    assert episodes.loc[0, "episodes"] == 2
    assert episodes.loc[0, "duree_moy"] == 1.5
    assert episodes.loc[1, "duree_med"] == 2
    matrix = matrice_transition(states, timeframe="H1")
    np.testing.assert_array_equal(matrix.attrs["counts"], [[1, 1], [0, 1]])
    np.testing.assert_allclose(matrix, [[.5, .5], [0, 1]])


def test_gap_between_different_states_creates_no_transition():
    index = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC").delete(2)
    states = pd.Series([0, 0, 1, 1], index=index)
    matrix = matrice_transition(states, timeframe="H1")
    np.testing.assert_array_equal(matrix.attrs["counts"], [[1, 0], [0, 1]])


def test_irregular_index_requires_declared_frequency():
    index = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC").delete(2)
    states = pd.Series([0] * 4, index=index)
    with pytest.raises(ValueError, match="ambigu"):
        persistance(states)
    with pytest.raises(ValueError, match="ambigu"):
        matrice_transition(states)


def test_state_without_outgoing_transition_is_not_estimable():
    matrix = matrice_transition(pd.Series([0, 0, -1, 1]))
    assert matrix.loc["STATE_1"].isna().all()
    assert matrix.attrs["counts"].loc["STATE_1"].sum() == 0


@pytest.mark.parametrize("values", [[], [-1, -1], [np.nan, np.nan]])
def test_empty_or_invalid_support(values):
    assert persistance(pd.Series(values, dtype=float)).empty
    assert matrice_transition(pd.Series(values, dtype=float)).empty


def test_generic_report_keeps_semantic_labels_and_counts(tmp_path):
    index = pd.date_range("2026-01-01", periods=64, freq="h", tz="UTC", name="timestamp")
    close = 100 + np.sin(np.arange(64))
    prices = pd.DataFrame({"open": close, "high": close+1, "low": close-1,
                           "close": close, "volume": 1.}, index=index)
    prices.attrs.update(asset="EURUSD", timeframe="H1")
    run = pd.DataFrame({"timestamp": index, "asset": "EURUSD", "timeframe": "H1",
                        "method": "adx", "regime": ["TENDANCE", "RANGE"] * 32,
                        "confidence": .8})
    run.to_parquet(tmp_path / "regimes.parquet", index=False)
    prices.to_parquet(tmp_path / "prices.parquet")
    result = rapport_complet(tmp_path, str(tmp_path / "prices.parquet"))
    assert "TENDANCE" in result and "RANGE" in result
    assert "STATE_" not in result
    assert "CALME" not in result and "AGITÉ" not in result
    assert "n_support_descriptif : 44" in result
    assert "Effectifs des transitions" in result
    _, states, _ = aligner_sur_features(run, prices)
    matrix = matrice_transition(states)
    assert matrix.loc["TENDANCE", "RANGE"] == 1
    assert matrix.attrs["counts"].loc["TENDANCE", "RANGE"] == 32
    assert persistance(states).episodes.sum() == 64
