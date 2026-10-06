"""Contrats de rigueur issus de l'audit du module study.

Les défauts initiaux sont corrigés et ces contrats doivent rester passants.
Aucun calcul métier n’est simulé ; les rapports utilisent des parquets
synthétiques locaux.
"""

import numpy as np
import pandas as pd
import pytest

from regime_lib.study import Study
from regime_lib.study.evaluation import ic_spearman
from regime_lib.study.report import aligner_sur_features, matrice_transition, rapport_complet
from regime_lib.study.split import split_holdout, split_walk_forward


@pytest.fixture
def run_hmm(tmp_path):
    index = pd.date_range(
        "2026-01-01", periods=120, freq="h", tz="UTC", name="timestamp"
    )
    close = 100 + np.sin(np.arange(120) / 4) + np.arange(120) / 100
    prices = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1.0}, index=index,
    )
    prices.attrs.update(asset="EURUSD", timeframe="H1")
    run = pd.DataFrame({
        "timestamp": index, "asset": "EURUSD", "timeframe": "H1",
        "method": "hmm_gaussian", "params_hash": "fixture",
        "regime": ["STATE_0", "STATE_1"] * 60, "confidence": 0.8,
        "is_valid": True, "is_partial": False, "is_warmup": False,
    })
    source = tmp_path / "prices.parquet"
    prices.to_parquet(source)
    run.to_parquet(tmp_path / "regimes.parquet", index=False)
    return tmp_path, source, prices, run


def test_rapport_accepte_une_identite_compatible(run_hmm):
    directory, source, _, _ = run_hmm
    result = rapport_complet(directory, str(source), method="hmm_gaussian")
    assert "RAPPORT D'ANALYSE" in result
    assert "STATE_0" in result
    assert "STATE_1" in result


def test_rapport_refuse_un_autre_actif(run_hmm):
    directory, source, prices, _ = run_hmm
    prices.attrs["asset"] = "GBPUSD"
    prices.to_parquet(source)
    with pytest.raises(ValueError, match="(?i)actif|asset|identité"):
        rapport_complet(directory, str(source), method="hmm_gaussian")


@pytest.mark.parametrize("flag,value", [
    ("is_valid", False), ("is_partial", True), ("is_warmup", True),
])
def test_alignement_exclut_les_observations_invalides(run_hmm, flag, value):
    _, _, prices, run = run_hmm
    invalid_index = prices.index[40]
    run.loc[40, flag] = value
    features, states, confidence = aligner_sur_features(run, prices)
    # Une ligne rejetée peut être supprimée ou masquée, mais jamais estimée.
    usable = features.notna().all(axis=1) & states.ge(0) & confidence.notna()
    assert invalid_index not in features.index[usable]
    assert usable.sum() == len(prices) - 1


def test_transitions_adjacentes_conservent_les_effectifs():
    matrix = matrice_transition(pd.Series([0, 0, 1, 1, 0]))
    expected = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]],
        index=["STATE_0", "STATE_1"], columns=["STATE_0", "STATE_1"],
    )
    pd.testing.assert_frame_equal(matrix, expected)


def test_transition_ne_traverse_pas_un_etat_inconnu():
    matrix = matrice_transition(pd.Series([0, 0, -1, 1, 1]))
    assert matrix.loc["STATE_0", "STATE_1"] == 0.0
    assert matrix.loc["STATE_0", "STATE_0"] == 1.0
    assert matrix.loc["STATE_1", "STATE_1"] == 1.0


def test_ic_non_applicable_aux_categories_meme_apres_renommage():
    feature = pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    target = pd.Series(np.arange(30, dtype=float))
    for labels in (feature, feature.replace({"A": "B", "B": "A"})):
        result = ic_spearman(labels, target)
        assert result["n"] == 30
        assert np.isnan(result["ic"])
        assert np.isnan(result["pvalue"])


def test_study_ne_presente_pas_la_confidence_comme_signal_directionnel(run_hmm):
    directory, source, _, _ = run_hmm
    study = Study(
        directory / "regimes.parquet", source,
        features=["confidence"], targets=["vol_fwd_5"],
        methodes_causales_only=False, tests_modalite=False,
    ).run()
    assert study.resultats["n_test"].min() >= 10
    assert study.resultats[["hit_train", "hit_test"]].isna().all().all()


def test_walk_forward_applique_la_proportion_train():
    index = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    for ratio in (0.2, 0.8):
        folds = split_walk_forward(index, n_folds=3, train_ratio=ratio)
        assert len(folds) == 3
        for train, test in folds:
            assert train[-1] < test[0]
            assert len(train) / (len(train) + len(test)) == pytest.approx(
                ratio, abs=1 / (len(train) + len(test)),
            )


@pytest.mark.parametrize("schema", [split_holdout, split_walk_forward])
def test_split_refuse_un_embargo_sans_test(schema):
    index = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="(?i)embargo|vide|test|fold"):
        schema(index, embargo=len(index))


def test_rapport_selectionne_les_timeframes_de_la_methode(run_hmm):
    directory, source, _, run = run_hmm
    baseline = rapport_complet(directory, str(source), method="hmm_gaussian")
    other = run.iloc[:1].assign(method="adx", timeframe="M30")
    pd.concat([run, other]).to_parquet(directory / "regimes.parquet", index=False)
    result = rapport_complet(directory, str(source), method="hmm_gaussian")
    assert result == baseline


def test_study_refuse_un_decoupage_vide_avant_export(run_hmm):
    directory, source, _, _ = run_hmm
    study = Study(
        directory / "regimes.parquet", source,
        targets=["ret_fwd_5"], methodes_causales_only=False,
        split="walk_forward", embargo=120, tests_modalite=False,
    )
    # Contrat : refuser lors du calcul, avant toute écriture partielle.
    with pytest.raises(ValueError, match="(?i)embargo|vide|test|fold"):
        study.run()
