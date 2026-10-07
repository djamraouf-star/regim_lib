"""Le criblage par contexte appartient à Study, sans script externe."""
import json

import numpy as np
import pandas as pd
import pytest

from regime_lib.study import Study
from regime_lib.study.inference import InferenceConfig, adjust_by, block_indices


def manual(*, inference=None, tests=True, split="in_sample"):
    index = pd.date_range("2026-01-01", periods=96, freq="h", tz="UTC")
    study = Study("unused", "unused", targets=["ret_fwd_1"],
                  stratify_by="ctx_session", inference=inference, tests_modalite=tests,
                  split=split, **({"n_folds": 2, "train_ratio": .5} if split == "walk_forward" else {}))
    study.df_features = pd.DataFrame({"adx__regime": ["A", "B"] * 48}, index=index)
    study.df_features.attrs["series_identity"] = {"timeframe": "H1"}
    study.df_strates = pd.DataFrame({"ctx_session": (["ASIE"] * 24 + ["LONDRES"] * 24) * 2}, index=index)
    study.df_cibles = pd.DataFrame({"ret_fwd_1": np.tile([3., 1.], 48) + np.sin(np.arange(96))}, index=index)
    study.df_fin_cibles = pd.DataFrame({"ret_fwd_1": pd.Series(index, index=index).shift(-1)})
    return study


def test_session_isolation_and_global_family():
    cfg = InferenceConfig(block_size=2, n_resamples=199, min_group_size=2)
    study = manual(inference=cfg)
    study._evaluer()
    before = study.resultats_stratifies.query("strate == 'LONDRES'").copy()
    study.df_cibles.loc[study.df_strates.ctx_session.eq("ASIE"), "ret_fwd_1"] *= -100
    study._evaluer()
    after = study.resultats_stratifies.query("strate == 'LONDRES'")
    local = ["n_a", "n_b", "mean_diff", "welch_p", "mw_p", "z_p", "block_p", "mean_diff_ci_low", "mean_diff_ci_high"]
    pd.testing.assert_frame_equal(before[local], after[local])
    global_table = study.tests_modalite_pour("adx__regime", "ret_fwd_1")
    all_p = [*global_table.block_p, *study.resultats_stratifies.block_p]
    all_q = [*global_table.block_p_by, *study.resultats_stratifies.block_p_by]
    np.testing.assert_allclose(all_q, adjust_by(all_p), equal_nan=True)
    assert set(study.resultats_stratifies.family_size) == {6}
    assert set(after.n_a) == {24}
    assert set(after.n_b) == {24}


def test_masked_grid_keeps_separate_episodes_and_blocks():
    study = manual()
    study.df_features.iloc[:, 0] = "A"
    study._evaluer()
    asia = study.resultats_stratifies.query("strate == 'ASIE'").iloc[0]
    assert asia.n_episodes_a == 2
    assert asia.n_b == 0
    assert asia.status == "insufficient_samples"
    valid = study.df_strates.ctx_session.eq("ASIE").to_numpy()
    blocks = block_indices(study.df_features.index, valid, 3, timeframe="H1")
    assert not ((blocks.min(axis=1) < 24) & (blocks.max(axis=1) >= 48)).any()


@pytest.mark.parametrize("tests", [True, False])
def test_all_folds_exports_and_disabled_cleanup(tmp_path, tests):
    study = manual(tests=tests, split="walk_forward")
    study._evaluer()
    study.save(tmp_path)
    exported = pd.read_parquet(tmp_path / "study_stratified.parquet")
    assert set(exported.fold) == {0, 1}
    assert set(exported.strate) == {"ASIE", "LONDRES"}
    assert ("welch_p" in exported) == tests
    assert "Analyse stratifiée : ctx_session" in (tmp_path / "study_report.md").read_text()
    metadata = json.loads((tmp_path / "study_metadata.json").read_text())
    assert metadata["configuration"]["stratify_by"] == "ctx_session"
    assert metadata["strata"] is not None
    study.stratify_by = None
    study._evaluer()
    study.save(tmp_path)
    assert not (tmp_path / "study_stratified.parquet").exists()
    assert not (tmp_path / "study_strata_coverage.parquet").exists()


def test_missing_outside_and_empty_support(tmp_path):
    study = manual()
    study.df_strates.iloc[:40, 0] = None
    study.df_strates.iloc[40:, 0] = "HORS_SESSION"
    study._evaluer()
    assert study.resultats_stratifies.empty
    coverage = study.couverture_strates.iloc[0]
    assert coverage.n_contexte_absent == 40
    assert coverage.n_hors_session == 56
    assert coverage.n_commun == 0
    assert study.resultats.n_test.iloc[0] == 96
    study.save(tmp_path)
    assert "Aucune estimation stratifiée" in (tmp_path / "study_report.md").read_text()


@pytest.fixture
def inputs(tmp_path):
    index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC", name="timestamp")
    close = 100 + np.arange(48, dtype=float)
    prices = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": 1.}, index=index)
    prices.attrs.update(asset="EURUSD", timeframe="H1")
    rows = pd.DataFrame({"timestamp": index, "asset": "EURUSD", "timeframe": "H1",
                         "method": "adx", "params_hash": "one", "regime": ["A", "B"] * 24,
                         "ctx_session": (["ASIE"] * 12 + ["LONDRES"] * 12) * 2})
    return tmp_path, prices, rows


def run_inputs(inputs, **kwargs):
    directory, prices, rows = inputs
    path = directory / "regimes.parquet"
    rows.to_parquet(path, index=False)
    return Study(path, "unused", ohlcv_data=prices, targets=["ret_fwd_5"], **kwargs).run()


def test_run_preserves_targets_and_global_results_and_tracks_mutations(inputs):
    global_study = run_inputs(inputs)
    study = run_inputs(inputs, stratify_by="ctx_session")
    pd.testing.assert_frame_equal(study.df_cibles, global_study.df_cibles)
    pd.testing.assert_frame_equal(study.resultats, global_study.resultats)
    # À 11h ASIE, la cible reste le rendement à 16h, en LONDRES.
    assert study.df_cibles.iloc[11, 0] == pytest.approx(np.log(116 / 111))
    study.save(inputs[0] / "output")
    study.df_strates.iloc[0, 0] = "LONDRES"
    with pytest.raises(ValueError, match="changé"):
        study.save(inputs[0] / "output")


@pytest.mark.parametrize("problem", ["absent", "conflict", "missing_conflict", "invalid"])
def test_context_validation(inputs, problem):
    directory, prices, rows = inputs
    other = rows.assign(method="er_kaufman")
    if problem == "absent":
        rows = rows.drop(columns="ctx_session")
    elif problem == "invalid":
        rows["ctx_session"] = 3
    else:
        other.loc[0, "ctx_session"] = None if problem == "missing_conflict" else "LONDRES"
        rows = pd.concat([rows, other], ignore_index=True)
    with pytest.raises(ValueError, match="[Cc]ontexte"):
        run_inputs((directory, prices, rows), stratify_by="ctx_session")


def test_context_ignores_unselected_configuration(inputs):
    directory, prices, rows = inputs
    other = rows.assign(params_hash="two", ctx_session="OTHER")
    study = run_inputs((directory, prices, pd.concat([rows, other])),
                       stratify_by="ctx_session", configurations={"adx": "one"})
    assert set(study.resultats_stratifies.strate) == {"ASIE", "LONDRES"}


@pytest.mark.parametrize("column", ["confidence", "invalid"])
def test_unknown_stratification_rejected(column):
    with pytest.raises(ValueError, match="stratify_by"):
        Study("unused", "unused", stratify_by=column)


@pytest.mark.parametrize('column,values', [
    ('ctx_day_of_week', [0, 1] * 24),
    ('ctx_is_holiday', [True, False] * 24),
    ('ctx_event_type', ['news', 'none'] * 24),
])
def test_other_discrete_contexts(inputs, column, values):
    directory, prices, rows = inputs
    rows[column] = values
    study = run_inputs((directory, prices, rows), stratify_by=column)
    assert set(study.resultats_stratifies.strate) == {str(value) for value in values}


def test_short_sessions_do_not_form_artificial_bootstrap_blocks():
    study = manual(inference=InferenceConfig(block_size=3, n_resamples=199))
    study.df_strates['ctx_session'] = ['ASIE', 'ASIE', 'LONDRES', 'LONDRES'] * 24
    study._evaluer()
    assert set(study.resultats_stratifies.inference_status) == {'insufficient_support'}
    assert study.resultats_stratifies.block_p.isna().all()


def test_stratified_results_mutation_prevents_export(inputs):
    study = run_inputs(inputs, stratify_by='ctx_session')
    study.resultats_stratifies.loc[0, 'mean_diff'] = 42.
    with pytest.raises(ValueError, match='changé'):
        study.save(inputs[0] / 'output')
