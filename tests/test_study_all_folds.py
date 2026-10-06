"""Effets, épisodes et exports de toutes les périodes."""
import numpy as np
import pandas as pd
import pytest
from regime_lib.study import Study
from regime_lib.study.evaluation import par_modalite_tests, stabilite_modalites


def test_known_effect_and_episode_counts():
    feature = pd.Series(['A', 'A', 'B', 'B'] * 3)
    target = pd.Series([3., 3., 1., 1.] * 3)
    table = par_modalite_tests(feature, target)
    assert table.loc['A', 'mean_diff'] == 2
    assert table.loc['A', 'median_diff'] == 2
    assert table.loc['A', 'cliffs_delta'] == 1
    assert table.loc['B', 'cliffs_delta'] == -1
    assert table.loc['A', 'n_episodes_a'] == 3
    assert table.loc['A', 'n_episodes_b'] == 3
    assert table.loc['A', 'status'] == 'descriptive'


def test_missing_target_splits_episode():
    feature = pd.Series(['A'] * 5 + ['B'] * 3)
    target = pd.Series([1., 1., np.nan, 1., 1., 0., 0., 0.])
    table = par_modalite_tests(feature, target)
    assert table.loc['A', 'n_a'] == 4
    assert table.loc['A', 'n_episodes_a'] == 2
    assert table.loc['A', 'status'] == 'insufficient_episodes'


def test_absent_comparator_has_no_effect():
    table = par_modalite_tests(pd.Series(['A'] * 10), pd.Series(range(10)))
    assert table.loc['A', 'n_b'] == 0
    assert np.isnan(table.loc['A', 'mean_diff'])
    assert np.isnan(table.loc['A', 'cliffs_delta'])
    assert table.loc['A', 'status'] == 'insufficient_samples'


def test_stability_exposes_missing_and_opposite_effects():
    tables = {}
    for fold, effect in enumerate([2., -2.]):
        tables[(fold, 'f', 'ret_fwd_1')] = pd.DataFrame(
            {'mean_diff': [effect], 'status': ['descriptive']}, index=['A'])
    row = stabilite_modalites(tables, 3).iloc[0]
    assert row.n_folds_total == 3
    assert row.n_folds_present == 2
    assert row.n_positive == row.n_negative == 1
    assert row.effect_mean == 0
    assert row.effect_std == pytest.approx(np.sqrt(8))
    assert row.sign_stability == .5


@pytest.mark.parametrize('enabled', [True, False])
def test_all_folds_export_and_access(tmp_path, enabled):
    index = pd.date_range('2026-01-01', periods=100, freq='h', tz='UTC')
    study = Study('unused', 'unused', split='walk_forward', n_folds=3,
                  train_ratio=.5, tests_modalite=enabled)
    study.df_features = pd.DataFrame({'adx__regime': ['A', 'B'] * 50}, index=index)
    study.df_features.attrs['series_identity'] = {'timeframe': 'H1'}
    study.df_cibles = pd.DataFrame({'ret_fwd_1': [1., -1.] * 50}, index=index)
    study.df_fin_cibles = pd.DataFrame({'ret_fwd_1': pd.Series(index, index=index).shift(-1)})
    study._evaluer()
    study.save(tmp_path)
    details = pd.read_parquet(tmp_path / 'study_conditionnel.parquet')
    assert set(details.fold) == {0, 1, 2}
    assert len(details) == 6
    assert 'n_episodes' in details
    report = (tmp_path / 'study_report.md').read_text()
    assert 'Fold 2' in report
    if enabled:
        tests = pd.read_parquet(tmp_path / 'study_tests_modalite.parquet')
        assert set(tests.fold) == {0, 1, 2}
        assert len(study.resultats_tests) == 3
        assert study.tests_modalite_pour('adx__regime', 'ret_fwd_1', fold=2).loc['A', 'mean_diff'] == 2
        stability = pd.read_parquet(tmp_path / 'study_stability.parquet')
        assert stability.n_folds_estimable.tolist() == [3, 3]
        assert stability.sign_stability.tolist() == [1., 1.]
    else:
        assert study.resultats_tests is None
        assert study.stabilite_modalites.empty
        assert not (tmp_path / 'study_tests_modalite.parquet').exists()
