"""Bootstrap temporel reproductible et famille globale BY."""
import numpy as np
import pandas as pd
import pytest

from regime_lib.study import InferenceConfig, Study
from regime_lib.study.evaluation import par_modalite_tests
from regime_lib.study.inference import adjust_by, block_indices, bootstrap_modalities, correct_family


def sample(seed=0, effect=0., n=800):
    rng = np.random.default_rng(seed)
    labels = np.repeat(rng.integers(0, 2, size=n // 10), 10)
    noise = rng.normal(size=n)
    for i in range(1, n):
        noise[i] += .65 * noise[i-1]
    index = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
    return pd.Series(labels, index=index), pd.Series(noise + labels * effect, index=index)


def infer(f, y, config=None):
    table = par_modalite_tests(f, y, timeframe='H1')
    return bootstrap_modalities(f, y, table, config or InferenceConfig(20, n_resamples=199),
                               key=(0, 'regime', 'ret_fwd_5'), horizon=5, timeframe='H1')


def test_reproducible_and_known_effect():
    f, y = sample(effect=3.)
    first = infer(f, y)
    pd.testing.assert_frame_equal(first, infer(f, y))
    assert first.loc[1, 'inference_status'] == 'ok'
    assert first.loc[1, 'mean_diff_ci_low'] > 0
    assert first.loc[1, 'mean_diff_ci_low'] < 3 < first.loc[1, 'mean_diff_ci_high']
    tables = {(0, 'f', 't'): first}
    correct_family(tables, .95)
    assert first.reject_by.all()


def test_dependent_null_simulation_is_not_systematically_significant():
    # Contrôle de plausibilité déterministe, pas une garantie de couverture.
    exclusions = 0
    for seed in range(20):
        f, y = sample(seed=seed)
        row = infer(f, y).loc[1]
        assert row.inference_status == 'ok'
        exclusions += not (row.mean_diff_ci_low <= 0 <= row.mean_diff_ci_high)
    assert exclusions <= 5


def test_by_known_values_nan_and_global_family():
    # H_3 = 11/6 ; valeurs ordonnées .01, .04, .2.
    np.testing.assert_allclose(adjust_by([.01, .04, .2]), [.055, .11, 11/30])
    assert np.isnan(adjust_by([np.nan])[0])
    a = pd.DataFrame({'block_p': [.01]}, index=['A'])
    b = pd.DataFrame({'block_p': [np.nan]}, index=['B'])
    correct_family({(0, 'f', 't'): a, (1, 'f', 't'): b}, .95)
    assert a.loc['A', 'family_size'] == 2
    assert a.loc['A', 'block_p_by'] == pytest.approx(.03)
    assert pd.isna(b.loc['B', 'reject_by'])


def test_blocks_do_not_bridge_holes_or_invalid_rows():
    index = pd.date_range('2026-01-01', periods=9, freq='h', tz='UTC').delete(4)
    blocks = block_indices(index, np.ones(8, dtype=bool), 3, timeframe='H1')
    np.testing.assert_array_equal(blocks, [[0,1,2], [1,2,3], [4,5,6], [5,6,7]])
    valid = np.ones(8, dtype=bool); valid[1] = False
    assert block_indices(index, valid, 3, timeframe='H1').size == 0


def test_insufficient_support_is_not_a_rejection():
    f, y = sample(n=100)
    result = infer(f, y, InferenceConfig(30, n_resamples=199))
    assert result.inference_status.eq('insufficient_support').all()
    assert result.block_p.isna().all()
    assert result.mean_diff_ci_low.isna().all()


@pytest.mark.parametrize('kwargs', [{'block_size': 0}, {'block_size': True},
    {'block_size': 2, 'n_resamples': 10}, {'block_size': 2, 'confidence_level': 1},
    {'block_size': 2, 'seed': -1}])
def test_configuration_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        InferenceConfig(**kwargs)


def test_horizon_guard_and_study_exports(tmp_path):
    with pytest.raises(ValueError, match='block_size'):
        Study('unused', 'unused', targets=['ret_fwd_20'], inference=InferenceConfig(5))
    f, y = sample(effect=2.)
    study = Study('unused', 'unused', inference=InferenceConfig(20, n_resamples=199))
    study.df_features = pd.DataFrame({'regime': f})
    study.df_features.attrs['series_identity'] = {'timeframe': 'H1'}
    study.df_cibles = pd.DataFrame({'ret_fwd_5': y})
    study._evaluer()
    study.save(tmp_path)
    result = pd.read_parquet(tmp_path / 'study_tests_modalite.parquet')
    assert result.inference_status.eq('ok').all()
    assert result.family_size.eq(2).all()
    assert (tmp_path / 'study_inference.json').exists()
    assert 'block_p_by' in (tmp_path / 'study_report.md').read_text()


def test_degenerate_bootstrap_does_not_claim_certainty():
    f, _ = sample()
    y = f.astype(float) * 2
    result = infer(f, y)
    assert result.inference_status.eq('degenerate_bootstrap').all()
    assert result.block_p.isna().all()


def test_declared_schedule_defines_adjacency():
    index = pd.date_range('2026-01-01', periods=9, freq='h', tz='UTC').delete(4)
    blocks = block_indices(index, np.ones(8, dtype=bool), 3,
                           timeframe='H1', expected_index=index)
    assert [2, 3, 4] in blocks.tolist()


def test_block_interval_reflects_autocorrelation_on_fixed_fixture():
    f, y = sample(seed=4)
    table = par_modalite_tests(f, y, timeframe='H1')
    results = [bootstrap_modalities(f, y, table, InferenceConfig(size, n_resamples=399),
               key='comparison', horizon=1, timeframe='H1').loc[1] for size in (1, 20)]
    widths = [row.mean_diff_ci_high - row.mean_diff_ci_low for row in results]
    assert widths[1] > widths[0]
