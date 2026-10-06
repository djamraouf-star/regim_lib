"""Exports reproductibles et absence de reliquats entre deux exécutions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from regime_lib.study import Study, InferenceConfig
from regime_lib.study.provenance import file_hash
from regime_lib.study.report import generer_rapport


@pytest.fixture
def study(tmp_path):
    index = pd.date_range('2026-01-01', periods=160, freq='h', tz='UTC', name='timestamp')
    close = 100 + np.arange(160) / 100 + np.sin(np.arange(160))
    prices = pd.DataFrame({'open':close, 'high':close+1, 'low':close-1,
                           'close':close, 'volume':1.}, index=index)
    prices.attrs.update(asset='TEST', timeframe='H1')
    path = tmp_path / 'regimes.parquet'
    pd.DataFrame({'timestamp':index, 'method':'adx', 'asset':'TEST', 'timeframe':'H1',
                  'regime':np.tile(['A']*4+['B']*4, 20), 'params_hash':'fixture'}).to_parquet(path)
    return Study(path, 'unused', ohlcv_data=prices, targets=['ret_fwd_1'],
                 split='holdout', train_ratio=.5,
                 inference=InferenceConfig(4, n_resamples=199, seed=42))


def test_two_runs_reproduce_results_and_manifest(study, tmp_path):
    study.run().save(tmp_path / 'first')
    before = study.resultats.copy()
    study.run().save(tmp_path / 'second')
    pd.testing.assert_frame_equal(before, study.resultats)
    first = json.loads((tmp_path / 'first/study_metadata.json').read_text())
    second = json.loads((tmp_path / 'second/study_metadata.json').read_text())
    assert first == second
    assert first['analysis_mode'] == 'evaluation_temporelle'
    assert first['configuration']['inference']['seed'] == 42
    assert first['folds'][0]['n_train_before_purge'] == 80
    assert first['coverage']
    for name, digest in first['artifacts'].items():
        assert file_hash(tmp_path / 'first' / name) == digest


def test_reexport_removes_only_stale_managed_files(study, tmp_path):
    output = tmp_path / 'output'
    study.run().save(output)
    (output / 'notes.txt').write_text('Conserver')
    assert (output / 'study_inference.json').exists()
    study.inference = None
    study.tests_modalite = False
    study.run().save(output)
    assert not (output / 'study_inference.json').exists()
    assert not (output / 'study_tests_modalite.parquet').exists()
    assert (output / 'notes.txt').read_text() == 'Conserver'
    meta = json.loads((output / 'study_metadata.json').read_text())
    assert 'study_tests_modalite.parquet' not in meta['artifacts']


def test_failed_preparation_preserves_previous_export(study, tmp_path, monkeypatch):
    output = tmp_path / 'output'
    study.run().save(output)
    before = {p.name:p.read_bytes() for p in output.iterdir()}
    def fail(*args, **kwargs):
        raise RuntimeError('Échec de rendu')
    monkeypatch.setattr('regime_lib.study.report.generer_rapport', fail)
    with pytest.raises(RuntimeError, match='rendu'):
        study.save(output)
    assert before == {p.name:p.read_bytes() for p in output.iterdir()}


def test_failed_run_cannot_export_stale_results(study, tmp_path):
    study.run()
    study.ohlcv_data.attrs['asset'] = 'OTHER'
    with pytest.raises(ValueError, match='Actif'):
        study.run()
    with pytest.raises(RuntimeError, match='run'):
        study.save(tmp_path / 'output')


def test_changed_configuration_requires_new_run(study, tmp_path):
    study.run()
    study.split_kwargs['train_ratio'] = .7
    with pytest.raises(ValueError, match='changé'):
        study.save(tmp_path / 'output')


def test_empty_report_and_no_valid_observations(study, tmp_path):
    assert 'Aucun résultat' in generer_rapport(pd.DataFrame())
    regimes = pd.read_parquet(study.regimes_path).assign(is_valid=False)
    regimes.to_parquet(study.regimes_path)
    study.run().save(tmp_path / 'empty')
    meta = json.loads((tmp_path / 'empty/study_metadata.json').read_text())
    assert meta['coverage'][0]['n_commun'] == 0
    assert 'support valide commun vide' in (tmp_path / 'empty/study_report.md').read_text()
