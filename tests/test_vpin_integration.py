"""Parcours VPIN public : CLI, export et features causales Study."""
import json

import numpy as np
import pandas as pd
import pytest

from regime_lib.cli import main
from regime_lib.study.features import extraire_features


@pytest.mark.parametrize("source_type", ["bars", "ticks"])
def test_vpin_cli_exports_multiple_timeframes_and_study_features(tmp_path, source_type):
    index = pd.date_range('2026-01-05 00:01', periods=30, freq='min', tz='UTC', name='timestamp')
    bars = pd.DataFrame({
        'open': 10., 'high': 11., 'low': 9., 'close': 10.,
        'volume': 100., 'bid_volume': 30., 'ask_volume': 70.,
    }, index=index)
    source_args = ['--source-timeframe', 'M1', '--timestamp-convention', 'close']
    if source_type == 'ticks':
        bars = pd.DataFrame({'bidPrice': 10., 'askPrice': 10.1,
                             'bidVolume': 30., 'askVolume': 70.}, index=index)
        source_args = []
    source = tmp_path / 'TEST.parquet'
    bars.to_parquet(source)
    output = tmp_path / 'output'
    assert main([
        '--url', str(source), '--asset', 'TEST',
        *source_args,
        '--timeframe', 'M1,M5', '--methods', 'vpin', '--output', str(output),
        '--method-params', json.dumps({'vpin': {'bucket_volume': 100., 'n_buckets': 2}}),
    ]) == 0
    exported = pd.read_parquet(output / 'regimes.parquet')
    assert set(exported.timeframe) == {'M1', 'M5'}
    assert exported.method.eq('vpin').all()
    assert exported.availability.eq('bar_close').all()
    assert not exported.revises_history.any()
    assert exported.available_at.equals(exported.timestamp)
    for timeframe in ('M1', 'M5'):
        selected = exported.loc[exported.timeframe.eq(timeframe)]
        np.testing.assert_allclose(selected.vpin_value.dropna(), .4)
        assert selected.vpin_value.notna().any()
        features = extraire_features(exported, features=['regime', 'vpin_value'],
                                     asset='TEST', timeframe=timeframe)
        assert set(features.columns) == {'vpin__regime', 'vpin__vpin_value'}
        np.testing.assert_allclose(features['vpin__vpin_value'].dropna(), .4)
    metadata_files = list(output.glob('*.json'))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text())
    assert metadata['methods'] == ['vpin']
    assert metadata['method_params']['vpin']['n_buckets'] == 2
    assert not metadata['run_descriptif']
