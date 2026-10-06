"""Contre-exemples et références des corrections des constats partiels."""
import json
import numpy as np
import pandas as pd
import pytest
from regime_lib.core.temporal import session_grid
from regime_lib.core.data_loader import resample
from regime_lib.core.output import to_uniform
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.study.features import extraire_features
from regime_lib.study.evaluation import _convention_succes
from regime_lib.methods.volume.volume_profile import VolumeProfileDetector
from regime_lib.methods.volume.vpin import VpinDetector
from regime_lib.core.utils import build_alternating_pivots


def prices(n=40):
    index = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC', name='timestamp')
    close = 10 + np.arange(n) / 100
    frame = pd.DataFrame(dict(open=close, high=close+1, low=close-1, close=close,
                              volume=10., bid_volume=3., ask_volume=7.), index=index)
    frame.attrs.update(timeframe='H1', asset='TEST')
    return frame


def test_local_sessions_follow_dst_and_holidays():
    grid = session_grid('2026-03-06', '2026-03-10', timezone='America/New_York',
                        opens='09:00', closes='12:00', timeframe='H1', holidays=['2026-03-10'])
    assert grid.tolist() == list(pd.to_datetime([
        '2026-03-06 15:00Z','2026-03-06 16:00Z','2026-03-06 17:00Z',
        '2026-03-09 14:00Z','2026-03-09 15:00Z','2026-03-09 16:00Z']))
    frame = prices(len(grid)); frame.index = grid
    out = resample(frame, 'H4', expected_index=grid)
    assert out.coverage.eq(1).all() and not out.is_partial.any()


def test_overnight_dst_counts_elapsed_hours():
    grid = session_grid('2026-10-24', '2026-10-24', timezone='Europe/Paris',
                        opens='22:00', closes='04:00', timeframe='H1', weekdays=(5,))
    assert len(grid) == 7 and not grid.has_duplicates
    with pytest.raises(Exception, match='2026-03-29 02:00'):
        session_grid('2026-03-29', '2026-03-29', timezone='Europe/Paris',
                     opens='02:00', closes='04:00', timeframe='H1', weekdays=(6,))


def test_semantics_export_and_causal_filter():
    frames=[]
    for name in ('adx','price_action'):
        detector=METHOD_REGISTRY[name](timeframe='H1',allow_lookahead=True)
        output=detector.fit_predict(prices())
        frames.append(to_uniform(output, method=name, timeframe='H1', asset='TEST',
                                  regime_map=detector.REGIME_MAP, params_hash='test'))
    data=pd.concat(frames)
    assert data.loc[data.method.eq('price_action'),'available_at'].isna().all()
    assert set(extraire_features(data).columns)=={'adx__regime'}
    data.loc[data.method.eq('adx'),'available_at'] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match='indisponible'):
        extraire_features(data)


def test_every_detector_declares_nominal_semantics():
    for cls in METHOD_REGISTRY.values():
        contract=cls.interpretation()
        assert contract['regime_dimension'] != 'unspecified'
        assert contract['availability'] in ('bar_close','retrospective')
        assert contract['regime_scale']=='nominal'


def test_less_negative_excursion_is_success():
    assert _convention_succes(pd.Series([-.1,-.05,-.01]),'dd_fwd_5').tolist()==[False,False,True]


def test_volume_profile_numeric_reference_and_future_invariance():
    detector=VolumeProfileDetector(timeframe='H1',fenetre=2,n_bins=4,pct_bas=.2,pct_haut=.8)
    # Deux barres uniformes [8,12] : quatre bins équipondérés.
    assert detector._value_area(np.array([12.,12.]),np.array([8.,8.]),np.array([10.,10.]))==(8.,12.)
    frame=prices()
    prefix=detector.fit_predict(frame.iloc[:20])
    full=detector.fit_predict(frame)
    pd.testing.assert_frame_equal(prefix,full.iloc[:20])
    altered=frame.copy(); altered.loc[altered.index[20:],['open','high','low','close']] *= 2
    pd.testing.assert_frame_equal(prefix,detector.fit_predict(altered).iloc[:20])


@pytest.mark.parametrize('cutoff',[9,10,19,20,25,29,30,39])
def test_vpin_past_is_invariant_to_extension_and_future_perturbation(cutoff):
    detector=VpinDetector(timeframe='H1',bucket_volume=100.,n_buckets=2)
    frame=prices()
    frame['ask_volume']=np.where(np.arange(len(frame))%3==0,8.,6.)
    frame['bid_volume']=10-frame.ask_volume
    before=detector.fit_predict(frame.iloc[:cutoff])
    pd.testing.assert_frame_equal(before,detector.fit_predict(frame).iloc[:cutoff])
    changed=frame.copy()
    changed.loc[changed.index[cutoff:],'ask_volume']=30.
    changed.loc[changed.index[cutoff:],'bid_volume']=1.
    pd.testing.assert_frame_equal(before,detector.fit_predict(changed).iloc[:cutoff])


def test_vpin_value_uses_only_preceding_buckets():
    detector=VpinDetector(timeframe='H1',bucket_volume=100.,n_buckets=2)
    frame=prices()
    values=detector.fit_predict(frame).vpin_value
    assert values.iloc[:19].isna().all()
    np.testing.assert_allclose(values.iloc[19:],.4)


def test_pivot_replacement_counterexample():
    # Deux sommets sans creux intermédiaire : le plus récent remplace l'ancien.
    high=np.array([1.,3.,2.,4.,1.]); low=high-.5
    peaks=np.array([False,True,False,True,False]); troughs=np.zeros(5,dtype=bool)
    assert build_alternating_pivots(peaks[:3],troughs[:3],high[:3],low[:3])==[(1,'S',3.)]
    assert build_alternating_pivots(peaks,troughs,high,low)==[(3,'S',4.)]
    assert METHOD_REGISTRY['price_action'].availability=='retrospective'
    assert METHOD_REGISTRY['hmm_gaussian'].availability=='retrospective'


def test_hmm_global_normalisation_changes_past_on_extension():
    from regime_lib.methods.statistical.hmm_gaussian import _normaliser
    prefix=pd.DataFrame({'x':[1.,2.,3.]})
    full=pd.DataFrame({'x':[1.,2.,3.,100.]})
    assert not np.allclose(_normaliser(prefix).x, _normaliser(full).x.iloc[:3])


def test_projection_numeric_reference():
    from regime_lib.projection.features import caracteriser_segments
    frame=prices(3)
    frame[['open','close']]=np.array([[10.,10.],[11.,11.],[12.,12.]])
    frame['high']=[11.,12.,13.]; frame['low']=[9.,10.,11.]
    result=caracteriser_segments(frame,pd.Series([0,0,0],index=frame.index),timeframe='H1').iloc[0]
    assert result.n_barres==3 and result.direction==1
    assert result.pente==2 and result.amplitude==4 and result.efficience==.5
    assert result.pente_atr==1


def test_sensitivity_preserves_all_sizes_and_global_family():
    from regime_lib.study.inference import InferenceConfig, block_sensitivity
    rng=np.random.default_rng(8)
    index=pd.date_range('2026-01-01',periods=400,freq='h',tz='UTC')
    feature=pd.Series(np.tile(np.repeat([0,1],10),20),index=index)
    target=pd.Series(rng.normal(size=400)+feature.to_numpy(),index=index)
    result=block_sensitivity(feature,target,block_sizes=[5,10],horizon=5,
                             config=InferenceConfig(5,n_resamples=199),timeframe='H1')
    assert set(result.index.get_level_values(0))=={5,10}
    assert result.family_size.eq(4).all()
    assert result.inference_status.eq('ok').all()


def test_detection_manifest_and_study_support_hash(tmp_path):
    from regime_lib import cli
    from regime_lib.study import Study
    from regime_lib.utils.provenance import frame_info, file_hash
    frame=prices(120)
    source=tmp_path/'prices.parquet'; frame.to_parquet(source)
    destination=tmp_path/'run'
    assert cli.main(['--url',str(source),'--output',str(destination),'--asset','TEST',
                     '--source-timeframe','H1','--timeframe','H1','--methods','adx'])==0
    meta=json.loads((destination/'run_meta.json').read_text())
    assert meta['protocol_version']=='1.0.0'
    assert meta['versions']['ruptures']!='unavailable'
    assert meta['resources_sha256'] and meta['code_sha256']
    assert meta['artifacts']['regimes.parquet']==file_hash(destination/'regimes.parquet')
    study=Study(destination/'regimes.parquet',source,targets=['ret_fwd_1']).run()
    assert study.metadata['common_support']==frame_info(study.support_commun)
    study.support_commun.iloc[0,0]=not study.support_commun.iloc[0,0]
    with pytest.raises(ValueError,match='modifi|run'):
        study.save(tmp_path/'study')


def test_legacy_vpin_is_not_certified_by_new_registry():
    detector=VpinDetector(timeframe='H1',bucket_volume=100.,n_buckets=2)
    data=to_uniform(detector.fit_predict(prices()),method='vpin',timeframe='H1',
                    asset='TEST',regime_map=detector.REGIME_MAP,params_hash='new')
    assert 'vpin__regime' in extraire_features(data)
    assert data.available_at.notna().all()
    with pytest.raises(ValueError,match='Aucune méthode'):
        extraire_features(data.drop(columns=['availability','available_at']))
