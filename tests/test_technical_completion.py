"""Références numériques et contrats causaux des compléments d'audit."""
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from regime_lib.methods.trend.adx import _compute_adx
from regime_lib.methods.price.price_action_causal import PriceActionCausalDetector
from regime_lib.methods.statistical.hmm_causal import HMMCausalDetector, filter_forward
from regime_lib.study.targets import calculer_cible
from regime_lib.study.calibration import ConfidenceCalibrator
from regime_lib.projection.agreement import comparer_methodes
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.core.output import to_uniform


def bars(n=180):
    rng = np.random.default_rng(1729)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    return pd.DataFrame({"open": close, "high": close + rng.uniform(.1, 2, n),
                         "low": close - rng.uniform(.1, 2, n), "close": close, "volume": 1.},
                        index=pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC", name="timestamp"))


def test_adx_fraction_reference():
    # Calcul rationnel indépendant : n=2, mouvements (+2,0),(0,1),(3,0),(0,2).
    # Amorçage de la somme sur n-1 transitions, puis Wilder, convention TA-Lib.
    high = np.array([10.,12.,11.,14.,12.,15.])
    low = np.array([8.,9.,8.,10.,8.,11.])
    close = np.array([9.,11.,9.,13.,9.,14.])
    plus, minus, adx = _compute_adx(high,low,close,2)
    # t2 : +DM=1, -DM=1, TR=4.5 ; DX=0.
    # t3 : +DM=3.5, -DM=.5 ; DX=75 ; ADX=37.5.
    # t4 : +DM=1.75, -DM=2.25 ; DX=12.5 ; ADX=25.
    np.testing.assert_allclose(plus[2],float(F(200,9)))
    np.testing.assert_allclose(minus[2],float(F(200,9)))
    np.testing.assert_allclose(adx[3:5],[37.5,25.])
    assert np.isnan(adx[:3]).all()
    flat = np.ones(8)
    np.testing.assert_allclose(_compute_adx(flat,flat,flat,2)[2][3:],0.)


@pytest.mark.parametrize("family, expected", [
    ("mfe_long", .2), ("mae_long", -.1), ("mfe_short", .1), ("mae_short", -.2),
    ("time_mfe_long", 2), ("time_mae_long", 1), ("time_mfe_short", 1), ("time_mae_short", 2),
])
def test_excursions_reference(family,expected):
    df = bars(5)
    df["close"] = [100,102,103,104,105]
    # L'extrême énorme de la barre d'entrée doit être exclu.
    df["high"] = [1000,110,120,120,120]
    df["low"] = [1,90,95,95,95]
    result = calculer_cible(df,f"{family}_fwd_2",timeframe="H1")
    assert result.iloc[0] == pytest.approx(expected)
    assert result.iloc[-2:].isna().all()
    df["is_partial"] = [False,True,False,False,False]
    assert np.isnan(calculer_cible(df,f"{family}_fwd_2",timeframe="H1").iloc[0])
    gap = df.drop(df.index[1]).drop(columns="is_partial")
    assert np.isnan(calculer_cible(gap,f"{family}_fwd_2",timeframe="H1").iloc[0])


def test_excursion_ties_and_zero():
    df=bars(4)
    df["close"],df["high"],df["low"] = 100,100,100
    assert calculer_cible(df,"time_mfe_long_fwd_2").iloc[0]==0
    df["high"] = [100,110,110,110]
    assert calculer_cible(df,"time_mfe_long_fwd_2").iloc[0]==1
    for old,new in [("dd","close_min"),("ru","close_max")]:
        np.testing.assert_allclose(calculer_cible(df,f"{old}_fwd_2"),calculer_cible(df,f"{new}_fwd_2"),equal_nan=True)


def test_partition_baselines():
    index=bars(12).index
    seg=pd.Series(np.repeat([0,1,2],4),index=index)
    result=comparer_methodes({"constant":pd.Series("A",index=index),
        "matched":pd.Series(np.repeat(["A","B","C"],4),index=index),
        "singleton":pd.Series([str(i) for i in range(12)],index=index)},seg)
    assert result.loc["constant","purete_ponderee"]==1
    assert result.loc["constant","ami"]==0
    assert result.loc["constant","partition_degeneree"]
    assert result.loc["matched","ami"]==pytest.approx(1)
    assert result.loc["matched","ari"]==pytest.approx(1)
    assert result.loc["singleton","ami"]==pytest.approx(0,abs=1e-12)
    assert result.loc["matched","gain_purete"]==pytest.approx(2/3)


def test_causal_pivots_prefixes_and_event_dates():
    df=bars()
    detector=PriceActionCausalDetector(n_fractale=2,timeframe="H1")
    full=detector.fit_predict(df)
    assert full.regime.ne("INCONNU").any()
    for length in [0,1,4,20,55,99,179]:
        actual=detector.fit_predict(df.iloc[:length])
        assert_frame_equal(actual,full.iloc[:length])
    events=full[full.pivot_kind.ne("")]
    assert len(events)>5
    assert (events.pivot_confirmed_at-events.pivot_observed_at).eq(pd.Timedelta(hours=2)).all()
    assert events.pivot_confirmed_at.eq(events.index.to_series()).all()
    disturbed=df.copy()
    disturbed.iloc[100:,:4]*=2
    assert_frame_equal(detector.fit_predict(disturbed).iloc[:100],full.iloc[:100])


def test_forward_filter_reference():
    model=SimpleNamespace(means_=np.array([[0.],[1.]]),covars_=np.ones((2,1,1)),
                          startprob_=np.array([.6,.4]),transmat_=np.array([[.8,.2],[.3,.7]]))
    result=filter_forward(model,np.array([[0.],[1.]]))
    first=np.array([.6,.4*np.exp(-.5)]);first/=first.sum()
    second=(first@model.transmat_)*np.array([np.exp(-.5),1]);second/=second.sum()
    np.testing.assert_allclose(result,[first,second])


def test_hmm_train_only_and_prefixes():
    df=bars(130)
    detector=HMMCausalDetector(train_size=65,n_iter=20,seed=3)
    full=detector.fit_predict(df)
    assert full.iloc[:65].regime.eq("INCONNU").all()
    assert full.iloc[65:].regime.ne("INCONNU").all()
    for n in [30,65,66,90,129]:
        prefix=detector.fit_predict(df.iloc[:n])
        assert_frame_equal(prefix,full.iloc[:n])
    mutated=df.copy();mutated.iloc[100:,:4]*=2
    other=detector.fit_predict(mutated)
    assert_frame_equal(full.iloc[:100],other.iloc[:100])
    assert full.attrs["hmm_diagnostics"]==other.attrs["hmm_diagnostics"]
    means=full.attrs["hmm_diagnostics"]["state_means"]
    assert means[0][0]<=means[1][0]


def test_calibration_reference_and_purge():
    index=bars(12).index
    scores=pd.Series([.1,.2,.8,.9],index=index[:4])
    outcomes=pd.Series([0,0,1,1],index=scores.index)
    ends=pd.Series(index[2:6],index=scores.index)
    calibrator=ConfidenceCalibrator("ret_fwd_2 > 0").fit(scores,outcomes,target_end=ends)
    with pytest.raises(ValueError,match="purge"):
        calibrator.predict(pd.Series([.5],index=index[5:6]))
    test_scores=pd.Series([.1,.9,.1,.9],index=index[6:10])
    labels=pd.Series([0,1,0,1],index=test_scores.index)
    report=calibrator.evaluate(test_scores,labels)
    assert report["brier"]==0
    assert report["baseline_brier"]==.25
    assert report["brier_skill"]==1
    assert report["ece"]==0
    # Modifier les issues du test change l'évaluation, jamais le calibrateur.
    before=calibrator.predict(test_scores)
    assert calibrator.evaluate(test_scores,1-labels)["brier"]==1
    pd.testing.assert_series_equal(before,calibrator.predict(test_scores))


def test_interpretation_export(tmp_path):
    for cls in METHOD_REGISTRY.values():
        interpretation=cls.interpretation()
        assert interpretation["confidence_description"]
        assert not interpretation["confidence_calibrated"]
    detector=PriceActionCausalDetector()
    result=detector.fit_predict(bars(50))
    uniform=to_uniform(result,method=detector.name,timeframe="H1",asset="TEST",regime_map=detector.REGIME_MAP)
    path=tmp_path/'regimes.parquet';uniform.to_parquet(path)
    loaded=pd.read_parquet(path)
    assert loaded.confidence_kind.eq("heuristic_margin").all()
    assert loaded.indicator_name.eq(detector.name).all()
    assert loaded.revises_history.eq(False).all()


def test_pivots_publish_only_after_confirmation():
    df=bars(7)
    values=np.array([10.,14.,11.,16.,13.,18.,15.])
    df["open"]=df["close"]=values
    df["high"],df["low"]=values+.1,values-.1
    result=PriceActionCausalDetector(n_fractale=1,k_atr=.01).fit_predict(df)
    assert result.regime.iloc[:5].eq("INCONNU").all()
    assert result.regime.iloc[5]=="TENDANCE_HAUSSIERE"
    assert result.pivot_observed_at.iloc[5]==df.index[4]
    assert result.pivot_confirmed_at.iloc[5]==df.index[5]
    assert result.pivot_kind.tolist()==["","","S","C","S","C","S"]


@pytest.mark.parametrize("method",["quote_size_imbalance","quote_absolute_imbalance_buckets","retrospective_pivot_breakout"])
def test_explicit_indicator_names(method):
    df=bars(60)
    df["bid_volume"],df["ask_volume"]=1.,3.
    params={"allow_lookahead":True}
    if method=="quote_absolute_imbalance_buckets":
        params.update(bucket_volume=4.,n_buckets=2)
    detector=METHOD_REGISTRY[method](**params)
    result=detector.fit_predict(df)
    assert result.regime.isin(detector.REGIME_MAP).all()
    assert result.attrs["indicator_name"]==method
    if method=="quote_size_imbalance":
        np.testing.assert_allclose(result.quote_imbalance.dropna(),.5)
        assert result.regime.eq("TAILLE_ASK_DOMINANTE").any()
    elif method=="quote_absolute_imbalance_buckets":
        np.testing.assert_allclose(result.quote_absolute_imbalance.dropna(),.5)
        assert result.regime.eq("DESEQUILIBRE_ELEVE").any()


def test_cli_causal_variants_and_study_targets(tmp_path):
    import json
    from regime_lib.cli import main
    from regime_lib.study import Study
    from regime_lib.study.features import extraire_features
    df=bars(160)
    df.attrs.update(asset="TEST",timeframe="H1")
    source=tmp_path/'TEST.parquet';df.to_parquet(source)
    output=tmp_path/'output'
    assert main(['--url',str(source),'--asset','TEST','--source-timeframe','H1',
                 '--timestamp-convention','close','--timeframe','H1',
                 '--methods','price_action_causal,hmm_causal','--output',str(output),
                 '--method-params',json.dumps({'hmm_causal':{'train_size':50,'n_iter':20},
                                              'price_action_causal':{'n_fractale':2}})])==0
    exported=pd.read_parquet(output/'regimes.parquet')
    extracted=extraire_features(exported,asset='TEST',timeframe='H1')
    assert set(extracted)=={'price_action_causal__regime','hmm_causal__regime'}
    rows=exported[exported.method.eq('hmm_causal')]
    assert rows.iloc[:50].is_warmup.all()
    assert json.loads(rows.hmm_diagnostics.iloc[50])['train_rows']==30
    df.attrs.update(source=str(source),calendar='continuous_utc_v1')
    study=Study(output/'regimes.parquet','unused',ohlcv_data=df,
                targets=['mfe_long_fwd_2','mae_short_fwd_2','time_mfe_long_fwd_2'],
                features=['confidence'],split='holdout',train_ratio=.7,tests_modalite=False).run()
    assert len(study.resultats)==6
    study.save(tmp_path/'study')
    assert (tmp_path/'study/study_target_ends.parquet').exists()


def test_hmm_rejects_incomplete_train_and_filter():
    df=bars(90)
    df['is_partial']=False
    detector=HMMCausalDetector(train_size=50,n_iter=10)
    df.loc[df.index[10],'is_partial']=True
    with pytest.raises(ValueError,match='train HMM'):
        detector.fit_predict(df)
    df['is_partial']=False
    df.loc[df.index[-1],'is_partial']=True
    with pytest.raises(ValueError,match='filtrage HMM'):
        detector.fit_predict(df)
