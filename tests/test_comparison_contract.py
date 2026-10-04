"""Contre-exemples d'identité, de support et de temporalité (étape 2)."""
import numpy as np
import pandas as pd
import pytest

from regime_lib.core.data_loader import resample, load_parquet, charger_tick, agreger_tick
from regime_lib.core.temporal import unaligned_mask
from regime_lib.core.output import to_uniform
from regime_lib.methods.trend.adx import ADXDetector
from regime_lib.projection import (caracteriser_segments, projeter_barres,
                                  purete_par_segment, comparer_methodes,
                                  croiser_methodes_segments)
from regime_lib.study import Study
from regime_lib.study.features import extraire_features
from regime_lib.study.targets import calculer_cible
from regime_lib.utils.validation import validate_ohlcv


def bars(n=12, freq="1h"):
    index = pd.date_range("2026-01-05", periods=n, freq=freq, tz="UTC", name="timestamp")
    prices = np.arange(n, dtype=float) + 100
    return pd.DataFrame({**{c: prices.copy() for c in ("open", "high", "low", "close")},
                         "volume": np.ones(n)}, index=index)


def regimes(index):
    return pd.DataFrame({"timestamp": index, "regime": "A", "method": "adx",
                         "asset": "INDEX", "timeframe": "H1", "params_hash": "one"})


@pytest.mark.parametrize("field,value", [("asset", "OTHER"), ("timeframe", "M5"),
                                         ("params_hash", "two")])
def test_ambiguous_series_and_configuration_rejected(field, value):
    first = regimes(bars().index)
    second = first.copy()
    second[field] = value
    with pytest.raises(ValueError, match="ambigu"):
        extraire_features(pd.concat([first, second]))


def test_select_series_and_configuration():
    first = regimes(bars().index)
    second = first.assign(asset="OTHER", timeframe="M5", params_hash="two")
    out = extraire_features(pd.concat([first, second]), asset="INDEX", timeframe="H1",
                            configurations={"adx": "one"})
    assert len(out) == 12
    assert out.attrs["series_identity"]["asset"] == "INDEX"


def test_duplicate_timestamp_rejected():
    data = regimes(bars().index)
    with pytest.raises(ValueError, match="doublons"):
        extraire_features(pd.concat([data, data.iloc[:1]]))


def test_invalid_representations_mask_all_features():
    data = regimes(bars(5).index).assign(confidence=1., is_partial=False)
    data.loc[0, "regime"] = "INCONNU"
    data.loc[1, "is_partial"] = True
    data.loc[2, "confidence"] = np.inf
    out = extraire_features(data, features=["regime", "confidence"])
    assert out.iloc[:2].isna().all().all()
    assert pd.isna(out.iloc[2]["adx__confidence"])


@pytest.mark.parametrize("col,value", [("close", np.inf), ("low", -1.),
                                      ("high", 1.), ("volume", -1.)])
def test_ohlcv_contract_used_by_direct_detectors(col, value):
    data = bars()
    data.loc[data.index[3], col] = value
    with pytest.raises(ValueError):
        ADXDetector(timeframe="H1").fit_predict(data)


def test_volume_unavailable_allowed_for_price_method():
    data = bars().assign(volume=np.nan)
    validate_ohlcv(data)
    ADXDetector(timeframe="H1").fit_predict(data)


def test_partiality_survives_resample_detector_and_export():
    data = bars(48).assign(is_partial=False)
    data.loc[data.index[5], "is_partial"] = True
    higher = resample(data, "H4")
    assert higher.loc[data.index[8], "is_partial"]
    assert higher.loc[data.index[0], "coverage"] == .25
    out = ADXDetector(timeframe="H4").fit_predict(higher)
    uniform = to_uniform(out, method="adx", timeframe="H4", asset="INDEX",
                         regime_map=ADXDetector.REGIME_MAP)
    assert uniform["is_partial"].tolist() == higher["is_partial"].tolist()


def test_complete_last_bar_is_not_artificially_partial():
    data = bars(5)
    higher = resample(data, "H4")
    assert not higher["is_partial"].iloc[-1]


def test_missing_source_bar_marks_aggregate_partial():
    data = bars(9).drop(bars(9).index[2])
    data.attrs["timeframe"] = "H1"
    higher = resample(data, "H4")
    assert higher["is_partial"].iloc[1]
    assert higher["coverage"].iloc[1] == .75


def test_upsampling_and_ambiguous_frequency_rejected():
    with pytest.raises(ValueError, match="Conversion impossible"):
        resample(bars(), "M5")
    with pytest.raises(ValueError, match="ambigu"):
        resample(bars().drop(bars().index[2]), "H4")


def test_weekly_grid_matches_aggregation():
    data = bars(24 * 15)
    weekly = resample(data, "W1")
    assert (weekly.index.dayofweek == 0).all()
    assert not unaligned_mask(weekly.index, "W1").any()
    ticks = pd.DataFrame({"bidPrice": data.close, "askPrice": data.close + 1,
                          "bidVolume": 1., "askVolume": 1.})
    direct = agreger_tick(ticks, "W1")
    via_hour = resample(agreger_tick(ticks, "H1"), "W1")
    pd.testing.assert_frame_equal(direct[["open", "close"]], via_hour[["open", "close"]])


@pytest.mark.parametrize("alias", ["timestamp", "datetime", "date", "time"])
def test_tick_aliases_same_in_batch_and_full(tmp_path, alias):
    path = tmp_path / "INDEX_ticks.parquet"
    frame = pd.DataFrame({alias: bars(5).index, "bidPrice": 100., "askPrice": 101.,
                          "bidVolume": 1., "askVolume": 1.})
    frame.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(charger_tick(str(path)), charger_tick(str(path), batch_size=2))


def test_open_timestamp_conversion_is_explicit(tmp_path):
    data = bars(5)
    path = tmp_path / "INDEX.parquet"
    data.to_parquet(path)
    loaded, _ = load_parquet(str(path), timeframe="H1", timestamp_convention="open")
    assert loaded.index.equals(data.index + pd.Timedelta(hours=1))
    assert loaded.attrs["original_timestamp_convention"] == "open"


@pytest.mark.parametrize("family", ["ret", "ru", "dd", "vol"])
def test_targets_exclude_entire_invalid_trajectory(family):
    data = bars(10).assign(is_partial=False)
    data.loc[data.index[3], "is_partial"] = True
    target = calculer_cible(data, f"{family}_fwd_2")
    assert target.iloc[1:4].isna().all()
    assert target.iloc[4:8].notna().all()
    assert target.iloc[-2:].isna().all()


def test_gap_is_not_a_shorter_horizon_and_calendar_closure_is_allowed():
    data = bars(8).drop(bars(8).index[3])
    target = calculer_cible(data, "ret_fwd_2", timeframe="H1")
    assert pd.isna(target.iloc[1])
    assert pd.isna(target.iloc[2])
    scheduled = calculer_cible(data, "ret_fwd_2", timeframe="H1", expected_index=data.index)
    assert scheduled.iloc[1] == pytest.approx(np.log(104 / 101))


def test_projection_rejects_same_size_shifted_index():
    data = bars()
    shifted = pd.Series(0, index=data.index + pd.Timedelta(hours=1))
    with pytest.raises(ValueError, match="Alignement"):
        caracteriser_segments(data, shifted)
    with pytest.raises(ValueError, match="Alignement"):
        projeter_barres(data, shifted, pd.DataFrame())
    with pytest.raises(ValueError, match="Alignement"):
        purete_par_segment(pd.Series("A", index=data.index), shifted)


def test_projection_missing_empty_and_invalid_segments():
    data = bars().assign(is_partial=False)
    data.iloc[2, data.columns.get_loc("is_partial")] = True
    ids = pd.Series(0, index=data.index)
    features = caracteriser_segments(data, ids)
    assert not features.loc[0, "is_valid"]
    assert pd.isna(features.loc[0, "amplitude"])
    assert not projeter_barres(data, ids, features).is_valid.any()
    assert caracteriser_segments(data.iloc[:0], ids.iloc[:0]).empty
    assert purete_par_segment(pd.Series("INCONNU", index=data.index), ids).empty
    with pytest.raises(ValueError, match="absentes"):
        projeter_barres(data, ids, features.iloc[:0])


def test_common_projection_features_must_match():
    data = bars()
    ids = pd.Series(0, index=data.index)
    ref = projeter_barres(data, ids, caracteriser_segments(data, ids)).assign(regime="A")
    other = ref.copy()
    other["amplitude"] += 1
    with pytest.raises(ValueError, match="commune différente"):
        croiser_methodes_segments({"a": ref, "b": other})


def test_agreement_uses_same_valid_support():
    index = bars(4).index
    result = comparer_methodes({"a": pd.Series(["INCONNU", "A", "A", "B"], index=index),
                               "b": pd.Series(["A", "A", "A", "INCONNU"], index=index)},
                              pd.Series(0, index=index))
    assert result.n_propre.tolist() == [3, 3]
    assert result.n_commun.tolist() == [2, 2]


def test_study_resamples_targets_to_regime_timeframe(tmp_path):
    data = bars(49, "5min")
    data.attrs["asset"] = "INDEX"
    hourly = pd.date_range(data.index[0], data.index[-1], freq="1h")
    path = tmp_path / "regimes.parquet"
    regimes(hourly).to_parquet(path)
    study = Study(path, "unused", ohlcv_data=data, targets=["ret_fwd_1"], tests_modalite=False).run()
    assert study.df_cibles.index.equals(hourly)
    assert study.df_cibles.iloc[1, 0] == pytest.approx(np.log(124 / 112))


def test_study_rejects_wrong_asset(tmp_path):
    data = bars()
    path = tmp_path / "regimes.parquet"
    regimes(data.index).to_parquet(path)
    data.attrs["asset"] = "OTHER"
    with pytest.raises(ValueError, match="Actif OHLCV"):
        Study(path, "unused", ohlcv_data=data).run()


def test_study_exports_own_and_common_support(tmp_path):
    data = bars(16)
    data.attrs["asset"] = "INDEX"
    a = regimes(data.index)
    b = a.assign(method="shannon")
    a.loc[0:2, "regime"] = "INCONNU"
    b.loc[0:4, "regime"] = "INCONNU"
    path = tmp_path / "regimes.parquet"
    pd.concat([a, b]).to_parquet(path)
    study = Study(path, "unused", ohlcv_data=data, targets=["ret_fwd_1"], tests_modalite=False).run()
    assert study.couverture.n_propre.tolist() == [12, 10]
    assert study.couverture.n_commun.tolist() == [10, 10]
    assert study.resultats.n_test.tolist() == [10, 10]
    study.save(tmp_path / "report")
    assert (tmp_path / "report/study_common_support.parquet").is_file()
    assert (tmp_path / "report/study_coverage.parquet").is_file()


def test_empty_common_support_has_no_estimate_and_is_reported(tmp_path):
    data = bars(12)
    data.attrs["asset"] = "INDEX"
    path = tmp_path / "regimes.parquet"
    regimes(data.index).assign(regime="INCONNU").to_parquet(path)
    study = Study(path, "unused", ohlcv_data=data, tests_modalite=False).run()
    assert (study.resultats.n_test == 0).all()
    assert study.resultats.ic_test.isna().all()
    study.save(tmp_path / "report")
    assert "support valide commun vide" in (tmp_path / "report/study_report.md").read_text()


def test_projection_gap_and_disjoint_ids_are_not_silently_joined():
    data = bars(8).drop(bars(8).index[3])
    data.attrs["timeframe"] = "H1"
    ids = pd.Series(0, index=data.index)
    assert not caracteriser_segments(data, ids).loc[0, "is_valid"]
    assert caracteriser_segments(data, ids, expected_index=data.index).loc[0, "is_valid"]
    ids.iloc[2] = 1
    with pytest.raises(ValueError, match="disjoints"):
        caracteriser_segments(data, ids)


def test_session_calendar_controls_aggregate_coverage():
    data = bars(5).iloc[[1, 3, 4]]
    result = resample(data, "H4", source_timeframe="H1", expected_index=data.index)
    assert result.coverage.iloc[0] == 1.
    assert not result.is_partial.iloc[0]
    assert result.attrs["calendar"] == "explicit_schedule"


def test_crossed_projections_allow_different_warmup():
    data = bars(5)
    ids = pd.Series(0, index=data.index)
    first = projeter_barres(data, ids, caracteriser_segments(data, ids)).assign(regime="A")
    second = first.assign(is_warmup=[True, True, False, False, False])
    out = croiser_methodes_segments({"a": first, "b": second})
    assert out.is_valid.tolist() == [False, False, True, True, True]
    assert out.b_regime.iloc[:2].isna().all()
