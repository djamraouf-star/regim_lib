"""Tests d'export des résultats d'étude."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_lib.study import Study


def test_study_uses_provided_timeframe_ohlcv(monkeypatch):
    index = pd.date_range(
        "2026-01-01", periods=4, freq="5min", tz="UTC", name="timestamp"
    )
    ohlcv = pd.DataFrame(
        {**{c: [1.0, 2.0, 4.0, 8.0] for c in ("open", "high", "low", "close")},
         "volume": [1.0] * 4},
        index=index,
    )
    study = Study(
        regimes_path="unused.parquet",
        ohlcv_path="unused.parquet",
        ohlcv_data=ohlcv,
        targets=["ret_fwd_1"],
    )

    def fail_load(_):
        raise AssertionError("Study should use the supplied timeframe bars")

    monkeypatch.setattr("regime_lib.study.study.load_parquet", fail_load)
    study._charger_cibles()

    np.testing.assert_allclose(
        study.df_cibles["ret_fwd_1"].iloc[:3],
        [np.log(2.0), np.log(2.0), np.log(2.0)],
    )
    assert study.df_cibles["ret_fwd_1"].iloc[-1:].isna().all()


def test_save_serializes_mixed_modalities_as_labels(tmp_path):
    study = Study(
        regimes_path="unused.parquet",
        ohlcv_path="unused.parquet",
    )
    study.resultats = pd.DataFrame([{
        "fold": 0,
        "feature": "adx__ctx_hour_ny",
        "cible": "ret_fwd_1",
        "n_test": 4,
        "ic_test": 0.1,
        "pvalue_test": 0.5,
        "hit_test": float("nan"),
    }])
    tests = pd.DataFrame({
        "n_a": [2, 2],
        "n_b": [2, 2],
        "mean_a": [0.1, 0.2],
        "mean_b": [0.0, 0.0],
        "p_a_pos": [0.5, 0.5],
        "p_b_pos": [0.5, 0.5],
        "welch_p": [0.5, 0.5],
        "mw_p": [0.5, 0.5],
        "z_p": [0.5, 0.5],
    }, index=pd.Index([8, "ASIE"], name="modalite"))
    study.resultats_tests = {
        ("adx__ctx_session", "ret_fwd_1"): tests,
    }

    study.save(tmp_path)

    saved = pd.read_parquet(tmp_path / "study_tests_modalite.parquet")
    assert set(saved["modalite"]) == {"8", "ASIE"}
