"""Tests d'export des résultats d'étude."""

from __future__ import annotations

import pandas as pd

from regime_lib.study import Study


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
