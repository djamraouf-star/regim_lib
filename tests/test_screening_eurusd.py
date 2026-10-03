"""Tests des fonctions statistiques du script de criblage EURUSD."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyses.EURUSD.scripts.screening_eurusd import (
    _adjust_bh,
    criblage_regimes_par_session,
)


def test_benjamini_hochberg_preserves_order_and_missing_values():
    pvalues = pd.Series([0.01, 0.04, np.nan, 0.03])

    adjusted = _adjust_bh(pvalues)

    assert adjusted.iloc[0] == 0.03
    assert adjusted.iloc[1] == 0.04
    assert np.isnan(adjusted.iloc[2])
    assert adjusted.iloc[3] == 0.04


def test_session_screen_compares_regimes_only_within_test_session():
    index = pd.date_range(
        "2026-01-05", periods=32, freq="5min", tz="UTC", name="timestamp"
    )
    regimes = pd.DataFrame({
        "timestamp": index,
        "method": "adx",
        "regime": ["A", "B"] * 16,
    })
    context = pd.DataFrame({
        "ctx_session": ["ASIE"] * 16 + ["LONDRES"] * 16,
    }, index=index)
    targets = pd.DataFrame({
        "ret_fwd_1": [
            0.001 * (i + 1) if i % 2 == 0 else -0.001 * (i + 1)
            for i in range(len(index))
        ],
    }, index=index)

    results = criblage_regimes_par_session(
        regimes,
        context,
        targets,
        folds=[(index[:0], index)],
        min_samples=2,
    )

    assert len(results) == 4
    assert set(results["session"]) == {"ASIE", "LONDRES"}
    assert set(results["regime"]) == {"A", "B"}
    assert {"welch_q", "mw_q", "z_q"} <= set(results.columns)
    assert results[["welch_q", "mw_q", "z_q"]].max().max() <= 1.0
