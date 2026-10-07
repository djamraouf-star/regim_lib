"""Buckets VPIN fractionnés : références numériques et causalité."""
import numpy as np
import pandas as pd
import pytest

from regime_lib.methods.volume.vpin import VpinDetector


def frame(total, ratios):
    total = np.asarray(total, dtype=float)
    ratios = np.broadcast_to(ratios, total.shape)
    ask = total * (1 + ratios) / 2
    return pd.DataFrame({
        'open': 10., 'high': 11., 'low': 9., 'close': 10., 'volume': total,
        'ask_volume': ask, 'bid_volume': total - ask,
    }, index=pd.date_range('2026-01-01', periods=len(total), freq='h', tz='UTC'))


def detector(bucket=100., window=2):
    return VpinDetector(timeframe='H1', bucket_volume=bucket, n_buckets=window)


def test_large_rows_have_no_empty_buckets():
    result = detector().fit_predict(frame([250.] * 80, .2))
    np.testing.assert_allclose(result.vpin_value, .2)
    assert not result.regime.eq('INCONNU').any()


def test_fractional_carry_and_exact_boundary():
    # Premier bucket : 60 équilibrés + 40 déséquilibrés => .4.
    # Deuxième bucket : 100 déséquilibrés => 1. Moyenne .7.
    # Troisième bucket : reliquat 10 déséquilibrés + 90 équilibrés => .1.
    result = detector().fit_predict(frame([60., 150., 90., 0.], [0., 1., 0., 0.]))
    np.testing.assert_allclose(result.vpin_value, [np.nan, .7, .55, .55], equal_nan=True)


def test_incomplete_bucket_does_not_change_published_score():
    result = detector().fit_predict(frame([200., 99., 1.], [0., 1., 1.]))
    np.testing.assert_allclose(result.vpin_value, [0., 0., .5])


def test_zero_volume_keeps_warmup_or_last_score():
    result = detector().fit_predict(frame([0., 100., 0., 100., 0.], .4))
    np.testing.assert_allclose(result.vpin_value, [np.nan, np.nan, np.nan, .4, .4], equal_nan=True)


def test_decimal_boundaries_close_without_extra_observation():
    result = detector(bucket=.1).fit_predict(frame([.03, .07, .1, .1], .4))
    np.testing.assert_allclose(result.vpin_value, [np.nan, np.nan, .4, .4], equal_nan=True)


def test_huge_volume_uses_bounded_window():
    result = detector(bucket=1.).fit_predict(frame([1.e12], .6))
    np.testing.assert_allclose(result.vpin_value, [.6])


@pytest.mark.parametrize('seed', range(5))
def test_against_independent_unit_volume_reference_and_future(seed):
    rng = np.random.default_rng(seed)
    totals = rng.integers(0, 45, 60)
    ratios = rng.choice([0., .25, .5, 1.], len(totals))
    data = frame(totals, ratios)
    det = detector(bucket=10., window=3)
    result = det.fit_predict(data)
    # Référence indépendante : développer chaque ligne en unités de volume,
    # puis regrouper dix unités par bucket, sans état partiel ni optimisation.
    expanded = np.repeat(ratios, totals)
    count = len(expanded) // 10
    buckets = expanded[:count * 10].reshape(-1, 10).mean(axis=1)
    expected = []
    for end in np.cumsum(totals):
        closed = end // 10
        expected.append(buckets[closed-3:closed].mean() if closed >= 3 else np.nan)
    np.testing.assert_allclose(result.vpin_value, expected, equal_nan=True, atol=1.e-14)
    for cut in [1, 2, 7, 19, 31, 59]:
        prefix = det.fit_predict(data.iloc[:cut])
        pd.testing.assert_frame_equal(prefix, result.iloc[:cut])
        changed = data.copy()
        changed.loc[changed.index[cut:], 'ask_volume'] = 300.
        changed.loc[changed.index[cut:], 'bid_volume'] = 1.
        pd.testing.assert_frame_equal(prefix, det.fit_predict(changed).iloc[:cut])


def test_partial_rows_remain_masked():
    data = frame([250.] * 3, .4)
    data['is_partial'] = [False, True, False]
    result = detector().fit_predict(data)
    assert result.regime.iloc[1] == 'INCONNU'
    assert np.isnan(result.vpin_value.iloc[1])
    np.testing.assert_allclose(result.vpin_value.iloc[[0, 2]], .4)
