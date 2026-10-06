"""Séparation sur les dates réelles de fin des cibles."""
import numpy as np
import pandas as pd
import pytest

from regime_lib.study import Study
from regime_lib.study.split import split_walk_forward


def make_study(tmp_path, *, horizon=5, embargo=0, scheduled=False):
    index = pd.date_range("2026-01-01", periods=41, freq="h", tz="UTC", name="timestamp")
    if scheduled:
        index = index.delete(19)
    else:
        index = index[:40]
    close = np.arange(40, dtype=float) + 100
    prices = pd.DataFrame({c: close for c in ("open", "high", "low", "close")}, index=index)
    prices["volume"] = 1.
    prices.attrs.update(asset="INDEX", timeframe="H1")
    path = tmp_path / "regimes.parquet"
    pd.DataFrame({"timestamp": index, "asset": "INDEX", "timeframe": "H1",
                  "method": "adx", "regime": "A", "confidence": np.linspace(.1, .9, 40)}).to_parquet(path)
    return Study(path, "unused", ohlcv_data=prices, targets=[f"ret_fwd_{horizon}"],
                 features=["confidence"], split="holdout", train_ratio=.5, embargo=embargo, tests_modalite=False,
                 expected_index=index if scheduled else None)


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("embargo,removed", [(0, 5), (3, 2), (5, 0)])
def test_purge_uses_target_end_including_boundary(tmp_path, scheduled, embargo, removed):
    study = make_study(tmp_path, scheduled=scheduled, embargo=embargo).run()
    row = study.resultats.iloc[0]
    assert row.n_purge_train == removed
    assert row.n_train == 20 - removed
    ends = study.df_fin_cibles.iloc[:20, 0]
    test_start = study.df_features.index[20 + embargo]
    kept = ends[ends < test_start]
    assert len(kept) == row.n_train
    assert kept.max() < test_start
    study.save(tmp_path / "export")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "export/study_target_ends.parquet"),
                                  study.df_fin_cibles, check_freq=False)


def test_purge_rejects_empty_train(tmp_path):
    with pytest.raises(ValueError, match="Train vide après purge"):
        make_study(tmp_path, horizon=25).run()


def test_test_prices_cannot_change_training_metrics(tmp_path):
    study = make_study(tmp_path).run()
    before = study.resultats.iloc[0].copy()
    study.ohlcv_data.loc[study.ohlcv_data.index[20]:, ["open", "high", "low", "close"]] *= 2
    study.run()
    after = study.resultats.iloc[0]
    for name in ("n_train", "ic_train", "hit_train"):
        assert (pd.isna(before[name]) and pd.isna(after[name])) or before[name] == after[name]
    # Vérifier aussi directement les cibles retenues, indépendamment des métriques nominales.
    assert study.df_cibles.iloc[14, 0] == pytest.approx(np.log(119 / 114))


def test_walk_forward_has_nonoverlapping_tests_and_expanding_train():
    index = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    folds = split_walk_forward(index, n_folds=3, train_ratio=.6, anchored=True)
    assert len(folds) == 3
    for i, (train, test) in enumerate(folds):
        assert train[0] == index[0]
        assert train[-1] < test[0]
        if i:
            assert folds[i-1][1][-1] < test[0]
            assert len(train) > len(folds[i-1][0])


@pytest.mark.parametrize("anchored", [False, True])
def test_each_walk_forward_fold_purges_each_target_horizon(tmp_path, anchored):
    study = make_study(tmp_path)
    study.noms_cibles = ["ret_fwd_1", "ret_fwd_5"]
    study.schema_split = "walk_forward"
    study.split_kwargs = {"n_folds": 3, "train_ratio": .6, "anchored": anchored}
    study.run()
    folds = split_walk_forward(study.df_features.index, **study.split_kwargs)
    assert len(study.resultats) == 6
    for row in study.resultats.itertuples():
        train, test = folds[row.fold]
        ends = study.df_fin_cibles.loc[train, row.cible]
        allowed = ends.notna() & ends.lt(test[0])
        horizon = int(row.cible.rsplit("_", 1)[1])
        assert row.n_purge_train == horizon
        assert row.n_train == int(allowed.sum())
        assert ends[allowed].max() < test[0]
        assert row.n_train + row.n_purge_train == len(train)


def test_in_sample_remains_exploratory_without_purge(tmp_path):
    study = make_study(tmp_path)
    study.schema_split = "in_sample"
    study.split_kwargs = {}
    study.run()
    row = study.resultats.iloc[0]
    assert row.n_purge_train == 0
    assert row.n_train == row.n_test == 35
