"""Tests des schémas de split."""

from __future__ import annotations

import pandas as pd
import pytest

from regime_lib.study.split import (
    split_holdout,
    split_in_sample,
    split_walk_forward,
)


@pytest.fixture
def index_1000():
    return pd.date_range("2026-01-01", periods=1000, freq="1h", tz="UTC")


def test_in_sample(index_1000):
    paires = split_in_sample(index_1000)
    assert len(paires) == 1
    train, test = paires[0]
    assert len(train) == len(index_1000)
    assert train.equals(test)


def test_holdout_basique(index_1000):
    paires = split_holdout(index_1000, train_ratio=0.7)
    assert len(paires) == 1
    train, test = paires[0]
    assert len(train) == 700
    assert len(test) == 300


def test_holdout_embargo(index_1000):
    paires = split_holdout(index_1000, train_ratio=0.7, embargo=20)
    train, test = paires[0]
    assert len(train) == 700
    # Le test commence 20 barres après la fin du train
    assert test[0] == index_1000[720]


def test_holdout_ratio_invalide(index_1000):
    with pytest.raises(ValueError, match="train_ratio"):
        split_holdout(index_1000, train_ratio=0.0)
    with pytest.raises(ValueError, match="train_ratio"):
        split_holdout(index_1000, train_ratio=1.0)


def test_walk_forward(index_1000):
    paires = split_walk_forward(index_1000, n_folds=5, train_ratio=0.5)
    # Au plus n_folds paires
    assert 1 <= len(paires) <= 5
    # Pas de chevauchement train/test
    for train, test in paires:
        assert train[-1] < test[0]


def test_walk_forward_ancored(index_1000):
    paires = split_walk_forward(index_1000, n_folds=3, anchored=True)
    # Avec anchored=True, le train démarre toujours à l'indice 0
    for train, _ in paires:
        assert train[0] == index_1000[0]


def test_walk_forward_glissant(index_1000):
    paires = split_walk_forward(index_1000, n_folds=3, anchored=False)
    # Avec anchored=False, le train ne démarre pas à 0 (sauf premier)
    if len(paires) > 1:
        assert paires[1][0][0] != index_1000[0]


def test_walk_forward_n_folds_invalide(index_1000):
    with pytest.raises(ValueError, match="n_folds"):
        split_walk_forward(index_1000, n_folds=0)
