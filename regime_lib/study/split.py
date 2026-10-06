"""
Découpage des données pour l'évaluation.

Trois schémas :

- `split_in_sample`   : tout l'index (analyse exploratoire)
- `split_holdout`     : une coupure, train / gap / test
- `split_walk_forward`: N fenêtres glissantes

L'embargo ajoute un espace entre train et test. Study purge en complément
les cibles du train dont la date de fin atteint la période de test.
"""

from __future__ import annotations

from numbers import Integral
from regime_lib.utils.validation import validate_time_index
import pandas as pd


def _validate(index, train_ratio, embargo):
    validate_time_index(index)
    if len(index) < 2:
        raise ValueError("Index trop court pour un train et un test non vides.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio doit être dans (0, 1).")
    if isinstance(embargo, bool) or not isinstance(embargo, Integral) or embargo < 0:
        raise ValueError("embargo doit être un entier >= 0.")


def split_in_sample(index: pd.DatetimeIndex) -> list[tuple]:
    """
    Renvoie une seule paire (index, index) : pas de séparation.

    Utile pour l'analyse exploratoire.
    """
    validate_time_index(index)
    if index.empty:
        raise ValueError("Index vide pour le split in_sample.")
    return [(index, index)]


def split_holdout(
    index: pd.DatetimeIndex,
    train_ratio: float = 0.7,
    embargo: int = 0,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Coupure unique train / test.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index complet, trié.
    train_ratio : float
        Fraction d'entraînement. Défaut 0.7.
    embargo : int
        Nombre de barres à ignorer entre train et test.
        Study purge automatiquement les cibles qui atteignent le test.

    Returns
    -------
    list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        Liste à une seule paire (train, test).
    """
    _validate(index, train_ratio, embargo)

    n = len(index)
    n_train = int(n * train_ratio)
    n_train = max(1, min(n_train, n - 1))

    train = index[:n_train]
    test = index[n_train + embargo:]
    if test.empty:
        raise ValueError("Embargo trop grand : test vide.")
    return [(train, test)]


def split_walk_forward(
    index: pd.DatetimeIndex,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    embargo: int = 0,
    anchored: bool = False,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Walk-forward : plusieurs paires train / test successives.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index complet, trié.
    n_folds : int
        Nombre de fenêtres.
    train_ratio : float
        Fraction train/(train+test), avant embargo et purge. En mode
        ancré, ce ratio définit le premier fold puis le train s'étend.
    embargo : int
        Barres ignorées entre train et test.
    anchored : bool
        Si True, le train commence toujours à l'indice 0 (fenêtre
        expansive). Si False, la fenêtre d'entraînement glisse
        (fenêtre glissante). Défaut : False.

    Returns
    -------
    list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        Liste de paires (train, test).
    """
    _validate(index, train_ratio, embargo)
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 1:
        raise ValueError("n_folds doit être un entier >= 1.")
    # T / (T + V) = ratio ; T + embargo + n_folds * V <= n.
    available = len(index) - embargo
    test_size = int(available / (n_folds + train_ratio / (1 - train_ratio)))
    train_size = max(1, round(test_size * train_ratio / (1 - train_ratio)))
    if test_size < 1 or train_size + embargo + n_folds * test_size > len(index):
        raise ValueError("Index trop court ou embargo trop grand pour les folds demandés.")
    pairs = []
    for fold in range(n_folds):
        end_train = train_size + fold * test_size
        start_train = 0 if anchored else end_train - train_size
        start_test = end_train + embargo
        pairs.append((index[start_train:end_train], index[start_test:start_test + test_size]))
    return pairs
