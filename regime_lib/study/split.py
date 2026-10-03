"""
Découpage des données pour l'évaluation.

Trois schémas :

- `split_in_sample`   : tout l'index (analyse exploratoire)
- `split_holdout`     : une coupure, train / gap / test
- `split_walk_forward`: N fenêtres glissantes

L'embargo (gap) est indispensable quand les cibles utilisent le futur :
sans lui, les dernières barres du train ont des cibles qui empiètent
sur le test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_in_sample(index: pd.DatetimeIndex) -> list[tuple]:
    """
    Renvoie une seule paire (index, index) : pas de séparation.

    Utile pour l'analyse exploratoire.
    """
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
        À mettre ≥ horizon de la cible pour éviter les fuites.

    Returns
    -------
    list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        Liste à une seule paire (train, test).
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio doit être dans (0, 1).")
    if embargo < 0:
        raise ValueError("embargo doit être >= 0.")

    n = len(index)
    n_train = int(n * train_ratio)
    n_train = max(1, min(n_train, n - 1))

    train = index[:n_train]
    test = index[n_train + embargo:]
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
        Fraction d'entraînement dans chaque fenêtre (si non ancré).
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
    if n_folds < 1:
        raise ValueError("n_folds doit être >= 1.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio doit être dans (0, 1).")
    if embargo < 0:
        raise ValueError("embargo doit être >= 0.")

    n = len(index)
    taille_fenetre = n // (n_folds + 1)
    if taille_fenetre < 2:
        raise ValueError(
            f"Index trop court ({n}) pour {n_folds} folds."
        )

    paires = []
    for fold in range(n_folds):
        fin_train = taille_fenetre * (fold + 1)
        debut_train = 0 if anchored else fin_train - taille_fenetre
        debut_test = fin_train + embargo
        fin_test = min(debut_test + taille_fenetre, n)

        if debut_test >= n:
            break

        train = index[debut_train:fin_train]
        test = index[debut_test:fin_test]
        if len(train) > 0 and len(test) > 0:
            paires.append((train, test))

    return paires
