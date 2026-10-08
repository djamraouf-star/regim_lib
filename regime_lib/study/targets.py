"""
Cibles d'étude : ce que les régimes doivent prédire.

Une cible est une fonction `df_ohlcv → Series` indexée par timestamp.

Familles supportées
-------------------
- `ret_fwd_K` : log-return forward sur K barres
               log(close[t+K] / close[t])
- `vol_fwd_K` : volatilité réalisée forward
               écart-type des log-returns sur [t, t+K]
- `dd_fwd_K`  : excursion minimale des clôtures (alias close_min)
               min(close[t:t+K+1]) / close[t] - 1
- `ru_fwd_K`  : excursion maximale des clôtures (alias close_max)
               max(close[t:t+K+1]) / close[t] - 1

Convention de nommage : `<famille>_fwd_<K>`, K entier strictement
positif.

Lookahead
---------
Les cibles utilisent **structurellement le futur** : c'est leur rôle.
Elles servent de vérité terrain pour évaluer les features. Les K
dernières barres de chaque cible sont NaN.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from regime_lib.core.temporal import resolve_timeframe, expected_grid
from regime_lib.utils.validation import validate_time_index, validity_mask


_PATTERN = re.compile(r"^([a-z_]+)_fwd_(\d+)$")

EXCURSIONS = {"mfe_long", "mae_long", "mfe_short", "mae_short"}
FAMILLES_VALIDES = {"ret", "vol", "dd", "ru", "close_min", "close_max"} | EXCURSIONS | {"time_" + f for f in EXCURSIONS}


def parser_cible(nom: str) -> tuple[str, int]:
    """
    Parse 'ret_fwd_5' → ('ret', 5).

    Raises
    ------
    ValueError
        Si le nom ne suit pas la convention ou si la famille est
        inconnue.
    """
    match = _PATTERN.match(nom)
    if match is None:
        raise ValueError(
            f"Nom de cible invalide : {nom!r}. "
            f"Format attendu : '<famille>_fwd_<K>' (ex. 'ret_fwd_5')."
        )
    famille, k_str = match.groups()
    if famille not in FAMILLES_VALIDES:
        raise ValueError(
            f"Famille de cible inconnue : {famille!r}. "
            f"Valides : {sorted(FAMILLES_VALIDES)}."
        )
    k = int(k_str)
    if k < 1:
        raise ValueError(f"K doit être >= 1, reçu {k}.")
    if famille == "vol" and k < 2:
        raise ValueError("vol_fwd_1 non définie : la volatilité exige K >= 2.")
    return famille, k


def calculer_cible(
    ohlcv: pd.DataFrame, nom: str, *, timeframe: str | None = None,
    expected_index: pd.DatetimeIndex | None = None,
) -> pd.Series:
    """
    Calcule une cible à partir d'un DataFrame OHLCV.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Doit contenir une colonne `close`, indexé par DatetimeIndex UTC.
    nom : str
        Nom de la cible (ex. 'ret_fwd_5', 'vol_fwd_20').

    Returns
    -------
    pd.Series
        Série nommée `nom`, indexée par le même DatetimeIndex que
        `ohlcv`. Les K dernières barres sont NaN.
    """
    famille, k = parser_cible(nom)

    if "close" not in ohlcv.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'close'.")

    validate_time_index(ohlcv.index)
    original_index = ohlcv.index
    values = ohlcv["close"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Les clôtures doivent être finies et strictement positives.")
    tf = resolve_timeframe(ohlcv, timeframe)
    grid = expected_grid(original_index, tf, expected_index)
    valid = validity_mask(ohlcv).reindex(grid, fill_value=False)
    close = ohlcv["close"].reindex(grid)

    if famille == "ret":
        cible = np.log(close.shift(-k) / close)

    elif famille == "vol":
        log_ret = np.log(close / close.shift(1))
        # Écart-type des log-returns sur la fenêtre [t+1, t+k]
        cible = log_ret.shift(-1).rolling(k, min_periods=k).std()
        cible = cible.shift(-(k - 1))

    elif famille in {"dd", "close_min"}:
        # Drawdown : inclut close[t] pour garantir dd ≤ 0
        cible = pd.Series(np.nan, index=close.index, dtype=float)
        arr = close.to_numpy()
        n = len(arr)
        for t in range(n - k):
            fenetre = arr[t : t + k + 1]   # inclut close[t]
            cible.iloc[t] = fenetre.min() / arr[t] - 1

    elif famille in {"ru", "close_max"}:
        # Run-up : inclut close[t] pour garantir ru ≥ 0
        cible = pd.Series(np.nan, index=close.index, dtype=float)
        arr = close.to_numpy()
        n = len(arr)
        for t in range(n - k):
            fenetre = arr[t : t + k + 1]   # inclut close[t]
            cible.iloc[t] = fenetre.max() / arr[t] - 1

    elif famille in EXCURSIONS or famille.startswith("time_"):
        for column in ("high", "low"):
            if column not in ohlcv:
                raise ValueError(f"La cible {nom} requiert {column}.")
            vals = ohlcv[column].to_numpy(dtype=float)
            if not np.isfinite(vals).all() or (vals <= 0).any():
                raise ValueError(f"{column} doit être fini et strictement positif.")
        if ((ohlcv.high < ohlcv.close) | (ohlcv.low > ohlcv.close)
                | (ohlcv.high < ohlcv.low)).any():
            raise ValueError("Relations high/low/close incohérentes.")
        timed = famille.startswith("time_")
        kind = famille.removeprefix("time_")
        use_high = kind in {"mfe_long", "mae_short"}
        prices = ohlcv["high" if use_high else "low"].reindex(grid).to_numpy()
        entries = close.to_numpy()
        result = np.full(len(close), np.nan)
        for t in range(len(close) - k):
            # Entrée à close[t] : seuls les extrêmes de t+1 ... t+K
            # sont accessibles. L'entrée est l'extrême initial (temps zéro).
            path = np.r_[entries[t], prices[t + 1:t + k + 1]]
            if not np.isfinite(path).all():
                continue
            position = int(np.argmax(path) if use_high else np.argmin(path))
            value = path[position] / entries[t] - 1
            if kind.endswith("short"):
                value = -value  # P&L normalisé par le prix d'entrée
            result[t] = position if timed else value
        cible = pd.Series(result, index=grid)

    else:  # pragma: no cover — famille validée par parser_cible
        raise ValueError(f"Famille non implémentée : {famille}")

    # Toute la trajectoire doit être observable, même pour un rendement terminal.
    complete = valid.astype(int).rolling(k + 1, min_periods=k + 1).sum().shift(-k).eq(k + 1)
    cible = cible.where(complete).reindex(original_index)
    cible.name = nom
    return cible


def liste_cibles(
    horizons: list[int] | None = None,
    familles: list[str] | None = None,
) -> list[str]:
    """
    Liste des noms de cibles standard.

    Parameters
    ----------
    horizons : list[int] | None
        Horizons K. Défaut : [1, 5, 20].
    familles : list[str] | None
        Familles. Défaut : ['ret', 'vol'].

    Returns
    -------
    list[str]
        Noms triés (ex. ['ret_fwd_1', 'ret_fwd_5', 'vol_fwd_5', ...]).
    """
    if horizons is None:
        horizons = [1, 5, 20]
    if familles is None:
        familles = ["ret", "vol"]
    return [
        f"{fam}_fwd_{k}"
        for fam in familles
        for k in horizons
        if not (fam == "vol" and k == 1)
    ]
