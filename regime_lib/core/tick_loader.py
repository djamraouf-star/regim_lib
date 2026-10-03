"""
Chargement et agrégation des données tick Dukascopy.

Format d'entrée
---------------
Parquet tick avec colonnes :
  - timestamp (DatetimeIndex UTC, index ou colonne)
  - bidPrice, askPrice
  - bidVolume, askVolume

Format de sortie
----------------
Barres OHLCV + colonnes annexes :

  OHLCV classique (basé sur le Bid) :
    - open, high, low, close
    - volume = bidVolume + askVolume

  Colonnes annexes (spread, participation) :
    - spread_mean    : moyenne de (ask - bid) sur la barre
    - spread_max     : maximum de (ask - bid)
    - spread_close   : (ask - bid) au tick de clôture
    - ask_close      : dernier ask de la barre
    - tick_count     : nombre de ticks dans la barre

  Marquage :
    - is_partial     : True pour la première et la dernière barre
                       (potentiellement incomplètes)

Choix de conception
-------------------
Le prix OHLCV est calculé sur le **Bid** — convention Dukascopy, et prix
réellement exécutable (vente). L'Ask est conservé séparément (`ask_close`)
pour permettre une reconstruction du Mid ou du coût d'achat si nécessaire.

Le spread n'est jamais moyenné en une seule valeur : mean, max et close
sont exposés séparément pour ne pas perdre l'information de tension.

Le volume est la somme des volumes bid et ask. Il ne reflète pas la
direction du flux — pour cela, il faudrait calculer un déséquilibre
(askVolume - bidVolume) / volume_total, non exposé par défaut.

Timeframes
----------
Les timeframes valides sont définis dans `TIMEFRAME_RULES`
(`core.data_loader`). Le resample suit la convention `label='right'`,
`closed='left'` : une barre étiquetée T couvre [T - période, T). Le tick
posé exactement sur la frontière T appartient donc à la barre suivante.
C'est la convention Dukascopy / Tickstory / MT5, et celle d'un agrégateur
live qui ferme la barre au premier tick dont l'horodatage est >= T. Elle
évite tout décalage de frontière entre backtest et live (biais lookahead).

Horodatage
----------
Les ticks Dukascopy ont une résolution milliseconde : plusieurs ticks
peuvent partager le même timestamp. Les doublons d'index sont donc
autorisés. Le tri est stable (`kind="stable"`) afin que `first` / `last`
(open / close, spread_close, ask_close) respectent l'ordre d'origine du
fichier pour les ticks de même timestamp.

L'index est ramené en UTC : un index naïf (sans fuseau) est interprété
comme UTC, un index avec fuseau est converti en UTC.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from regime_lib.core.data_loader import TIMEFRAME_RULES


# Colonnes obligatoires dans un parquet tick Dukascopy.
_COLONNES_TICK = ["bidPrice", "askPrice", "bidVolume", "askVolume"]

# Colonnes de sortie de `agreger_tick`.
_COLONNES_BARRES = [
    "open", "high", "low", "close", "volume",
    "spread_mean", "spread_max", "spread_close",
    "ask_close", "tick_count", "is_partial",
]


def detecter_sauts_tick(
    df_tick: pd.DataFrame,
    seuil: float | None = None,
    seuil_rel: float | None = None,
) -> pd.Series:
    """
    Compte, par jour UTC, les sauts de Bid tick à tick au-delà d'un seuil.

    Garde-fou de diagnostic pour repérer les artefacts de flux (ticks
    aberrants). Il ne modifie jamais les données.

    Le seuil est soit absolu (`seuil`, en unités de prix : propre à un
    instrument), soit relatif (`seuil_rel`, en fraction du Bid précédent :
    0.005 = 0.5 %, comparable d'un actif à l'autre). Exactement un des deux
    doit être renseigné.

    Attention : le premier tick après un week-end ou une fermeture peut
    légitimement sauter (gap d'ouverture). Lire le résultat comme une liste
    de dates à inspecter, pas comme une liste de ticks à supprimer.

    Parameters
    ----------
    df_tick : pd.DataFrame
        Sortie de `charger_tick` (index DatetimeIndex UTC).
    seuil : float, optional
        Saut absolu minimal (en unités de prix) pour être signalé.
    seuil_rel : float, optional
        Saut relatif minimal (fraction du Bid précédent) pour être signalé.

    Returns
    -------
    pd.Series
        Nombre de sauts par date (index = date UTC), uniquement les dates
        avec au moins un saut. Série vide si aucun saut.
    """
    if (seuil is None) == (seuil_rel is None):
        raise ValueError(
            "Renseigner exactement un seuil : `seuil` (absolu) ou "
            "`seuil_rel` (relatif)."
        )
    bid = df_tick["bidPrice"].to_numpy()
    saut = np.abs(np.diff(bid))
    if seuil_rel is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            saut = saut / np.abs(bid[:-1])
        limite = seuil_rel
    else:
        limite = seuil
    # saut[i] relie le tick i au tick i+1 → on date le saut au tick i+1.
    dates = df_tick.index[1:].normalize()
    comptes = pd.Series(saut > limite, index=dates).groupby(level=0).sum()
    comptes = comptes[comptes > 0].astype("int64")
    comptes.index.name = "date"
    return comptes


def charger_tick(
    url: str,
    saut_max: float | None = None,
    saut_max_rel: float | None = None,
) -> pd.DataFrame:
    """
    Charge un parquet tick Dukascopy et valide sa structure.

    Parameters
    ----------
    url : str
        Chemin local ou URL du parquet tick.
    saut_max : float, optional
        Seuil absolu (unités de prix) : émet un `UserWarning` listant les
        dates contenant au moins un saut de Bid tick à tick supérieur à ce
        seuil (voir `detecter_sauts_tick`). Aucune donnée n'est modifiée
        ni supprimée.
    saut_max_rel : float, optional
        Même contrôle avec un seuil relatif (fraction du Bid précédent,
        0.005 = 0.5 %), comparable d'un actif à l'autre. Exclusif avec
        `saut_max`. Par défaut : pas de contrôle.

    Returns
    -------
    pd.DataFrame
        DataFrame indexé par DatetimeIndex UTC nommé `timestamp`, trié
        (tri stable), avec colonnes `bidPrice`, `askPrice`, `bidVolume`,
        `askVolume`. Les timestamps dupliqués sont conservés.

    Raises
    ------
    ValueError
        Si les colonnes obligatoires manquent, si l'index temporel est
        invalide, ou si des incohérences de prix sont détectées.
    """
    df = pd.read_parquet(url)

    # Cas 1 : timestamp en colonne → promouvoir en index
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Le parquet tick doit avoir un DatetimeIndex ou une colonne "
            "'timestamp'."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    # Fuseau : tout est ramené en UTC (naïf → interprété comme UTC).
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"

    # Vérifier les colonnes
    manquantes = [c for c in _COLONNES_TICK if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes tick manquantes : {manquantes}. "
            f"Attendu : {_COLONNES_TICK}."
        )

    # Index trié. Tri stable : l'ordre d'origine des ticks de même
    # timestamp est préservé (first/last déterministes). Les doublons
    # d'index sont autorisés (résolution milliseconde).
    if not df.index.is_monotonic_increasing:
        df = df.sort_index(kind="stable")

    # Vérifier la cohérence bid ≤ ask
    spread = df["askPrice"] - df["bidPrice"]
    if (spread < 0).any():
        n_bad = (spread < 0).sum()
        raise ValueError(
            f"{n_bad} ticks avec ask < bid détectés. "
            f"Données incohérentes."
        )

    # Garde-fou optionnel : sauts de prix suspects (diagnostic seulement)
    if saut_max is not None and saut_max_rel is not None:
        raise ValueError(
            "`saut_max` et `saut_max_rel` sont exclusifs : en choisir un."
        )
    if saut_max is not None or saut_max_rel is not None:
        sauts = detecter_sauts_tick(
            df, seuil=saut_max, seuil_rel=saut_max_rel
        )
        if len(sauts) > 0:
            detail = ", ".join(
                f"{d.date()} ({n})" for d, n in sauts.items()
            )
            seuil_txt = (
                f"{saut_max}" if saut_max is not None
                else f"{saut_max_rel:.3%} du prix"
            )
            warnings.warn(
                f"Sauts de Bid tick à tick > {seuil_txt} détectés sur "
                f"{len(sauts)} date(s) : {detail}. Données non modifiées ; "
                f"à inspecter avant usage.",
                UserWarning,
                stacklevel=2,
            )

    return df


def agreger_tick(
    df_tick: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """
    Agrège un DataFrame tick en barres OHLCV + colonnes annexes.

    Parameters
    ----------
    df_tick : pd.DataFrame
        Sortie de `charger_tick` (index DatetimeIndex UTC).
    timeframe : str
        Clé de `TIMEFRAME_RULES` (ex. 'M1', 'H1', 'H4').

    Returns
    -------
    pd.DataFrame
        Barres OHLCV + spread + tick_count + is_partial, indexées par
        DatetimeIndex UTC. Une barre étiquetée T couvre [T - période, T).
        `is_partial` vaut True pour la première et la dernière barre.
    """
    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(
            f"Timeframe inconnu : {timeframe!r}. "
            f"Valides : {sorted(TIMEFRAME_RULES)}."
        )
    if len(df_tick) == 0:
        return pd.DataFrame(
            columns=_COLONNES_BARRES,
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    rule = TIMEFRAME_RULES[timeframe]

    # Sous-ensemble minimal de colonnes (économie mémoire sur ~10^8 ticks),
    # avec spread et volume précalculés une seule fois.
    df_work = pd.DataFrame(
        {
            "bidPrice": df_tick["bidPrice"].to_numpy(),
            "askPrice": df_tick["askPrice"].to_numpy(),
            "_spread": (
                df_tick["askPrice"].to_numpy()
                - df_tick["bidPrice"].to_numpy()
            ),
            "_volume": (
                df_tick["bidVolume"].to_numpy()
                + df_tick["askVolume"].to_numpy()
            ),
        },
        index=df_tick.index,
    )

    # Agrégation par resample
    # Convention label='right', closed='left' → barre [T-période, T)
    # étiquetée sur sa fin ; cohérent avec Dukascopy/Tickstory/MT5 et
    # avec un agrégateur live.
    agg = df_work.resample(rule, label="right", closed="left").agg(
        open=("bidPrice", "first"),
        high=("bidPrice", "max"),
        low=("bidPrice", "min"),
        close=("bidPrice", "last"),
        volume=("_volume", "sum"),
        spread_mean=("_spread", "mean"),
        spread_max=("_spread", "max"),
        spread_close=("_spread", "last"),
        ask_close=("askPrice", "last"),
        tick_count=("bidPrice", "size"),
    )

    # Supprimer les barres vides (aucun tick dans la fenêtre)
    agg = agg.dropna(subset=["open", "high", "low", "close"])

    # Marquage de la première et de la dernière barre comme partielles
    # (les données peuvent commencer / finir en milieu de période).
    agg["is_partial"] = False
    if len(agg) > 0:
        col = agg.columns.get_loc("is_partial")
        agg.iloc[0, col] = True
        agg.iloc[-1, col] = True

    # Types : tick_count en int64 (size ne produit pas de NaN)
    agg["tick_count"] = agg["tick_count"].astype("int64")

    agg.index.name = "timestamp"
    return agg


def charger_et_agreger(
    url: str,
    timeframe: str,
    saut_max: float | None = None,
    saut_max_rel: float | None = None,
) -> pd.DataFrame:
    """
    Raccourci : charger un tick + agréger en barres.

    Parameters
    ----------
    url : str
        Chemin du parquet tick.
    timeframe : str
        Timeframe cible.
    saut_max : float, optional
        Seuil absolu de saut de Bid tick à tick (garde-fou de diagnostic,
        voir `charger_tick`).
    saut_max_rel : float, optional
        Seuil relatif (fraction du prix), exclusif avec `saut_max`.
        Par défaut : pas de contrôle.

    Returns
    -------
    pd.DataFrame
        Barres OHLCV + colonnes annexes.
    """
    df_tick = charger_tick(url, saut_max=saut_max, saut_max_rel=saut_max_rel)
    return agreger_tick(df_tick, timeframe)
