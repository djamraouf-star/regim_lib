"""
Chargement unifié des données de marché : barres OHLCV ou flux de ticks.

Formats supportés
-----------------
1. **Barres OHLCV** : colonnes `open`, `high`, `low`, `close`, `volume`,
   index DatetimeIndex UTC.

2. **Ticks** : colonnes `bidPrice`, `askPrice`, `bidVolume`, `askVolume`,
   index DatetimeIndex UTC (résolution ms, doublons autorisés). Alias
   acceptés : `bid_price`, `Bid Price`, `bidprice`, etc.

Pour les ticks, l'agrégation en barres est **enrichie** :

    open, high, low, close    → OHLCV classique (basé sur le Bid)
    volume                    → bidVolume + askVolume
    bid_volume, ask_volume    → volumes séparés (pour l'OFI)
    spread_mean               → moyenne de (ask - bid)
    spread_max                → maximum de (ask - bid)
    spread_close              → (ask - bid) au tick de clôture
    ask_close                 → dernier ask de la barre
    tick_count                → nombre de ticks dans la barre
    is_partial                → True pour la première / dernière barre

Les colonnes annexes (`bid_volume`, `ask_volume`, `spread_*`,
`ask_close`, `tick_count`) sont propagées lors du resample vers un
timeframe supérieur, avec des règles d'agrégation adaptées.

Conventions de frontière (barres étiquetées par leur fin)
---------------------------------------------------------
Ticks → barres (`agreger_tick`) : `label='right'`, `closed='left'`.
Une barre étiquetée T couvre `[T - période, T)`. C'est la convention
Dukascopy / Tickstory / MT5 et d'un agrégateur live. Un tick posé
exactement sur la frontière T appartient à la barre suivante.

Barres → timeframe supérieur (`resample`) : `label='right'`,
`closed='right'`. Les barres source étant déjà étiquetées par leur fin,
la barre H1 étiquetée 01:00 regroupe les M1 étiquetés (00:00, 01:00],
soit exactement les ticks [00:00, 01:00). `closed='left'` décalerait
chaque barre d'une période source.

Ainsi ticks → H1 et ticks → M1 → H1 donnent des barres identiques.

Côté du prix (ticks uniquement)
-------------------------------
  - "bid" (défaut) : prix exécutable à la vente (convention Dukascopy) ;
  - "ask" : prix d'achat ;
  - "mid" : (bid + ask) / 2, neutre mais non exécutable.

Le côté retenu est enregistré dans `df.attrs["price_side"]`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from regime_lib.utils.validation import validate_ohlcv


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
TIMEFRAME_RULES: Final[dict[str, str]] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}

OHLCV_AGG: Final[dict] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": lambda values: values.sum(min_count=1),
}

# Règles d'agrégation pour les colonnes annexes produites par
# `agreger_tick`. Appliquées lors du resample vers un timeframe
# supérieur (`resample`), si les colonnes sont présentes.
COLONNES_ANNEXES_AGG: Final[dict] = {
    "bid_volume": "sum",
    "ask_volume": "sum",
    "tick_count": "sum",
    "spread_mean": "mean",
    "spread_max": "max",
    "spread_close": "last",
    "ask_close": "last",
}

TICK_TIMESTAMP_COLUMNS: Final[tuple[str, ...]] = (
    "timestamp", "datetime", "date", "time",
)

PRICE_SIDES: Final[tuple[str, ...]] = ("bid", "ask", "mid")

# Colonnes tick canoniques après normalisation
_COLONNES_TICK: Final[tuple[str, ...]] = (
    "bidPrice", "askPrice", "bidVolume", "askVolume",
)

# Colonnes de sortie d'`agreger_tick`
_COLONNES_BARRES_TICK: Final[list[str]] = [
    "open", "high", "low", "close", "volume",
    "bid_volume", "ask_volume",
    "spread_mean", "spread_max", "spread_close",
    "ask_close", "tick_count", "is_partial",
]

# Alias de colonnes : clé = forme canonique après suppression des
# séparateurs et mise en minuscules. Valeur = nom canonique.
_ALIASES_COLONNES: Final[dict[str, str]] = {
    "bidprice": "bidPrice",
    "askprice": "askPrice",
    "bidvolume": "bidVolume",
    "askvolume": "askVolume",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_price_side(prix: str) -> None:
    if prix not in PRICE_SIDES:
        raise ValueError(
            f"Côté de prix inconnu : {prix!r}. Valides : {list(PRICE_SIDES)}."
        )


def _normalize_column_name(name: object) -> object:
    """Normalise un nom de colonne vers sa forme canonique."""
    if not isinstance(name, str):
        return name
    key = name.strip().lower().replace("-", "").replace("_", "")
    key = key.replace(" ", "")
    if key in _ALIASES_COLONNES:
        return _ALIASES_COLONNES[key]
    # Nom hors alias : minuscules et underscores (timestamp, date, ...).
    return "_".join(name.strip().lower().replace("-", " ").split())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes vers leur forme canonique."""
    normalized = [_normalize_column_name(name) for name in df.columns]
    duplicates = sorted({
        name for name in normalized if normalized.count(name) > 1
    })
    if duplicates:
        raise ValueError(
            f"Colonnes dupliquées après normalisation : {duplicates}."
        )
    result = df.copy()
    result.columns = normalized
    return result


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Force l'index DatetimeIndex en UTC."""
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df


def _tick_prices(df: pd.DataFrame, prix: str) -> pd.Series:
    """Série de prix des ticks selon le côté demandé."""
    _check_price_side(prix)
    if prix == "bid":
        s = pd.to_numeric(df["bidPrice"], errors="raise")
    elif prix == "ask":
        s = pd.to_numeric(df["askPrice"], errors="raise")
    else:  # mid
        b = pd.to_numeric(df["bidPrice"], errors="raise")
        a = pd.to_numeric(df["askPrice"], errors="raise")
        s = (a + b) / 2.0
    if s.isna().any():
        raise ValueError(
            f"Le prix demandé ({prix}) contient des valeurs nulles."
        )
    return s


# ---------------------------------------------------------------------------
# Chargement tick
# ---------------------------------------------------------------------------
def detecter_sauts_tick(
    df_tick: pd.DataFrame,
    seuil: float | None = None,
    seuil_rel: float | None = None,
) -> pd.Series:
    """
    Compte, par jour UTC, les sauts de Bid tick à tick au-delà d'un seuil.

    Garde-fou de diagnostic (aucune donnée modifiée). Le seuil est soit
    absolu (`seuil`, propre à l'instrument), soit relatif (`seuil_rel`,
    fraction du Bid précédent, comparable entre actifs).
    """
    if (seuil is None) == (seuil_rel is None):
        raise ValueError(
            "Renseigner exactement un seuil : `seuil` ou `seuil_rel`."
        )
    bid = df_tick["bidPrice"].to_numpy()
    saut = np.abs(np.diff(bid))
    if seuil_rel is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            saut = saut / np.abs(bid[:-1])
        limite = seuil_rel
    else:
        limite = seuil
    dates = df_tick.index[1:].normalize()
    comptes = pd.Series(saut > limite, index=dates).groupby(level=0).sum()
    comptes = comptes[comptes > 0].astype("int64")
    comptes.index.name = "date"
    return comptes


def charger_tick(
    url: str,
    saut_max: float | None = None,
    saut_max_rel: float | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """
    Charge un parquet tick Dukascopy (local ou distant).

    Parameters
    ----------
    url : str
        Chemin local ou URL du parquet tick.
    saut_max : float, optional
        Seuil absolu de saut de Bid tick à tick → UserWarning.
    saut_max_rel : float, optional
        Seuil relatif (fraction du prix) → UserWarning. Exclusif avec
        `saut_max`.
    batch_size : int, optional
        Si fourni, chargement par lots (streaming). Recommandé pour les
        fichiers > 1 Go. Défaut : chargement complet en mémoire.

    Returns
    -------
    pd.DataFrame
        Index DatetimeIndex UTC nommé `timestamp`, colonnes canoniques
        `bidPrice`, `askPrice`, `bidVolume`, `askVolume`. Doublons
        d'index autorisés (résolution ms).
    """
    if saut_max is not None and saut_max_rel is not None:
        raise ValueError(
            "`saut_max` et `saut_max_rel` sont exclusifs."
        )

    local_path = Path(url)
    if batch_size is not None and local_path.is_file():
        df = _charger_tick_par_lots(local_path, batch_size)
    else:
        df = _normalize_columns(pd.read_parquet(url))
    return _preparer_tick(df, saut_max, saut_max_rel)


def _preparer_tick(
    df: pd.DataFrame,
    saut_max: float | None = None,
    saut_max_rel: float | None = None,
) -> pd.DataFrame:
    """
    Valide et prépare un DataFrame tick déjà lu (colonnes normalisées) :
    index UTC, colonnes obligatoires, tri stable, bid ≤ ask, garde-fou de
    sauts.
    """
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Le parquet tick doit avoir un DatetimeIndex ou une "
            "colonne 'timestamp'."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)

    df = _ensure_utc_index(df)

    manquantes = [c for c in _COLONNES_TICK if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes tick manquantes : {manquantes}. "
            f"Attendu : {list(_COLONNES_TICK)}."
        )

    if not df.index.is_monotonic_increasing:
        df = df.sort_index(kind="stable")

    spread = df["askPrice"] - df["bidPrice"]
    if (spread < 0).any():
        n_bad = int((spread < 0).sum())
        raise ValueError(
            f"{n_bad} ticks avec ask < bid détectés. Données incohérentes."
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
                f"Sauts de Bid tick à tick > {seuil_txt} sur "
                f"{len(sauts)} date(s) : {detail}. Données non modifiées.",
                UserWarning,
                stacklevel=3,
            )

    return df


def _charger_tick_par_lots(
    path: Path,
    batch_size: int = 1_000_000,
) -> pd.DataFrame:
    """Charge un parquet tick local en streaming par lots."""
    parquet = pq.ParquetFile(path)
    norm_vers_reel = {
        _normalize_column_name(c): c for c in parquet.schema.names
    }
    ts_col = next(
        (norm_vers_reel[c] for c in TICK_TIMESTAMP_COLUMNS
         if c in norm_vers_reel),
        None,
    )
    if ts_col is None:
        raise ValueError(
            "Aucune colonne timestamp/datetime/date/time trouvée."
        )

    canonique_vers_reel: dict[str, str] = {}
    for col in parquet.schema.names:
        norm = _normalize_column_name(col)
        if norm in _COLONNES_TICK:
            canonique_vers_reel[norm] = col

    manquantes = [
        c for c in _COLONNES_TICK if c not in canonique_vers_reel
    ]
    if manquantes:
        raise ValueError(
            f"Colonnes tick manquantes dans le parquet : {manquantes}."
        )

    cols_a_lire = [ts_col] + [
        canonique_vers_reel[c] for c in _COLONNES_TICK
    ]

    chunks: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=cols_a_lire
    ):
        chunk = batch.to_pandas()
        renommage = {ts_col: "timestamp"}
        for canon, reel in canonique_vers_reel.items():
            renommage[reel] = canon
        chunk = chunk.rename(columns=renommage)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True, errors="raise"
    )
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    return df


# ---------------------------------------------------------------------------
# Agrégation tick → barres
# ---------------------------------------------------------------------------
def agreger_tick(
    df_tick: pd.DataFrame,
    timeframe: str,
    prix: str = "bid",
) -> pd.DataFrame:
    """
    Agrège un DataFrame tick en barres OHLCV enrichies.

    Parameters
    ----------
    df_tick : pd.DataFrame
        Sortie de `charger_tick`.
    timeframe : str
        Clé de TIMEFRAME_RULES.
    prix : {"bid", "ask", "mid"}
        Côté de prix pour OHLCV. Défaut "bid".

    Returns
    -------
    pd.DataFrame
        Barres OHLCV + colonnes annexes. Barre étiquetée T couvre
        [T - période, T).
    """
    _check_price_side(prix)
    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(
            f"Timeframe inconnu : {timeframe!r}. "
            f"Valides : {sorted(TIMEFRAME_RULES)}."
        )
    if len(df_tick) == 0:
        return pd.DataFrame(
            columns=_COLONNES_BARRES_TICK,
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    rule = TIMEFRAME_RULES[timeframe]

    df_work = pd.DataFrame(
        {
            "prix": _tick_prices(df_tick, prix).to_numpy(),
            "askPrice": df_tick["askPrice"].to_numpy(),
            "bidVolume": df_tick["bidVolume"].to_numpy(),
            "askVolume": df_tick["askVolume"].to_numpy(),
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

    agg = df_work.resample(rule, label="right", closed="left").agg(
        open=("prix", "first"),
        high=("prix", "max"),
        low=("prix", "min"),
        close=("prix", "last"),
        volume=("_volume", "sum"),
        bid_volume=("bidVolume", "sum"),
        ask_volume=("askVolume", "sum"),
        spread_mean=("_spread", "mean"),
        spread_max=("_spread", "max"),
        spread_close=("_spread", "last"),
        ask_close=("askPrice", "last"),
        tick_count=("prix", "size"),
    )

    agg = agg.dropna(subset=["open", "high", "low", "close"])

    agg["is_partial"] = False
    if len(agg) > 0:
        col = agg.columns.get_loc("is_partial")
        agg.iloc[0, col] = True
        agg.iloc[-1, col] = True

    agg["tick_count"] = agg["tick_count"].astype("int64")
    agg.index.name = "timestamp"
    return agg


def charger_et_agreger(
    url: str,
    timeframe: str,
    prix: str = "bid",
    saut_max: float | None = None,
    saut_max_rel: float | None = None,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Raccourci : charger un tick + agréger."""
    df_tick = charger_tick(
        url, saut_max=saut_max, saut_max_rel=saut_max_rel,
        batch_size=batch_size,
    )
    return agreger_tick(df_tick, timeframe, prix=prix)


# ---------------------------------------------------------------------------
# Chargement barres OHLCV
# ---------------------------------------------------------------------------
def infer_asset_from_url(url: str) -> str:
    """Déduit un symbole depuis le nom de fichier ('EURUSD_M1.parquet')."""
    stem = Path(url).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample un DataFrame OHLCV vers le timeframe cible.

    Convention : label='right', closed='right'. Les barres source sont
    étiquetées par leur fin : la barre étiquetée T regroupe les barres
    source étiquetées (T - période, T], soit les ticks [T - période, T)
    comme dans `agreger_tick`.

    Les colonnes annexes (`bid_volume`, `ask_volume`, `spread_*`,
    `ask_close`, `tick_count`) sont également agrégées si présentes,
    avec les règles définies dans `COLONNES_ANNEXES_AGG`.

    La colonne `is_partial` marque la dernière barre (potentiellement
    incomplète) et toute barre dont la fin dépasse la dernière
    observation source.
    """
    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(
            f"Timeframe inconnu : {timeframe!r}. "
            f"Valides : {sorted(TIMEFRAME_RULES)}."
        )

    rule = TIMEFRAME_RULES[timeframe]

    # Construire le dict d'agrégation : OHLCV + annexes présentes.
    agg_dict = dict(OHLCV_AGG)
    for col, regle in COLONNES_ANNEXES_AGG.items():
        if col in df.columns:
            agg_dict[col] = regle

    resampled = df.resample(
        rule, label="right", closed="right"
    ).agg(agg_dict)

    last_src_ts = df.index.max()
    resampled["is_partial"] = False
    if len(resampled) > 0:
        resampled.iloc[-1, resampled.columns.get_loc("is_partial")] = True
        mask_oob = resampled.index > last_src_ts
        resampled.loc[mask_oob, "is_partial"] = True

    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    # Forcer tick_count en int64 si présent (somme de booléens/ints).
    if "tick_count" in resampled.columns:
        resampled["tick_count"] = resampled["tick_count"].astype("int64")

    if "price_side" in df.attrs:
        resampled.attrs["price_side"] = df.attrs["price_side"]

    return resampled


# ---------------------------------------------------------------------------
# Chargeur unifié
# ---------------------------------------------------------------------------
def load_parquet(
    url: str,
    asset: str | None = None,
    prix: str = "bid",
    batch_size: int | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Charge un parquet OHLCV ou tick. Détecte automatiquement le format.

    Parameters
    ----------
    url : str
        Chemin local ou URL.
    asset : str | None
        Symbole explicite. Fallback : inféré du nom de fichier.
    prix : {"bid", "ask", "mid"}
        Côté de prix pour l'agrégation tick (défaut "bid"). Ignoré pour
        les barres.
    batch_size : int | None
        Taille de lot pour le streaming des ticks (local uniquement).
        Recommandé > 10 M ticks.

    Returns
    -------
    (df, asset) : tuple[pd.DataFrame, str]
        - Si barres : OHLCV validé.
        - Si ticks : M1 enrichi (spread, tick_count, bid/ask volumes).
        - `df.attrs["price_side"]` = prix pour ticks, None pour barres.
        - `df.attrs["source_type"]` = "barres" ou "ticks".
        - `df.attrs["source_rows"]` = lignes originales.
    """
    if not url:
        raise ValueError("URL vide.")
    _check_price_side(prix)

    local_path = Path(url)
    source_type: str | None = None
    source_rows: int | None = None

    if local_path.is_file():
        parquet = pq.ParquetFile(local_path)
        source_rows = int(parquet.metadata.num_rows)
        original_columns = parquet.schema.names
        normalized = [_normalize_column_name(c) for c in original_columns]
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "Noms de colonnes ambigus après normalisation."
            )
        column_map = dict(zip(normalized, original_columns))

        est_tick = (
            any(c in column_map for c in _COLONNES_TICK[:2])  # bid ou ask
            and any(c in column_map for c in TICK_TIMESTAMP_COLUMNS)
        )

        if est_tick:
            source_type = "ticks"
            df = charger_tick(
                url, batch_size=batch_size
            )
            df = agreger_tick(df, "M1", prix=prix)
        else:
            source_type = "barres"
            df = _normalize_columns(pd.read_parquet(url))
    else:
        # URL distante : pas de streaming
        df = _normalize_columns(pd.read_parquet(url))
        if any(c in df.columns for c in _COLONNES_TICK[:2]):
            source_type = "ticks"
            source_rows = len(df)
            df = agreger_tick(_preparer_tick(df), "M1", prix=prix)
        else:
            source_type = "barres"

    # Traitement barres
    if source_type != "ticks":
        if not isinstance(df.index, pd.DatetimeIndex):
            for candidate in TICK_TIMESTAMP_COLUMNS:
                if candidate in df.columns:
                    df = df.set_index(candidate)
                    break
            else:
                raise ValueError(
                    "Le parquet doit avoir un DatetimeIndex ou une "
                    "colonne timestamp/datetime/date/time."
                )

        df = _ensure_utc_index(df)
        if source_rows is None:
            source_rows = len(df)
        df = df.sort_index(kind="stable")
        validate_ohlcv(df)

    df.attrs["source_type"] = source_type
    df.attrs["source_rows"] = source_rows
    df.attrs["price_side"] = prix if source_type == "ticks" else None

    if asset is None:
        inferred = infer_asset_from_url(url)
        if not inferred:
            raise ValueError(
                "Impossible d'inférer l'actif : fournir --asset."
            )
        asset = inferred

    return df, asset
