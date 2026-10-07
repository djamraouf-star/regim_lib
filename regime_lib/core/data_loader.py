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

from regime_lib.utils.validation import validate_ohlcv, validity_mask
from regime_lib.core.temporal import (
    TIMEFRAME_RULES, TIMEFRAME_SECONDS, grid_origin, resolve_timeframe, expected_grid,
)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
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
        Taille des lots lus depuis un parquet local. Les lots sont ensuite
        accumulés et concaténés en mémoire : ce paramètre réduit la mémoire
        de lecture transitoire, mais ne borne pas la mémoire totale.

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
    df = _normalize_columns(df)
    ts_columns = [c for c in TICK_TIMESTAMP_COLUMNS if c in df.columns]
    if len(ts_columns) > 1:
        raise ValueError("Plusieurs colonnes temporelles tick : sélection ambiguë.")
    if ts_columns:
        df = df.set_index(ts_columns[0])
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "Le parquet tick doit avoir un DatetimeIndex ou une "
            "colonne 'timestamp'."
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, errors="raise"))

    df = _ensure_utc_index(df)

    manquantes = [c for c in _COLONNES_TICK if c not in df.columns]
    if manquantes:
        raise ValueError(
            f"Colonnes tick manquantes : {manquantes}. "
            f"Attendu : {list(_COLONNES_TICK)}."
        )

    if not df.index.is_monotonic_increasing:
        df = df.sort_index(kind="stable")

    if df.index.hasnans:
        raise ValueError("Timestamp tick manquant.")
    for col in _COLONNES_TICK:
        df[col] = pd.to_numeric(df[col], errors="raise")
    values = df[list(_COLONNES_TICK)].to_numpy(dtype=float)
    if (not np.isfinite(values).all() or (values[:, :2] <= 0).any()
            or (values[:, 2:] < 0).any()):
        raise ValueError("Prix tick positifs et volumes non négatifs, tous finis, requis.")

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
    """Lit un parquet tick local par lots, puis matérialise tous les ticks."""
    parquet = pq.ParquetFile(path)
    if batch_size <= 0:
        raise ValueError("batch_size doit être strictement positif.")
    chunks = [batch.to_pandas() for batch in parquet.iter_batches(batch_size=batch_size)]
    if not chunks:
        return _normalize_columns(parquet.read().to_pandas())
    return _normalize_columns(pd.concat(chunks))


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
    df_tick = _preparer_tick(df_tick)
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

    agg = df_work.resample(rule, label="right", closed="left",
                                origin=grid_origin(df_work.index, timeframe)).agg(
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
    agg.attrs.update(timeframe=timeframe, timestamp_convention="close",
                     calendar="continuous_utc_v1", price_side=prix)
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


def resample(
    df: pd.DataFrame, timeframe: str, *, source_timeframe: str | None = None,
    expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Agrégation de clôtures UTC ; préserve partialité et couverture source.

    Sans calendrier explicite, les barres sont attendues en continu UTC.
    `expected_index` énumère les clôtures source attendues dans les séances.
    Une conversion vers une fréquence plus fine ou non divisible est refusée.
    """
    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(f"Timeframe inconnu : {timeframe!r}.")
    validate_ohlcv(df)
    if df.attrs.get("timestamp_convention", "close") != "close":
        raise ValueError("Convertir explicitement les barres d'ouverture en clôtures.")
    source_tf = resolve_timeframe(df, source_timeframe)
    source_step = TIMEFRAME_SECONDS[source_tf]
    target_step = TIMEFRAME_SECONDS[timeframe]
    if target_step < source_step or target_step % source_step:
        raise ValueError("Conversion impossible : fréquence cible plus fine ou non divisible.")
    work = df.copy()
    work["is_partial"] = ~validity_mask(df)
    kwargs = dict(rule=TIMEFRAME_RULES[timeframe], label="right", closed="right",
                  origin=grid_origin(df.index, timeframe))
    agg_dict = dict(OHLCV_AGG)
    agg_dict.update({c: rule for c, rule in COLONNES_ANNEXES_AGG.items() if c in df})
    agg_dict["is_partial"] = "max"
    result = work.resample(**kwargs).agg(agg_dict)
    actual = pd.Series(1, index=df.index, dtype="int64").resample(**kwargs).sum()
    if expected_index is None:
        expected = pd.Series(target_step // source_step, index=result.index)
    else:
        grid = expected_grid(df.index, source_tf, expected_index)
        expected = pd.Series(1, index=grid, dtype="int64").resample(**kwargs).sum()
        expected = expected.reindex(result.index, fill_value=0)
    result["source_count"] = actual
    result["expected_count"] = expected
    result["coverage"] = actual.div(expected.where(expected > 0))
    # Composition : une barre source déjà incomplète ne devient pas complète.
    result["is_partial"] = result["is_partial"].fillna(True).astype(bool) | actual.ne(expected)
    result = result.dropna(subset=["open", "high", "low", "close"])
    if "tick_count" in result:
        result["tick_count"] = result["tick_count"].astype("int64")
    result.attrs = dict(df.attrs)
    result.attrs.update(timeframe=timeframe, timestamp_convention="close",
                        calendar=df.attrs.get("calendar", "continuous_utc_v1"))
    if expected_index is not None and result.attrs["calendar"] == "continuous_utc_v1":
        result.attrs["calendar"] = "explicit_schedule"
    return result


# ---------------------------------------------------------------------------
# Chargeur unifié
# ---------------------------------------------------------------------------
def load_parquet(
    url: str,
    asset: str | None = None,
    prix: str = "bid",
    batch_size: int | None = None,
    *, timeframe: str | None = None,
    timestamp_convention: str | None = None,
    source_timezone: str = "UTC",
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
        Taille des lots lus pour les ticks (local uniquement). Les lots
        restent accumulés avant concaténation et agrégation ; cela ne borne
        pas la mémoire totale. Voir `charger_tick`.

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

        est_tick = any(c in column_map for c in _COLONNES_TICK[:2])

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
        # Les alias tick sont normalisés en camelCase à la lecture. Sur des
        # barres, restaurer les noms attendus par le resampling et les méthodes.
        df = df.rename(columns={"bidVolume": "bid_volume", "askVolume": "ask_volume"})
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

        df.index = pd.DatetimeIndex(pd.to_datetime(df.index, errors="raise"))
        if df.index.tz is None:
            df.index = df.index.tz_localize(source_timezone)
        df = _ensure_utc_index(df)
        convention = timestamp_convention or df.attrs.get("timestamp_convention", "close")
        if convention not in ("open", "close"):
            raise ValueError("timestamp_convention doit être open ou close.")
        if convention == "open":
            tf = timeframe or df.attrs.get("timeframe")
            if tf not in TIMEFRAME_SECONDS:
                raise ValueError("Déclarer le timeframe pour convertir les ouvertures.")
            df.index = df.index + pd.Timedelta(seconds=TIMEFRAME_SECONDS[tf])
        df.attrs["original_timestamp_convention"] = convention
        df.attrs["timestamp_convention"] = "close"
        df.attrs["source_timezone"] = source_timezone
        if timeframe is not None:
            if df.attrs.get("timeframe") not in (None, timeframe):
                raise ValueError("Timeframe source incompatible avec les métadonnées.")
            df.attrs["timeframe"] = timeframe
        if source_rows is None:
            source_rows = len(df)
        df = df.sort_index(kind="stable")
        validate_ohlcv(df)

    if source_type == "ticks" and timeframe not in (None, "M1"):
        raise ValueError("Le chargement tick produit M1 ; timeframe source incompatible.")
    if source_type == "ticks" and (timestamp_convention is not None or source_timezone != "UTC"):
        raise ValueError("Les options de convention/fuseau source concernent les barres, pas les ticks.")
    df.attrs["source_type"] = source_type
    df.attrs["source_rows"] = source_rows
    df.attrs["price_side"] = prix if source_type == "ticks" else df.attrs.get("price_side")

    if asset is None:
        inferred = df.attrs.get("asset") or infer_asset_from_url(url)
        if not inferred:
            raise ValueError(
                "Impossible d'inférer l'actif : fournir --asset."
            )
        asset = inferred

    if df.attrs.get("asset") not in (None, asset):
        raise ValueError("Actif déclaré incompatible avec les métadonnées source.")
    df.attrs["asset"] = asset
    df.attrs.setdefault("source", str(url))
    df.attrs.setdefault("calendar", "continuous_utc_v1")
    return df, asset
