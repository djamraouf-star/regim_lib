"""
Format de sortie uniforme et helpers d'export.

Contrat
-------
Chaque méthode produit un DataFrame au format uniforme :
  - REQUIRED_COLUMNS en préfixe, dans cet ordre ;
  - OPTIONAL_COLUMNS autorisées après, dans n'importe quel ordre.

L'index `timestamp` est matérialisé en colonne à la sortie
(reset_index systématique), l'index de sortie est un RangeIndex.

Note sur le consensus
---------------------
Un ancien utilitaire `add_consensus` a été retiré. Il matérialisait un
vote de labels entre méthodes, ce qui est conceptuellement faux : les
labels ne sont PAS comparables entre méthodes (le RANGE d'ATR n'est pas
le RANGE d'ADX). Pour agréger plusieurs méthodes, utiliser directement
pandas :

    pivot = df.pivot(index="timestamp", columns="method", values="regime")
    barres_alignees = pivot[pivot.eq(pivot.iloc[:, 0], axis=0).all(axis=1)]

Voir `doc/archives/consensus.md` pour l'historique de cette décision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd
from regime_lib.utils.validation import validate_time_index

REQUIRED_COLUMNS: Final[list[str]] = [
    "timestamp",
    "regime",
    "regime_id",
    "confidence",
    "method",
    "timeframe",
    "asset",
]

OPTIONAL_COLUMNS: Final[list[str]] = [
    "is_partial",
    "params_hash",
]


def to_uniform(
    df: pd.DataFrame,
    *,
    method: str,
    timeframe: str,
    asset: str,
    regime_map: dict[str, int] | None = None,
    params_hash: str | None = None,
) -> pd.DataFrame:
    """
    Normalise un DataFrame de régimes au format uniforme.

    Parameters
    ----------
    df : pd.DataFrame
        Index DatetimeIndex (timestamp). Doit contenir au moins `regime`.
    method : str
        Nom de la méthode (colonne `method`).
    timeframe : str
        Timeframe utilisé (colonne `timeframe`).
    asset : str
        Symbole de l'actif (colonne `asset`).
    regime_map : dict[str, int] | None
        Correspondance label -> id. Requis si `regime` est en labels str
        et que `regime_id` n'est pas déjà présent.
    params_hash : str | None
        Hash court des hyperparamètres. Optionnel.

    Returns
    -------
    pd.DataFrame
        Colonnes REQUIRED_COLUMNS en préfixe, OPTIONAL_COLUMNS après.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(
            "to_uniform attend un DatetimeIndex nommé 'timestamp' en entrée."
        )
    if "regime" not in df.columns:
        raise ValueError("Colonne 'regime' manquante.")

    out = df.copy()
    validate_time_index(out.index)
    for field in ("source", "price_side", "adjustment", "calendar", "timestamp_convention"):
        if out.attrs.get(field) is not None:
            out[field] = out.attrs[field]
    out.index.name = "timestamp"
    out = out.reset_index()

    # regime_id
    if "regime_id" not in out.columns:
        if regime_map is None:
            raise ValueError(
                "regime_id absent et regime_map non fourni : impossible de "
                "dériver l'identifiant numérique."
            )
        out["regime_id"] = out["regime"].map(regime_map)
        if out["regime_id"].isna().any():
            bad = out.loc[out["regime_id"].isna(), "regime"].unique().tolist()
            raise ValueError(
                f"Labels de régime inconnus dans regime_map : {bad}."
            )
        out["regime_id"] = out["regime_id"].astype(int)

    # confidence
    if "confidence" not in out.columns:
        out["confidence"] = 1.0
    if out["confidence"].isna().any():
        raise ValueError("La colonne 'confidence' ne peut pas contenir de NaN.")
    if not out["confidence"].between(0.0, 1.0).all():
        raise ValueError("La colonne 'confidence' doit être dans [0, 1].")

    # Colonnes constantes
    out["method"] = method
    out["timeframe"] = timeframe
    out["asset"] = asset

    # params_hash optionnel
    if params_hash is not None:
        out["params_hash"] = params_hash

    # Ordre final : REQUIRED en préfixe, puis OPTIONAL présentes.
    ordered = REQUIRED_COLUMNS + [
        c for c in OPTIONAL_COLUMNS if c in out.columns
    ]
    extra = [c for c in out.columns if c not in ordered]
    out = out[ordered + extra]

    return out


def export_long(
    frames: list[pd.DataFrame],
    output_dir: str | Path,
    filename: str = "regimes.parquet",
) -> Path:
    """
    Empile une liste de DataFrames uniformes et écrit un parquet long.

    Returns
    -------
    Path
        Chemin du fichier écrit.
    """
    if not frames:
        raise ValueError("Aucun DataFrame à exporter.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_df = pd.concat(frames, ignore_index=True)
    path = output_dir / filename
    long_df.to_parquet(path, index=False)
    return path


def export_meta(
    meta: dict,
    output_dir: str | Path,
    filename: str = "run_meta.json",
) -> Path:
    """Écrit un sidecar JSON de métadonnées de run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
    return path
