"""
Extraction des features depuis un parquet de régimes.

Une feature est une colonne du parquet long, considérée comme
susceptible d'apporter de l'information sur une cible.

Catégories de features
----------------------
- `regime` : label du régime produit par la méthode
- `confidence` : confiance de la classification
- `ctx_*` : contexte temporel (session, rollover, fériés, événements)

Le module fournit :
  - `lister_features_disponibles` : inventaire
  - `extraire_features` : sélection et pivot d'un parquet long
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from regime_lib.utils.validation import validate_time_index, validity_mask

SERIES_FIELDS = ("asset", "timeframe", "source", "price_side", "adjustment", "calendar")


FEATURES_CATEGORIELLES = {
    "regime",
    "ctx_session",
    "ctx_event_type",
}

FEATURES_NUMERIQUES = {
    "confidence",
    "ctx_hour_ny",
    "ctx_hour_utc",
    "ctx_day_of_week",
}

FEATURES_BOOLEENNES = {
    "ctx_is_rollover",
    "ctx_is_holiday",
    "ctx_in_event",
}


def lister_features_disponibles(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Inventorie les features disponibles dans un parquet long.

    Parameters
    ----------
    df : pd.DataFrame
        Parquet long issu du CLI.

    Returns
    -------
    dict[str, list[str]]
        Clés : 'categorielles', 'numeriques', 'booleennes', 'autres'.
        Chaque valeur est triée.
    """
    colonnes = set(df.columns)

    categorielles = sorted(colonnes & FEATURES_CATEGORIELLES)
    numeriques = sorted(colonnes & FEATURES_NUMERIQUES)
    booleennes = sorted(colonnes & FEATURES_BOOLEENNES)

    connues = (
        FEATURES_CATEGORIELLES
        | FEATURES_NUMERIQUES
        | FEATURES_BOOLEENNES
    )
    metadonnees = {
        "timestamp", "regime_id", "method", "timeframe", "asset",
        "params_hash", "is_partial", "is_valid", "is_warmup", "coverage",
        "source_count", "expected_count", "source", "price_side", "adjustment",
        "calendar", "timestamp_convention",
    }
    autres = sorted(
        c for c in colonnes
        if c not in connues
        and c not in metadonnees
        and c != "regime"
    )

    return {
        "categorielles": categorielles,
        "numeriques": numeriques,
        "booleennes": booleennes,
        "autres": autres,
    }


def extraire_features(
    df_regimes: pd.DataFrame,
    methodes: list[str] | None = None,
    features: list[str] | None = None,
    methodes_causales_only: bool = True,
    *,
    asset: str | None = None,
    timeframe: str | None = None,
    configurations: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Extrait les features depuis un parquet long.

    Parameters
    ----------
    df_regimes : pd.DataFrame
        Parquet long issu du CLI.
    methodes : list[str] | None
        Méthodes à inclure. None = toutes (sous réserve du filtre
        causal).
    features : list[str] | None
        Features à extraire. None = ['regime'] (la plus courante).
    methodes_causales_only : bool
        Si True, exclut les méthodes offline (lookahead structurel).
        Défaut : True.

    Returns
    -------
    pd.DataFrame
        Indexé par timestamp, une colonne par (méthode, feature),
        nommée `<methode>__<feature>`.
    """
    METHODES_OFFLINE = {"hmm_gaussian", "price_action", "mss_choch"}

    if features is None:
        features = ["regime"]

    df = df_regimes.copy()
    for name, value in (("asset", asset), ("timeframe", timeframe)):
        if name not in df:
            raise ValueError(f"Identité de série absente : {name}.")
        if value is not None:
            df = df[df[name].eq(value)]

    if methodes is None:
        methodes = sorted(df["method"].unique().tolist())

    if methodes_causales_only:
        methodes = [m for m in methodes if m not in METHODES_OFFLINE]

    manquantes = [f for f in features if f not in df.columns]
    if manquantes:
        raise ValueError(
            f"Features absentes du parquet : {manquantes}. "
            f"Disponibles : {sorted(df.columns)}."
        )

    df = df[df["method"].isin(methodes)]
    if df.empty:
        raise ValueError("Aucune méthode retenue après filtrage.")
    identity = {}
    for field in SERIES_FIELDS:
        if field in df:
            if df[field].isna().any() or df[field].nunique() != 1:
                raise ValueError(f"Identité ambiguë ({field}) : sélectionner une série unique.")
            identity[field] = df[field].iloc[0]

    # Pivot : index = timestamp, colonnes = (method, feature) → nom
    # combiné. Une seule méthode par ligne à l'origine, on pivote
    # pour avoir une colonne par (methode, feature).
    pieces = []
    exclusions = {}
    for methode in methodes:
        sub = df[df["method"] == methode].copy()
        if configurations and methode in configurations:
            if "params_hash" not in sub:
                raise ValueError("params_hash absent pour la sélection de configuration.")
            sub = sub[sub["params_hash"].eq(configurations[methode])]
        if sub.empty:
            raise ValueError(f"Méthode/configuration absente : {methode}.")
        if "params_hash" in sub and (sub["params_hash"].isna().any() or
                                    sub["params_hash"].nunique() != 1):
            raise ValueError(f"Configurations ambiguës pour {methode} : sélectionner params_hash.")
        sub = sub.set_index("timestamp").sort_index()
        validate_time_index(sub.index)
        valid = validity_mask(sub)
        exclusions[methode] = {
            "n_partiel": int(sub["is_partial"].fillna(True).sum()) if "is_partial" in sub else 0,
            "n_chauffe": int(sub["is_warmup"].fillna(True).sum()) if "is_warmup" in sub else 0,
            "n_inconnu": int(sub["regime"].eq("INCONNU").sum()) if "regime" in sub else 0,
            "n_absent": int(sub["regime"].isna().sum()) if "regime" in sub else 0,
        }
        sub = sub[features].copy()
        for col in features:
            if pd.api.types.is_numeric_dtype(sub[col]):
                sub[col] = sub[col].where(np.isfinite(sub[col]))
            sub[col] = sub[col].where(valid)
        sub = sub.add_prefix(f"{methode}__")
        pieces.append(sub)

    if not pieces:
        raise ValueError("Aucune méthode retenue après filtrage.")

    out = pd.concat(pieces, axis=1)
    out = out.sort_index()
    out.attrs["series_identity"] = identity
    out.attrs["exclusions"] = exclusions
    return out
