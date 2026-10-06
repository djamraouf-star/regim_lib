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

from regime_lib.study.validation import prepare_regimes, series_identity


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
        "calendar", "timestamp_convention", "availability", "available_at",
        "regime_dimension", "regime_description", "regime_scale", "revises_history",
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
        from regime_lib.core.registry import METHOD_REGISTRY
        retained = []
        for name in methodes:
            rows = df[df["method"].eq(name)]
            if rows.empty:
                continue
            cls = METHOD_REGISTRY.get(name)
            availability = getattr(cls, "legacy_availability", cls.availability) if cls is not None else "unknown"
            if "availability" in rows:
                declared = rows["availability"]
                if declared.isna().any() or declared.nunique() != 1:
                    raise ValueError(f"Disponibilité ambiguë pour {name}.")
                availability = declared.iloc[0]
                if cls is not None and cls.availability != "bar_close" and availability == "bar_close":
                    raise ValueError(f"Disponibilité incompatible avec le détecteur {name}.")
            if availability == "bar_close":
                if "available_at" in rows:
                    published = pd.to_datetime(rows["available_at"], utc=True)
                    timestamps = pd.to_datetime(rows["timestamp"], utc=True)
                    if published.isna().any() or (published > timestamps).any():
                        raise ValueError(f"Feature indisponible à date pour {name}.")
                retained.append(name)
        methodes = retained

    manquantes = [f for f in features if f not in df.columns]
    if manquantes:
        raise ValueError(
            f"Features absentes du parquet : {manquantes}. "
            f"Disponibles : {sorted(df.columns)}."
        )

    df = df[df["method"].isin(methodes)]
    if df.empty:
        raise ValueError("Aucune méthode retenue après filtrage.")
    # Pivot : index = timestamp, colonnes = (method, feature) → nom
    # combiné. Une seule méthode par ligne à l'origine, on pivote
    # pour avoir une colonne par (methode, feature).
    pieces = []
    exclusions = {}
    identities = []
    for methode in methodes:
        sub = df[df["method"] == methode].copy()
        sub, valid = prepare_regimes(
            sub, (configurations or {}).get(methode),
        )
        exclusions[methode] = sub.attrs["exclusions"]
        identities.append(sub.attrs["series_identity"])
        sub = sub[features].copy()
        for col in features:
            if pd.api.types.is_bool_dtype(sub[col]):
                sub[col] = sub[col].astype("boolean")
            if pd.api.types.is_numeric_dtype(sub[col]):
                sub[col] = sub[col].where(np.isfinite(sub[col]))
            sub[col] = sub[col].where(valid)
        sub = sub.add_prefix(f"{methode}__")
        pieces.append(sub)

    if not pieces:
        raise ValueError("Aucune méthode retenue après filtrage.")

    out = pd.concat(pieces, axis=1)
    out = out.sort_index()
    out.attrs["series_identity"] = series_identity(pd.DataFrame(identities))
    out.attrs["exclusions"] = exclusions
    from regime_lib.core.registry import METHOD_REGISTRY
    out.attrs["interpretations"] = {
        name: METHOD_REGISTRY[name].interpretation()
        for name in methodes if name in METHOD_REGISTRY
    }
    return out
