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
        "params_hash",
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

    # Pivot : index = timestamp, colonnes = (method, feature) → nom
    # combiné. Une seule méthode par ligne à l'origine, on pivote
    # pour avoir une colonne par (methode, feature).
    pieces = []
    for methode in methodes:
        sub = df[df["method"] == methode].set_index("timestamp")
        sub = sub[features].add_prefix(f"{methode}__")
        pieces.append(sub)

    if not pieces:
        raise ValueError("Aucune méthode retenue après filtrage.")

    out = pd.concat(pieces, axis=1)
    out = out.sort_index()
    return out
