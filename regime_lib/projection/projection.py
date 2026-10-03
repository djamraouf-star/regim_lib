"""
Projection des méthodes dans l'espace des segments.

Principe
--------
Pour chaque barre, on associe :
  - son `segment_id` (issu de la segmentation du prix) ;
  - les features de son segment (durée, pente, efficience, etc.).

Puis on croise avec les régimes produits par les méthodes. Cela permet
de mesurer si les méthodes détectent la même structure que celle
identifiée par la segmentation.

Cette projection est **descriptive** : elle n'établit pas qu'une méthode
est « meilleure » qu'une autre. Elle montre leur **degré d'accord** avec
la partition objective du prix.

Lookahead
---------
La projection hérite du lookahead de la segmentation (offline). Elle
est destinée à l'analyse post-mortem et au criblage offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def projeter_barres(
    df: pd.DataFrame,
    segment_id: pd.Series,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Associe à chaque barre son segment et les features du segment.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV indexé par DatetimeIndex.
    segment_id : pd.Series
        Segment par barre (résultat de `segmenter_dataframe`).
    features : pd.DataFrame
        Features par segment (résultat de `caracteriser_segments`),
        indexé par `segment_id`.

    Returns
    -------
    pd.DataFrame
        DataFrame avec l'index de `df`, colonnes :
        segment_id + toutes les colonnes de features (sauf debut/fin).
    """
    if len(df) != len(segment_id):
        raise ValueError("df et segment_id doivent avoir la même longueur.")

    seg_series = pd.Series(
        segment_id.values, index=df.index, name="segment_id"
    )

    # Colonnes de features à propager (exclut debut/fin qui sont des
    # timestamps de segment, pas des attributs par barre).
    cols_features = [
        c for c in features.columns if c not in ("debut", "fin")
    ]
    feats_par_barre = features[cols_features].reindex(
        seg_series.values
    )
    feats_par_barre.index = df.index

    return pd.concat([seg_series, feats_par_barre], axis=1)


def croiser_methodes_segments(
    projections: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Croise les régimes de plusieurs méthodes avec les features de segment.

    Parameters
    ----------
    projections : dict[str, pd.DataFrame]
        Dictionnaire {nom_methode: DataFrame} où chaque DataFrame est
        indexé par timestamp et contient au moins une colonne `regime`.
        Il doit aussi contenir les colonnes de features communes,
        identiques entre méthodes (elles viennent de la même segmentation).

    Returns
    -------
    pd.DataFrame
        DataFrame indexé par timestamp avec, pour chaque méthode, une
        colonne `<methode>_regime` et une colonne `<methode>_confidence`,
        plus les features de segment une seule fois.
    """
    if not projections:
        raise ValueError("Aucune projection fournie.")

    # Prendre la première pour les features de segment.
    ref_name = next(iter(projections))
    ref = projections[ref_name]

    cols_features = [
        c for c in ref.columns
        if c not in ("regime", "confidence")
    ]
    result = ref[cols_features].copy()

    for name, proj in projections.items():
        result[f"{name}_regime"] = proj["regime"]
        if "confidence" in proj.columns:
            result[f"{name}_confidence"] = proj["confidence"]

    return result
