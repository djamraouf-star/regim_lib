"""
Mesure d'accord entre méthodes et segments.

Objectif
--------
Pour chaque méthode, on mesure dans quelle mesure ses régimes
correspondent à la structure du prix identifiée par la segmentation.

Trois indicateurs :

1. **Pureté par segment** : pour chaque segment, quelle fraction des
   barres partage le régime majoritaire ? Un segment avec pureté 1.0
   signifie que la méthode lui attribue un seul régime — signe de
   cohérence.

2. **Pureté moyenne pondérée** : moyenne des puretés, pondérée par la
   durée des segments. Mesure globale de cohérence.

3. **Nombre de régimes par segment** : combien de régimes différents
   apparaissent dans un segment. Plus ce nombre est bas, plus la
   méthode est cohérente avec la segmentation.

Ces mesures ne disent PAS qu'une méthode est « correcte ». Elles
mesurent sa **cohérence interne** avec la partition du prix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from regime_lib.utils.validation import validate_alignment


def purete_par_segment(
    regimes: pd.Series,
    segment_id: pd.Series,
) -> pd.DataFrame:
    """
    Pureté des régimes par segment.

    Parameters
    ----------
    regimes : pd.Series
        Régime par barre (résultat d'une méthode).
    segment_id : pd.Series
        Segment par barre (résultat de `segmenter_dataframe`).

    Returns
    -------
    pd.DataFrame
        Indexé par `segment_id`, colonnes :
        - n_barres : taille du segment
        - regime_dominant : régime le plus fréquent
        - purete : fraction des barres dans le régime dominant
        - n_regimes : nombre de régimes distincts dans le segment
        - regimes : liste des régimes présents (utile pour debug)
    """
    validate_alignment(regimes.index, segment_id.index)
    if segment_id.isna().any():
        raise ValueError("Identifiants de segment manquants.")
    tmp = pd.DataFrame({"regime": regimes, "segment_id": segment_id})
    tmp = tmp[tmp["regime"].notna() & tmp["regime"].ne("INCONNU")]

    rows: list[dict] = []
    for sid, g in tmp.groupby("segment_id", sort=True):
        counts = g["regime"].value_counts()
        total = len(g)
        rows.append({
            "segment_id": int(sid),
            "n_barres": total,
            "regime_dominant": counts.index[0],
            "purete": float(counts.iloc[0] / total),
            "n_regimes": int(len(counts)),
            "regimes": counts.to_dict(),
        })

    return pd.DataFrame(rows, columns=["segment_id", "n_barres", "regime_dominant",
                                       "purete", "n_regimes", "regimes"]).set_index("segment_id")


def purete_globale_ponderee(
    purete_df: pd.DataFrame,
) -> float:
    """
    Pureté moyenne pondérée par la taille des segments.
    """
    if len(purete_df) == 0:
        return float("nan")
    poids = purete_df["n_barres"]
    return float(
        (purete_df["purete"] * poids).sum() / poids.sum()
    )


def nombre_moyen_regimes_par_segment(
    purete_df: pd.DataFrame,
) -> float:
    """Nombre moyen de régimes distincts par segment."""
    if len(purete_df) == 0:
        return float("nan")
    return float(purete_df["n_regimes"].mean())


def comparer_methodes(
    methodes: dict[str, pd.Series],
    segment_id: pd.Series,
) -> pd.DataFrame:
    """
    Tableau comparatif des méthodes.

    Parameters
    ----------
    methodes : dict[str, pd.Series]
        {nom_methode: régimes par barre}.
    segment_id : pd.Series
        Segment par barre.

    Returns
    -------
    pd.DataFrame
        Une ligne par méthode avec :
        - purete_ponderee : pureté moyenne pondérée ∈ [0, 1]
        - n_regimes_moy : nombre moyen de régimes par segment
        - purete_min : pureté du segment le moins pur
        - purete_max : pureté du segment le plus pur
        - n_segments_purs : nombre de segments à pureté = 1.0
    """
    shared = pd.Series(True, index=segment_id.index)
    for regimes in methodes.values():
        validate_alignment(regimes.index, segment_id.index)
        shared &= regimes.notna() & regimes.ne("INCONNU")
    rows: list[dict] = []
    for name, regimes in methodes.items():
        own = regimes.notna() & regimes.ne("INCONNU")
        p = purete_par_segment(regimes.where(shared), segment_id)
        rows.append({
            "methode": name,
            "n_propre": int(own.sum()),
            "n_commun": int(shared.sum()),
            "purete_ponderee": purete_globale_ponderee(p),
            "n_regimes_moy": nombre_moyen_regimes_par_segment(p),
            "purete_min": float(p["purete"].min()) if len(p) else float("nan"),
            "purete_max": float(p["purete"].max()) if len(p) else float("nan"),
            "n_segments_purs": int((p["purete"] == 1.0).sum()),
        })
    return pd.DataFrame(rows, columns=["methode", "n_propre", "n_commun", "purete_ponderee",
                                       "n_regimes_moy", "purete_min", "purete_max",
                                       "n_segments_purs"]).set_index("methode")
