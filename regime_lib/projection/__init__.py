"""
Projection des méthodes dans l'espace du prix.

Objectif
--------
Le prix est la seule chose que toutes les méthodes partagent. En
segmentant le prix en phases homogènes (détection de points de
changement), on obtient une **référence de projection** : une partition
objective du temps, indépendante de toute méthode.

Chaque méthode peut ensuite être **projetée** sur ces segments pour
mesurer sa cohérence avec la structure réelle du prix.

Attention — la segmentation n'est PAS une vérité terrain
--------------------------------------------------------
La segmentation décide d'un découpage selon ses paramètres (pénalité,
modèle de coût, taille minimale). Elle ne « découvre » pas les régimes
réels — elle propose une lecture parmi d'autres. Elle sert de référence
commune pour comparer les méthodes, pas de juge de vérité.

Lookahead
---------
PELT utilise toute la série pour détecter les points de changement.
C'est donc une méthode **offline par construction**, comme le HMM.

Modules
-------
- `segmentation` : PELT sur le log-prix → segments
- `features`     : caractérisation des segments (à venir)
- `projection`   : projection des méthodes (à venir)
- `agreement`    : mesure d'accord méthode × segment (à venir)
"""

"""
Projection des méthodes dans l'espace du prix.

... (docstring existante) ...
"""

from regime_lib.projection.segmentation import (
    segmenter_dataframe,
    segmenter_pelt,
)
from regime_lib.projection.features import caracteriser_segments
from regime_lib.projection.projection import (
    croiser_methodes_segments,
    projeter_barres,
)
from regime_lib.projection.agreement import (
    comparer_methodes,
    nombre_moyen_regimes_par_segment,
    purete_globale_ponderee,
    purete_par_segment,
)

__all__ = [
    "caracteriser_segments",
    "comparer_methodes",
    "croiser_methodes_segments",
    "nombre_moyen_regimes_par_segment",
    "projeter_barres",
    "purete_globale_ponderee",
    "purete_par_segment",
    "segmenter_dataframe",
    "segmenter_pelt",
]
