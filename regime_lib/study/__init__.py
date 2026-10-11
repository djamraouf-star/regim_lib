"""Outils d'analyse de la relation entre features et cibles futures.

Le module fournit le pipeline ``Study``, ainsi que des fonctions
indépendantes pour calculer les cibles, extraire les features, découper
les données et évaluer les résultats. Les méthodes offline qui utilisent
du lookahead sont exclues par défaut. Voir ``doc/study.md`` pour le guide
d'utilisation.
"""

from regime_lib.study.targets import (
    calculer_cible,
    parser_cible,
    liste_cibles,
)
from regime_lib.study.features import (
    extraire_features,
    lister_features_disponibles,
)
from regime_lib.study.split import (
    split_holdout,
    split_walk_forward,
    split_in_sample,
)
from regime_lib.study.evaluation import (
    ic_spearman,
    hit_rate_directionnel,
    stats_conditionnelles,
)
from regime_lib.study.study import Study

__all__ = [
    "Study",
    "InferenceConfig",
    "calculer_cible",
    "extraire_features",
    "hit_rate_directionnel",
    "ic_spearman",
    "liste_cibles",
    "lister_features_disponibles",
    "parser_cible",
    "split_holdout",
    "split_in_sample",
    "split_walk_forward",
    "stats_conditionnelles",
]

from regime_lib.study.inference import InferenceConfig
