"""
Méthodes basées sur le prix brut (ex. patterns, niveaux, price action).

Ce module importe explicitement chaque méthode concrète afin de déclencher
le décorateur @register_method et de peupler METHOD_REGISTRY.

Pour ajouter une méthode de price action :
  1. créer `regime_lib/methods/price/ma_methode.py` avec une classe
     héritant de RegimeDetector et décorée @register_method ;
  2. ajouter l'import correspondant ci-dessous.

⚠ Attention au lookahead
-----------------------
Les méthodes de price action utilisant des patterns (fractales, figures
chartistes, pivots) ont souvent un lookahead **structurel** : un sommet
n'est confirmé qu'après avoir vu les barres suivantes. Ces méthodes
doivent définir `requires_lookahead = True` et sont destinées à l'analyse
a posteriori, pas au signal tradable en temps réel.
"""

from regime_lib.methods.price import price_action  # noqa: F401
from regime_lib.methods.price import mss_choch  # noqa: F401

# À venir
# from regime_lib.methods.price import support_resistance  # noqa: F401

from .price_action_causal import PriceActionCausalDetector
