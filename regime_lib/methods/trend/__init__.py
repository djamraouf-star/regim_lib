"""
Méthodes de tendance (ex. ADX, pentes, moyennes mobiles).

Ce module importe explicitement chaque méthode concrète afin de déclencher
le décorateur @register_method et de peupler METHOD_REGISTRY.

Pour ajouter une méthode de tendance :
  1. créer `regime_lib/methods/trend/ma_methode.py` avec une classe
     héritant de RegimeDetector et décorée @register_method ;
  2. ajouter l'import correspondant ci-dessous.
"""

from regime_lib.methods.trend import er_kaufman  # noqa: F401

# À venir
from regime_lib.methods.trend import adx  # noqa: F401
# from regime_lib.methods.trend import slope  # noqa: F401
