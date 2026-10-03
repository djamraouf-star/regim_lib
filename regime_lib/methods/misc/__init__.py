"""
Méthodes diverses (Minkowski, analyse causale, expérimentations).

Ce module importe explicitement chaque méthode concrète afin de
déclencher le décorateur @register_method et de peupler METHOD_REGISTRY.
"""

from regime_lib.methods.misc import minkowski_causal  # noqa: F401
