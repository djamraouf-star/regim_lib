"""
Méthodes de volatilité (ATR, GARCH, régimes de variance).

Ce module importe explicitement chaque méthode concrète afin de
déclencher le décorateur @register_method et de peupler
METHOD_REGISTRY.
"""

from regime_lib.methods.volatility import atr_volatility  # noqa: F401
