"""
Registre des méthodes disponibles, découplé du CLI.

Ce module est volontairement neutre : il ne dépend ni de `cli` ni de
`methods`, ce qui casse le cycle d'import
qui apparaîtrait si `register_method` vivait dans `cli.py`.

.. code-block:: text

    cli -> methods -> cli
"""

from __future__ import annotations

from regime_lib.core.base import RegimeDetector


METHOD_REGISTRY: dict[str, type[RegimeDetector]] = {}


def register_method(cls: type[RegimeDetector]) -> type[RegimeDetector]:
    """Décorateur d'enregistrement d'une méthode dans le registre global."""
    if not isinstance(cls, type) or not issubclass(cls, RegimeDetector):
        raise TypeError(
            f"register_method attend une sous-classe de RegimeDetector, "
            f"reçu : {cls!r}."
        )
    if cls.name in METHOD_REGISTRY:
        raise ValueError(f"Méthode déjà enregistrée : {cls.name!r}.")
    METHOD_REGISTRY[cls.name] = cls
    return cls
