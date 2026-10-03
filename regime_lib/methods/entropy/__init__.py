"""
Famille entropie — méthodes basées sur le désordre / la complexité.

Ces méthodes produisent une mesure continue (entropie) qui quantifie
le désordre dans une fenêtre de rendements :

  - Entropie élevée → marché chaotique, imprévisible
  - Entropie faible → structure, ordre, potentiellement prévisible

Un label est dérivé de la mesure pour compatibilité avec le contrat
`RegimeDetector` (ORDONNE / NEUTRE / DESORDRE).
"""

from regime_lib.methods.entropy import shannon  # noqa: F401
