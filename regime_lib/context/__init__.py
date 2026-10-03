"""
Contexte temporel : sessions, calendrier, événements.

Ce module enrichit les barres avec des attributs contextuels calculés
à partir du timestamp seul :
  - session de marché (ASIE, LONDRES, OVERLAP, NY) ;
  - heure locale et UTC ;
  - jour de semaine, fériés, rollover ;
  - fenêtres d'événements macro.

Toutes les heures sont exprimées dans le fuseau `reference_tz` du
profil (défaut : America/New_York).

Voir `doc/context.md` pour la sémantique complète.
"""

from regime_lib.context.sessions import (
    calculer_sessions,
    enrichir_sessions,
)
from regime_lib.context.calendar import (
    calculer_calendrier,
    charger_holidays,
    enrichir_calendrier,
)
from regime_lib.context.events import (
    calculer_events,
    charger_events,
    enrichir_events,
)
from regime_lib.context.enrichment import enrichir_contexte

__all__ = [
    "calculer_calendrier",
    "calculer_events",
    "calculer_sessions",
    "charger_events",
    "charger_holidays",
    "enrichir_calendrier",
    "enrichir_contexte",
    "enrichir_events",
    "enrichir_sessions",
]
