"""
Méthodes basées sur le volume.

Ces méthodes exploitent les colonnes de volume issues de l'agrégation
tick : `volume`, `bid_volume`, `ask_volume`, `tick_count`.

⚠ Fiabilité des données
-----------------------
Le volume Dukascopy est une mesure de **profondeur observée**, pas de
volume exécuté. Sur forex (marché décentralisé), il représente une
fraction de l'activité mondiale. Voir `doc/volume.md`.

Ce module importe explicitement chaque méthode concrète.
"""

from regime_lib.methods.volume import divergence_pv   # noqa: F401
from regime_lib.methods.volume import ofi             # noqa: F401
from regime_lib.methods.volume import volume_profile  # noqa: F401
from regime_lib.methods.volume import vpin            # noqa: F401
