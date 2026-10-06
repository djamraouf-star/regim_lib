"""
Gestion de la configuration par profil.

Un **profil** est un fichier YAML qui contient tous les paramètres
d'un actif : timeframes disponibles, paramètres des méthodes, options
de segmentation, etc.

Objectif : rendre la lib **portable entre actifs** sans toucher au
code. Changer d'actif = changer de profil.

Structure d'un profil
---------------------
    nom: eurusd
    type: forex_majeur
    description: "EURUSD, calibration 2025-2026"
    
    donnees:
      source: barres
      timeframes: [M1, M5, M15, H1]
    
    methodes:
      atr_volatility:
        timeframes:
          M1: {n_atr: 20, fenetre: 30000, min_periods: 15000}
          ...
      adx:
        n_adx: 14
        seuil_range: 20.0
        seuil_tendance: 25.0
    
    segmentation:
      modele: normal
      pen: 3.0
      min_size: 20

Héritage
--------
Les défauts YAML et les surcharges restent séparés jusqu'au résolveur
commun. Les paramètres effectifs et leur provenance sont disponibles
sur le détecteur. Voir doc/config.md pour les priorités et la migration.

Usage
-----
    from regime_lib.config import load_profile
    from regime_lib.methods.trend.adx import ADXDetector

    profil = load_profile("eurusd")
    detecteur = ADXDetector(timeframe="H1", profile=profil.for_method("adx"))
    params = detecteur.params

"""

from regime_lib.config.schema import DonneesConfig, ProfilConfig
from regime_lib.config.loader import (
    load_profile,
    list_profiles,
    profile_hash,
)

__all__ = [
    "DonneesConfig",
    "ProfilConfig",
    "load_profile",
    "list_profiles",
    "profile_hash",
]
