"""
Schéma de configuration : structures de données validées.

Les dataclasses définissent la forme **générale** d'un profil. Les
paramètres spécifiques à chaque méthode restent libres (dict) : chaque
méthode valide les siens dans son constructeur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers de validation
# ---------------------------------------------------------------------------
def _valider_heure(valeur: object, chemin: str) -> None:
    """Vérifie qu'une valeur est au format 'HH:MM'."""
    if not isinstance(valeur, str):
        raise ValueError(
            f"context.{chemin} doit être une chaîne 'HH:MM', "
            f"reçu {valeur!r}."
        )
    parts = valeur.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"context.{chemin} : format invalide {valeur!r} "
            f"(attendu 'HH:MM')."
        )
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"context.{chemin} : {valeur!r} n'est pas 'HH:MM'."
        )
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(
            f"context.{chemin} : {valeur!r} hors bornes."
        )


# ---------------------------------------------------------------------------
# Sous-structures : données
# ---------------------------------------------------------------------------
@dataclass
class TickAggregationConfig:
    """
    Configuration déclarative de l'agrégation tick → barre.

    Le pipeline actuel détecte les colonnes tick automatiquement et utilise
    une agrégation temporelle Ask. Les autres modes de cette configuration
    ne sont pas encore consommés par le CLI.
    """
    mode: str = "time"                              # time; autres modes réservés
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DonneesConfig:
    """Configuration des données source et des timeframes cibles."""
    source: str = "barres"                          # 'barres' ou 'tick'
    timeframes: list[str] = field(default_factory=lambda: ["M1"])
    tick_aggregation: TickAggregationConfig | None = None

    def __post_init__(self) -> None:
        if self.source not in ("barres", "tick"):
            raise ValueError(
                f"source invalide : {self.source!r}. "
                f"Attendu : 'barres' ou 'tick'."
            )
        if not self.timeframes:
            raise ValueError("donnees.timeframes ne peut pas être vide.")


# ---------------------------------------------------------------------------
# Sous-structures : contexte temporel
# ---------------------------------------------------------------------------
@dataclass
class ContextConfig:
    """
    Configuration du contexte temporel.

    Toutes les heures (sessions, rollover) sont exprimées dans le fuseau
    `reference_tz`. Les événements macro et jours fériés sont lus depuis
    des fichiers CSV (chemins relatifs à la racine du projet).

    Voir `doc/context.md` pour la sémantique complète.

    Attributes
    ----------
    reference_tz : str
        Fuseau de référence (nom IANA). Défaut : 'America/New_York'.
    sessions : dict[str, dict[str, str]]
        Sessions de marché. Clés = noms libres, valeurs = {debut, fin}
        au format 'HH:MM' dans `reference_tz`.
    rollover : dict[str, str] | None
        Fenêtre de rollover forex, ou None si absente.
    events : dict
        Configuration des événements macro. Structure attendue :
        {fichier: str | None, fenetres: {NOM: {avant_min, apres_min}}}.
    holidays : dict
        Configuration des jours fériés. Structure attendue :
        {fichier: str | None}.
    """
    reference_tz: str = "America/New_York"
    sessions: dict[str, dict[str, str]] = field(default_factory=dict)
    rollover: dict[str, str] | None = None
    events: dict = field(default_factory=dict)
    holidays: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 1. Fuseau valide
        from zoneinfo import ZoneInfo
        try:
            ZoneInfo(self.reference_tz)
        except Exception as exc:
            raise ValueError(
                f"context.reference_tz invalide : {self.reference_tz!r} "
                f"({exc})."
            )

        # 2. Sessions : chaque entrée a {debut, fin} au format HH:MM
        if not isinstance(self.sessions, dict):
            raise ValueError("context.sessions doit être un dict.")
        for nom, bornes in self.sessions.items():
            if not isinstance(bornes, dict):
                raise ValueError(
                    f"context.sessions[{nom!r}] doit être un dict "
                    f"{{debut, fin}}."
                )
            for cle in ("debut", "fin"):
                if cle not in bornes:
                    raise ValueError(
                        f"context.sessions[{nom!r}] : clé {cle!r} manquante."
                    )
                _valider_heure(bornes[cle], f"sessions.{nom}.{cle}")

        # 3. Rollover : idem, ou None
        if self.rollover is not None:
            if not isinstance(self.rollover, dict):
                raise ValueError(
                    "context.rollover doit être un dict ou null."
                )
            for cle in ("debut", "fin"):
                if cle not in self.rollover:
                    raise ValueError(
                        f"context.rollover : clé {cle!r} manquante."
                    )
                _valider_heure(self.rollover[cle], f"rollover.{cle}")

        # 4. Events : fenêtres avec avant_min / apres_min entiers >= 0
        if not isinstance(self.events, dict):
            raise ValueError("context.events doit être un dict.")
        fenetres = self.events.get("fenetres", {})
        if not isinstance(fenetres, dict):
            raise ValueError("context.events.fenetres doit être un dict.")
        for nom, f in fenetres.items():
            if not isinstance(f, dict):
                raise ValueError(
                    f"context.events.fenetres[{nom!r}] doit être un dict."
                )
            for cle in ("avant_min", "apres_min"):
                if cle not in f:
                    raise ValueError(
                        f"context.events.fenetres[{nom!r}] : "
                        f"clé {cle!r} manquante."
                    )
                if isinstance(f[cle], bool) or not isinstance(f[cle], int):
                    raise ValueError(
                        f"context.events.fenetres[{nom!r}].{cle} doit "
                        f"être un entier >= 0."
                    )
                if f[cle] < 0:
                    raise ValueError(
                        f"context.events.fenetres[{nom!r}].{cle} doit "
                        f"être >= 0."
                    )

        # 5. Holidays : structure attendue {fichier}
        if not isinstance(self.holidays, dict):
            raise ValueError("context.holidays doit être un dict.")


# ---------------------------------------------------------------------------
# Sous-structures : projection
# ---------------------------------------------------------------------------
@dataclass
class SegmentationConfig:
    """
    Configuration du module de projection (segmentation PELT).

    Voir `projection/segmentation.py` (phase 6).
    """
    modele: str = "normal"                          # 'l2', 'normal', 'rbf'
    pen: float = 3.0
    min_size: int = 20


# ---------------------------------------------------------------------------
# Profil complet
# ---------------------------------------------------------------------------
@dataclass
class ProfilConfig:
    """
    Profil complet d'un actif.

    Attributes
    ----------
    nom : str
        Identifiant court ('eurusd', 'btcusd', 'default').
    type : str
        Catégorie ('forex_majeur', 'crypto', 'indice', 'default').
    description : str
        Texte libre, à usage documentaire.
    donnees : DonneesConfig
        Configuration de la source et des timeframes.
    methodes : dict[str, dict]
        Paramètres par méthode. Structure libre : chaque méthode lit
        ce dont elle a besoin et ignore le reste.
    context : ContextConfig
        Configuration du contexte temporel (sessions, événements).
    segmentation : SegmentationConfig
        Configuration de la projection.
    source_path : Path | None
        Chemin du fichier YAML d'origine (traçabilité). None si
        construit programmatiquement.
    """
    nom: str
    type: str = "default"
    description: str = ""
    donnees: DonneesConfig = field(default_factory=DonneesConfig)
    methodes: dict[str, dict] = field(default_factory=dict)
    context: ContextConfig = field(default_factory=ContextConfig)
    segmentation: SegmentationConfig = field(
        default_factory=SegmentationConfig
    )
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.nom:
            raise ValueError("ProfilConfig.nom ne peut pas être vide.")
        if not isinstance(self.methodes, dict):
            raise ValueError(
                "ProfilConfig.methodes doit être un dict "
                "{nom_methode: params}."
            )
        for nom, params in self.methodes.items():
            if not isinstance(nom, str):
                raise ValueError(
                    f"Nom de méthode invalide : {nom!r} (doit être str)."
                )
            if not isinstance(params, dict):
                raise ValueError(
                    f"Params de {nom!r} invalides : doit être un dict."
                )
        if not isinstance(self.context, ContextConfig):
            raise ValueError(
                "ProfilConfig.context doit être un ContextConfig."
            )
