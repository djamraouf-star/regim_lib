"""
Chargement, fusion et hachage des profils.

Le loader est le point d'entrée principal du module config. Il gère :
  - le chargement d'un fichier YAML ;
  - la fusion profonde avec le profil par défaut ;
  - la construction d'un `ProfilConfig` validé ;
  - le calcul d'un hash stable pour la reproductibilité.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from regime_lib.config.schema import (
    ContextConfig,
    DonneesConfig,
    ProfilConfig,
    SegmentationConfig,
    TickAggregationConfig,
)


# Répertoire des profils embarqués.
_PROFILES_DIR = Path(__file__).parent / "profiles"


# ---------------------------------------------------------------------------
# Fusion profonde
# ---------------------------------------------------------------------------
def _deep_merge(base: dict, override: dict) -> dict:
    """
    Fusion récursive de deux dicts. Les valeurs de `override` écrasent
    celles de `base`. Les sous-dicts sont fusionnés récursivement, les
    listes et scalaires sont remplacés.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Chargement YAML
# ---------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict:
    """Lit un fichier YAML et vérifie que la racine est un dict."""
    if not path.exists():
        raise FileNotFoundError(f"Profil introuvable : {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Fichier YAML vide : {path}")
    if not isinstance(data, dict):
        raise ValueError(
            f"Le YAML doit avoir un dict à la racine : {path}"
        )
    return data


# ---------------------------------------------------------------------------
# Construction des sous-configurations
# ---------------------------------------------------------------------------
def _build_donnees(raw: dict | None) -> DonneesConfig:
    """Construit un DonneesConfig depuis un dict YAML."""
    if raw is None:
        return DonneesConfig()

    tick_raw = raw.get("tick_aggregation")
    tick_cfg = None
    if tick_raw is not None:
        tick_cfg = TickAggregationConfig(
            mode=tick_raw.get("mode", "time"),
            params=tick_raw.get("params", {}),
        )

    return DonneesConfig(
        source=raw.get("source", "barres"),
        timeframes=list(raw.get("timeframes", ["M1"])),
        tick_aggregation=tick_cfg,
    )


def _build_context(raw: dict | None) -> ContextConfig:
    """
    Construit un ContextConfig depuis un dict YAML.

    Toutes les clés sont optionnelles : un profil qui ne définit pas de
    contexte obtient la configuration par défaut (forex, NY).
    """
    if raw is None:
        return ContextConfig()

    return ContextConfig(
        reference_tz=raw.get("reference_tz", "America/New_York"),
        sessions=raw.get("sessions", {}) or {},
        rollover=raw.get("rollover"),         # None accepté explicitement
        events=raw.get("events", {}) or {},
        holidays=raw.get("holidays", {}) or {},
    )


def _build_segmentation(raw: dict | None) -> SegmentationConfig:
    """Construit un SegmentationConfig depuis un dict YAML."""
    if raw is None:
        return SegmentationConfig()
    return SegmentationConfig(
        modele=raw.get("modele", "normal"),
        pen=float(raw.get("pen", 3.0)),
        min_size=int(raw.get("min_size", 20)),
    )


def _build_profil(data: dict, source_path: Path | None = None) -> ProfilConfig:
    """Construit un ProfilConfig validé depuis un dict fusionné."""
    return ProfilConfig(
        nom=data.get("nom", "default"),
        type=data.get("type", "default"),
        description=data.get("description", ""),
        donnees=_build_donnees(data.get("donnees")),
        methodes=data.get("methodes", {}),
        context=_build_context(data.get("context")),
        segmentation=_build_segmentation(data.get("segmentation")),
        source_path=source_path,
    )


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------
def load_profile(nom_ou_chemin: str | Path) -> ProfilConfig:
    """
    Charge un profil par nom ou par chemin.

    Le profil est fusionné avec `default.yaml` (héritage implicite).
    Si le nom vaut 'default', le fichier default.yaml est chargé seul.

    Parameters
    ----------
    nom_ou_chemin : str | Path
        Nom court ('eurusd', 'btcusd') ou chemin explicite vers un YAML.

    Returns
    -------
    ProfilConfig
        Profil validé, prêt à l'emploi.

    Examples
    --------
    >>> profil = load_profile("eurusd")
    >>> profil.nom
    'eurusd'
    >>> profil.context.reference_tz
    'America/New_York'
    """
    # Résolution du chemin
    if isinstance(nom_ou_chemin, Path) or str(nom_ou_chemin).endswith(".yaml"):
        user_path = Path(nom_ou_chemin)
        nom = user_path.stem
    else:
        nom = str(nom_ou_chemin)
        user_path = _PROFILES_DIR / f"{nom}.yaml"

    # Chargement du profil default (toujours, sauf si c'est lui-même)
    default_path = _PROFILES_DIR / "default.yaml"
    default_data = _read_yaml(default_path)

    if nom == "default":
        merged = default_data
        source_path = default_path
    else:
        user_data = _read_yaml(user_path)
        merged = _deep_merge(default_data, user_data)
        source_path = user_path

    return _build_profil(merged, source_path=source_path)


def list_profiles() -> list[str]:
    """
    Liste les noms des profils embarqués (sans l'extension .yaml).

    Returns
    -------
    list[str]
        Noms triés alphabétiquement.
    """
    if not _PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in _PROFILES_DIR.glob("*.yaml"))


def profile_hash(profil: ProfilConfig, length: int = 16) -> str:
    """
    Calcule un hash SHA256 stable d'un profil.

    Utilisé pour la reproductibilité : deux runs avec le même profil
    ont le même hash. Le hash couvre tous les paramètres, y compris
    le contexte temporel.

    Parameters
    ----------
    profil : ProfilConfig
        Profil à hasher.
    length : int
        Longueur du préfixe hexadécimal retourné.

    Returns
    -------
    str
        Hash hexadécimal tronqué.
    """
    # Sérialisation canonique : exclut source_path (dépend du FS).
    payload = {
        "nom": profil.nom,
        "type": profil.type,
        "donnees": {
            "source": profil.donnees.source,
            "timeframes": sorted(profil.donnees.timeframes),
            "tick_aggregation": (
                None if profil.donnees.tick_aggregation is None
                else {
                    "mode": profil.donnees.tick_aggregation.mode,
                    "params": profil.donnees.tick_aggregation.params,
                }
            ),
        },
        "methodes": profil.methodes,
        "context": {
            "reference_tz": profil.context.reference_tz,
            "sessions": profil.context.sessions,
            "rollover": profil.context.rollover,
            "events": profil.context.events,
            "holidays": profil.context.holidays,
        },
        "segmentation": {
            "modele": profil.segmentation.modele,
            "pen": profil.segmentation.pen,
            "min_size": profil.segmentation.min_size,
        },
    }
    texte = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(texte.encode("utf-8")).hexdigest()
    return digest[:length]
