"""
Chargement, fusion et hachage des profils.

Le loader est le point d'entrée principal du module config. Il gère :
  - le chargement d'un fichier YAML ;
  - la fusion des sections générales et la conservation des couches de méthodes ;
  - la construction d'un `ProfilConfig` validé ;
  - le calcul d'un hash stable pour la reproductibilité.
"""

from __future__ import annotations

from copy import deepcopy
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


class _UniqueKeyLoader(yaml.SafeLoader):
    """Refuse les doublons YAML que SafeLoader écraserait silencieusement."""

    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise yaml.constructor.ConstructorError(
                    None, None, "Utiliser les profils plutôt que la fusion YAML <<.",
                    key_node.start_mark,
                )
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise yaml.constructor.ConstructorError(
                    None, None, "Les clés YAML doivent être des chaînes.", key_node.start_mark,
                )
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    None, None, f"Clé YAML dupliquée : {key!r}.", key_node.start_mark,
                )
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


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
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML invalide dans {path}: {exc}") from exc
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


def _build_profil(
    data: dict, source_path: Path | None = None, *, method_defaults: dict | None = None,
) -> ProfilConfig:
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
        method_defaults=method_defaults,
    )


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------
def load_profile(nom_ou_chemin: str | Path) -> ProfilConfig:
    """
    Charge un profil par nom ou par chemin.

    Les méthodes conservent séparément défauts et surcharges. Seules les
    sections générales sont fusionnées. Le nom 'default' désigne le fichier
    embarqué ; un chemin explicite nommé default.yaml reste un profil utilisateur.

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
    default_path = _PROFILES_DIR / "default.yaml"
    if isinstance(nom_ou_chemin, Path) or str(nom_ou_chemin).endswith((".yaml", ".yml")):
        user_path = Path(nom_ou_chemin)
    else:
        user_path = _PROFILES_DIR / f"{nom_ou_chemin}.yaml"
    default_data = _read_yaml(default_path)
    is_default = user_path.resolve() == default_path.resolve()
    user_data = {} if is_default else _read_yaml(user_path)
    defaults = default_data.get("methodes", {})
    overrides = user_data.get("methodes", {})
    # Seules les sections générales conservent leur fusion profonde historique.
    # Les méthodes gardent deux couches, résolues une seule fois selon le TF.
    merged = _deep_merge(
        {k: v for k, v in default_data.items() if k != "methodes"},
        {k: v for k, v in user_data.items() if k != "methodes"},
    )
    merged["methodes"] = deepcopy(overrides)
    return _build_profil(merged, source_path=user_path, method_defaults=deepcopy(defaults))


def _validate_methods(defaults: dict, overrides: dict, source: str) -> None:
    from regime_lib.core.registry import METHOD_REGISTRY
    from regime_lib.config.parameters import MethodProfile, TIMEFRAMES, resolve_parameters

    for layer in (defaults, overrides):
        if not isinstance(layer, dict):
            raise ValueError(f"{source}: methodes doit être un dictionnaire.")
        for name in layer:
            if name not in METHOD_REGISTRY:
                raise ValueError(f"{source}: methodes.{name}: méthode inconnue.")
    for name in defaults.keys() | overrides.keys() | METHOD_REGISTRY.keys():
        schema = METHOD_REGISTRY[name].PARAM_SCHEMA
        if schema is None:
            if name in defaults or name in overrides:
                raise ValueError(f"methodes.{name}: schéma de paramètres absent.")
            continue
        profile = MethodProfile(name, defaults.get(name, {}), overrides.get(name, {}), source)
        for tf in (None, *TIMEFRAMES):
            resolve_parameters(name, schema, tf, profile)


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
        "method_defaults": profil.method_defaults,
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
