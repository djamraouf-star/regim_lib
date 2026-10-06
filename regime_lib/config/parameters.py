"""Schémas déclaratifs et résolution unique des paramètres des détecteurs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from difflib import get_close_matches
import math
from typing import Any

TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1")


@dataclass(frozen=True)
class Parameter:
    """Contrat d'une valeur ; les défauts appartiennent exclusivement au YAML.

    ``half_window`` autorise null : la valeur est alors calculée comme la
    moitié entière du paramètre indiqué, après résolution de toutes les couches.
    """
    type: type
    minimum: float | None = None
    maximum: float | None = None
    exclusive_min: bool = False
    exclusive_max: bool = False
    per_timeframe: bool = False
    half_window: str | None = None

    def validate(self, value: Any, path: str) -> Any:
        if value is None and self.half_window is not None:
            return None
        valid = (
            type(value) is int if self.type is int else
            type(value) in (int, float) if self.type is float else
            type(value) is self.type
        )
        if not valid:
            raise ValueError(f"{path} doit être de type {self.type.__name__}, reçu {value!r}.")
        if self.type is float:
            value = float(value)
        if self.type in (int, float):
            if not math.isfinite(value):
                raise ValueError(f"{path} doit être fini.")
            if self.minimum is not None and (
                value < self.minimum or self.exclusive_min and value == self.minimum
            ):
                op = ">" if self.exclusive_min else ">="
                raise ValueError(f"{path} doit être {op} {self.minimum}.")
            if self.maximum is not None and (
                value > self.maximum or self.exclusive_max and value == self.maximum
            ):
                op = "<" if self.exclusive_max else "<="
                raise ValueError(f"{path} doit être {op} {self.maximum}.")
        return value


@dataclass(frozen=True)
class Ordered:
    """Contrainte entre deux paramètres : lower < upper (ou <=)."""
    lower: str
    upper: str
    equal: bool = False

    def validate(self, values: dict, path: str) -> None:
        a, b = values[self.lower], values[self.upper]
        if not (a <= b if self.equal else a < b):
            op = "<=" if self.equal else "<"
            raise ValueError(f"{path}: il faut {self.lower} {op} {self.upper}.")


@dataclass(frozen=True)
class ParameterSchema:
    parameters: dict[str, Parameter]
    constraints: tuple[Ordered, ...] = ()

    def explicit_values(self, values: dict) -> dict:
        """None signifie absent dans les arguments, sans masquer une clé inconnue."""
        if "timeframes" in values:
            raise ValueError("Les arguments explicites doivent être scalaires.")
        return {
            key: value for key, value in values.items()
            if value is not None or key not in self.parameters
        }

    def validate_layer(self, layer: dict, path: str, *, timeframe: bool = False) -> None:
        if not isinstance(layer, dict):
            raise ValueError(f"{path} doit être un dictionnaire.")
        for key, value in layer.items():
            if key == "timeframes" and not timeframe:
                if not isinstance(value, dict):
                    raise ValueError(f"{path}.timeframes doit être un dictionnaire.")
                for tf, params in value.items():
                    validate_timeframe(tf)
                    if tf is None:
                        raise ValueError(f"{path}.timeframes: null n'est pas un timeframe.")
                    self.validate_layer(params, f"{path}.timeframes.{tf}", timeframe=True)
                continue
            if key not in self.parameters:
                matches = get_close_matches(str(key), self.parameters, n=1)
                hint = f" Vouliez-vous dire {matches[0]!r} ?" if matches else ""
                raise ValueError(f"{path}.{key}: paramètre inconnu.{hint}")
            spec = self.parameters[key]
            if timeframe and not spec.per_timeframe:
                raise ValueError(f"{path}.{key} n'accepte pas de valeur par timeframe.")
            spec.validate(value, f"{path}.{key}")


@dataclass(frozen=True)
class MethodProfile:
    """Instantané des deux couches, jamais fusionnées au chargement."""
    method: str
    defaults: dict
    overrides: dict
    source: str = "profile"


@dataclass(frozen=True)
class ResolvedParameters:
    values: dict
    sources: dict[str, str]


def validate_timeframe(timeframe: str | None) -> None:
    if timeframe is not None and timeframe not in TIMEFRAMES:
        raise ValueError(f"Timeframe inconnu : {timeframe!r}. Attendu : {TIMEFRAMES}.")


def resolve_parameters(
    method: str,
    schema: ParameterSchema,
    timeframe: str | None = None,
    profile: dict | MethodProfile | None = None,
    explicit: dict | None = None,
) -> ResolvedParameters:
    """Défauts généraux/TF < profil général/TF < arguments explicites.

    None dans les arguments Python signifie absent (API historique). Dans un
    YAML, null n'est accepté que pour un paramètre à calcul automatique déclaré.
    """
    validate_timeframe(timeframe)
    if isinstance(profile, MethodProfile):
        if profile.method != method:
            raise ValueError(f"Profil de {profile.method!r} fourni à {method!r}.")
        snapshot = profile
    else:
        if profile is not None and not isinstance(profile, dict):
            raise TypeError("profile doit être un dict, un MethodProfile ou None.")
        from regime_lib.config.loader import load_profile
        defaults = load_profile("default").for_method(method)
        snapshot = MethodProfile(method, defaults.defaults, deepcopy(profile or {}))
    path = f"methodes.{method}"
    schema.validate_layer(snapshot.defaults, f"default.{path}")
    schema.validate_layer(snapshot.overrides, f"{snapshot.source}.{path}")
    explicit = schema.explicit_values(explicit or {})
    schema.validate_layer(explicit, f"explicit.{path}")
    values: dict = {}
    sources: dict[str, str] = {}
    for layer, source in (
        (snapshot.defaults, f"default.{path}"),
        (snapshot.overrides, f"{snapshot.source}.{path}"),
    ):
        for key, value in layer.items():
            if key != "timeframes":
                values[key], sources[key] = value, f"{source}.{key}"
        if timeframe is not None:
            for key, value in layer.get("timeframes", {}).get(timeframe, {}).items():
                values[key], sources[key] = value, f"{source}.timeframes.{timeframe}.{key}"
    for key, value in explicit.items():
        values[key], sources[key] = value, f"explicit.{key}"
    for key, spec in schema.parameters.items():
        if key not in values:
            raise ValueError(f"{path}.{key}: valeur par défaut manquante dans le YAML.")
        values[key] = spec.validate(values[key], f"{path}.{key}")
    for key, spec in schema.parameters.items():
        if values[key] is None and spec.half_window:
            values[key] = max(1, values[spec.half_window] // 2)
            sources[key] += f" (auto: {spec.half_window} // 2)"
            spec.validate(values[key], f"{path}.{key}")
    for constraint in schema.constraints:
        constraint.validate(values, path)
    return ResolvedParameters(deepcopy(values), sources)
