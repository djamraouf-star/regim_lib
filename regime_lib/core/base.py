"""
Contrat de base pour tous les détecteurs de régime.

Toute nouvelle méthode DOIT :
  - hériter de `RegimeDetector` ;
  - définir un attribut de classe `REGIME_MAP` (dict[str, int]) stable ;
  - définir un attribut de classe `name` (str, snake_case court) ;
  - implémenter `fit_predict(df) -> pd.DataFrame`.

Le DataFrame retourné doit respecter le format uniforme décrit dans
`regime_lib.core.output`.

Résolution des paramètres
-------------------------
Le schéma PARAM_SCHEMA déclare types et contraintes. Le résolveur commun
applique les défauts YAML, le profil, puis les arguments explicites. Aucune
valeur par défaut n'est définie dans les détecteurs. ``params`` contient les
valeurs effectives et ``param_sources`` leur provenance.

Colonnes requises
-----------------
Une méthode peut déclarer `REQUIRES_COLUMNS`, une liste de colonnes
qui doivent être présentes dans le DataFrame d'entrée. Elle appelle
alors `verifier_colonnes(df)` au début de `fit_predict`. Utile pour
les méthodes qui dépendent de colonnes tick enrichies (ex. `ofi`
requiert `bid_volume` et `ask_volume`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from regime_lib.config.parameters import MethodProfile, ParameterSchema, resolve_parameters


class RegimeDetector(ABC):
    """
    Classe de base pour tous les détecteurs de régime.

    Attributs de classe (à surcharger)
    ----------------------------------
    name : str
        Nom court de la méthode, snake_case (ex. "adx", "hmm_gaussian").
        Utilisé tel quel dans la colonne `method`, dans le CLI, et comme
        clé dans `default.yaml`.
    REGIME_MAP : dict[str, int]
        Correspondance stable entre label lisible et identifiant numérique.
        Contraintes validées à l'instanciation :
          - non vide ;
          - clés de type `str` ;
          - valeurs de type `int` (les `bool` sont explicitement exclus,
            car `bool` est une sous-classe de `int` en Python) ;
          - valeurs uniques (bijection label -> id).
    requires_lookahead : bool
        Si True, la méthode exige explicitement le lookahead. Le runner
        lèvera une erreur si `allow_lookahead` est False.
    REQUIRES_COLUMNS : list[str]
        Colonnes obligatoires dans le DataFrame d'entrée (après
        normalisation lowercase). La sous-classe appelle
        `self.verifier_colonnes(df)` au début de `fit_predict`. Défaut :
        liste vide (aucune exigence spécifique au-delà du contrat OHLCV).

    Attributs d'instance
    --------------------
    allow_lookahead : bool
        Autorisation effective du lookahead pour cette instance.
    timeframe : str | None
        Timeframe courant (ex. "H1"). Utilisé par certaines méthodes
        pour résoudre leurs paramètres par timeframe.
    profile : dict | MethodProfile | None
        Surcharges brutes ou instantané de cette méthode. None si absent.
    params : dict
        Paramètres résolus et validés, plus le timeframe si fourni.
        Utilisés pour la reproductibilité (`params_hash`).
    """

    PARAM_SCHEMA: ClassVar[ParameterSchema | None] = None
    name: ClassVar[str] = "base"
    REGIME_MAP: ClassVar[dict[str, int]] = {}
    requires_lookahead: ClassVar[bool] = False
    REQUIRES_COLUMNS: ClassVar[list[str]] = []

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | MethodProfile | None = None,
        **params,
    ) -> None:
        self.allow_lookahead = allow_lookahead
        self.timeframe = timeframe
        self.profile = profile
        if self.PARAM_SCHEMA is not None:
            resolved = resolve_parameters(
                self.name, self.PARAM_SCHEMA, timeframe, profile, params,
            )
            self.params = resolved.values
            self.param_sources = resolved.sources
            for key, value in self.params.items():
                setattr(self, key, value)
        else:
            # Sous-classes sans paramètres déclarés (contrat de base).
            self.params = dict(params)
            self.param_sources = {key: "explicit" for key in params}
        if timeframe is not None:
            self.params["timeframe"] = timeframe
        self._configure_regime_map()

        # --- Validation de REGIME_MAP -----------------------------------
        if not self.REGIME_MAP:
            raise ValueError(
                f"{type(self).__name__} doit définir un REGIME_MAP non vide."
            )

        if not all(isinstance(k, str) for k in self.REGIME_MAP.keys()):
            bad = [k for k in self.REGIME_MAP.keys() if not isinstance(k, str)]
            raise ValueError(
                f"{type(self).__name__} : toutes les clés de REGIME_MAP "
                f"doivent être des str. Clés invalides : {bad!r}."
            )

        if not all(
            isinstance(v, int) and not isinstance(v, bool)
            for v in self.REGIME_MAP.values()
        ):
            bad = [
                (k, v) for k, v in self.REGIME_MAP.items()
                if not (isinstance(v, int) and not isinstance(v, bool))
            ]
            raise ValueError(
                f"{type(self).__name__} : toutes les valeurs de REGIME_MAP "
                f"doivent être des int (bool exclus). "
                f"Entrées invalides : {bad!r}."
            )

        if len(set(self.REGIME_MAP.values())) != len(self.REGIME_MAP):
            raise ValueError(
                f"{type(self).__name__} : REGIME_MAP contient des valeurs "
                f"dupliquées."
            )

        # --- Validation du lookahead ------------------------------------
        if self.requires_lookahead and not allow_lookahead:
            raise ValueError(
                f"{self.name} exige le lookahead mais il est désactivé. "
                f"Utiliser --allow-lookahead true ou "
                f"--allow-lookahead-override {self.name}:true."
            )

        # --- Validation du profil ---------------------------------------
        if profile is not None and not isinstance(profile, (dict, MethodProfile)):
            raise TypeError(
                f"{type(self).__name__} : profile doit être un dict, un MethodProfile ou None, "
                f"reçu {type(profile).__name__}."
            )

        # --- Validation de REQUIRES_COLUMNS -----------------------------
        if not isinstance(self.REQUIRES_COLUMNS, list):
            raise TypeError(
                f"{type(self).__name__} : REQUIRES_COLUMNS doit être une "
                f"liste, reçu {type(self.REQUIRES_COLUMNS).__name__}."
            )
        if not all(isinstance(c, str) for c in self.REQUIRES_COLUMNS):
            bad = [
                c for c in self.REQUIRES_COLUMNS
                if not isinstance(c, str)
            ]
            raise ValueError(
                f"{type(self).__name__} : toutes les entrées de "
                f"REQUIRES_COLUMNS doivent être des str. Invalides : {bad!r}."
            )

    def _configure_regime_map(self) -> None:
        """Hook après résolution, pour les labels dépendant des paramètres."""

    # ------------------------------------------------------------------
    # Vérification des colonnes requises
    # ------------------------------------------------------------------
    def verifier_colonnes(self, df: pd.DataFrame) -> None:
        """
        Vérifie la présence des colonnes requises dans `df`.

        Chaque méthode qui déclare `REQUIRES_COLUMNS` doit appeler
        cette fonction au début de `fit_predict`, **après** avoir
        normalisé les noms de colonnes en lowercase.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame d'entrée (après normalisation lowercase).

        Raises
        ------
        ValueError
            Si une colonne de `REQUIRES_COLUMNS` est absente. Le message
            liste les colonnes manquantes et les colonnes disponibles.
        """
        if not self.REQUIRES_COLUMNS:
            return
        manquantes = [
            c for c in self.REQUIRES_COLUMNS if c not in df.columns
        ]
        if manquantes:
            raise ValueError(
                f"{self.name} requiert les colonnes {manquantes}, "
                f"absentes du DataFrame. "
                f"Colonnes disponibles : {sorted(df.columns)}."
            )

    @abstractmethod
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les régimes et retourne le DataFrame au format uniforme.

        Parameters
        ----------
        df : pd.DataFrame
            Doit contenir au minimum les colonnes : open, high, low, close, volume.
            Index : DatetimeIndex nommé `timestamp`, trié strictement croissant,
            sans doublons.

        Returns
        -------
        pd.DataFrame
            Colonnes obligatoires (préfixe, dans cet ordre) :
                timestamp, regime, regime_id, confidence, method, timeframe, asset
            Colonnes optionnelles autorisées après :
                is_partial, params_hash
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"timeframe={self.timeframe!r}, "
            f"allow_lookahead={self.allow_lookahead}, "
            f"profile={self.profile!r}, "
            f"params={self.params!r})"
        )
