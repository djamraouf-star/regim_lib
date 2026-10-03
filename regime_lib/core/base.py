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
Chaque méthode résout ses paramètres par cascade de priorité :

    1. Argument explicite au constructeur
    2. Sous-profil passé via `profile=...`
    3. `default.yaml` (source de vérité du projet)
    4. Valeur codée en dur dans la méthode (filet de sécurité ultime)

Les niveaux 3 et 4 sont volontairement séparés : `default.yaml` est la
**source de vérité** que les utilisateurs modifient pour changer le
comportement par défaut. Les valeurs codées en dur dans les méthodes ne
sont consultées que si le YAML est indisponible ou corrompu. En
pratique, elles ne sont jamais utilisées dans un environnement sain.

Cette cascade évite la duplication : modifier `default.yaml` suffit,
sans avoir à synchroniser manuellement chaque méthode.

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
from functools import lru_cache
from typing import ClassVar

import pandas as pd


@lru_cache(maxsize=1)
def _load_default_profile():
    """
    Charge le profil `default.yaml` une seule fois (cache mémoire).

    Utilisé comme fallback dans `_resolve` : si aucun profil n'est
    fourni et que l'argument explicite est absent, on lit la valeur
    depuis le YAML. Le fallback codé en dur dans chaque méthode devient
    un ultime filet de sécurité (si le YAML est indisponible).

    Import différé pour éviter un cycle `core ↔ config`.
    Retourne None si le chargement échoue — la cascade continuera alors
    vers le fallback codé en dur.
    """
    try:
        from regime_lib.config import load_profile
        return load_profile("default")
    except Exception:
        return None


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
    profile : dict
        Sous-profil de cette méthode (issu du profil global). Vide si
        aucun profil n'est fourni.
    params : dict
        Trace des paramètres effectivement passés au constructeur
        (hors allow_lookahead, timeframe, profile). Utilisé pour la
        reproductibilité (`params_hash`).
    """

    name: ClassVar[str] = "base"
    REGIME_MAP: ClassVar[dict[str, int]] = {}
    requires_lookahead: ClassVar[bool] = False
    REQUIRES_COLUMNS: ClassVar[list[str]] = []

    def __init__(
        self,
        allow_lookahead: bool = False,
        timeframe: str | None = None,
        profile: dict | None = None,
        **params,
    ) -> None:
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
        if profile is not None and not isinstance(profile, dict):
            raise TypeError(
                f"{type(self).__name__} : profile doit être un dict ou None, "
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

        self.allow_lookahead: bool = allow_lookahead
        self.timeframe: str | None = timeframe
        self.profile: dict = dict(profile) if profile else {}
        self.params: dict = dict(params)
        # On mémorise le timeframe dans params pour qu'il entre dans le hash.
        if timeframe is not None:
            self.params.setdefault("timeframe", timeframe)

    # ------------------------------------------------------------------
    # Helpers de résolution de paramètres (à utiliser dans les sous-classes)
    # ------------------------------------------------------------------
    def _resolve(
        self,
        key: str,
        explicit: object,
        default: object,
        per_timeframe: bool = False,
    ) -> object:
        """
        Résout un paramètre par cascade :

            explicite
            > sous-profil (avec timeframes si applicable)
            > default.yaml (source de vérité)
            > valeur codée en dur (filet de sécurité ultime)

        Parameters
        ----------
        key : str
            Nom du paramètre (clé dans le sous-profil / le YAML).
        explicit : object
            Valeur passée explicitement au constructeur. `None` signifie
            "non fourni".
        default : object
            Fallback ultime si ni le profil passé, ni `default.yaml` ne
            définissent la clé. Devrait être rarement utilisé.
        per_timeframe : bool
            Si True, cherche la clé dans
            `profile['timeframes'][timeframe]` puis dans
            `default.yaml → methodes.<name>.timeframes.<timeframe>`.

        Returns
        -------
        object
            La valeur résolue (peut être None si `default` est None et
            que rien n'est trouvé, à la sous-classe de gérer).
        """
        # --- 1. Explicite ----------------------------------------------
        if explicit is not None:
            return explicit

        # --- 2. Sous-profil passé à l'instance --------------------------
        if per_timeframe and self.timeframe is not None:
            tf_params = self.profile.get("timeframes", {}).get(
                self.timeframe, {}
            )
            if key in tf_params:
                return tf_params[key]
        if key in self.profile:
            return self.profile[key]

        # --- 3. default.yaml (source de vérité) -------------------------
        profil_def = _load_default_profile()
        if profil_def is not None:
            method_cfg = profil_def.methodes.get(self.name, {})
            if per_timeframe and self.timeframe is not None:
                tf_cfg = method_cfg.get("timeframes", {}).get(
                    self.timeframe, {}
                )
                if key in tf_cfg:
                    return tf_cfg[key]
            if key in method_cfg:
                return method_cfg[key]

        # --- 4. Filet de sécurité ultime --------------------------------
        return default

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
            f"profile_keys={list(self.profile.keys())}, "
            f"params={self.params!r})"
        )
