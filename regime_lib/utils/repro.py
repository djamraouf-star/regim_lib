"""
Helpers de reproductibilité : hash court et stable des hyperparamètres.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_params(params: dict[str, Any], length: int = 16) -> str:
    """
    Calcule un hash SHA256 court et stable d'un dictionnaire de paramètres.

    Les clés sont triées pour garantir la stabilité. Les valeurs non
    sérialisables JSON sont converties via `repr` en dernier recours.

    Parameters
    ----------
    params : dict
        Hyperparamètres.
    length : int
        Longueur du préfixe hexadécimal retourné.

    Returns
    -------
    str
        Hash hexadécimal tronqué.
    """
    try:
        payload = json.dumps(params, sort_keys=True, ensure_ascii=False,
                             default=str)
    except TypeError:
        payload = json.dumps({k: repr(v) for k, v in params.items()},
                             sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]
