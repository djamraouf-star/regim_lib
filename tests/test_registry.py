"""Tests du registre des méthodes."""

from __future__ import annotations

import pytest

from regime_lib.core.base import RegimeDetector
from regime_lib.core.registry import METHOD_REGISTRY, register_method


def test_registry_contains_atr_volatility():
    # L'import du package methods doit avoir peuplé le registre via cli.
    import regime_lib.cli  # noqa: F401
    assert "atr_volatility" in METHOD_REGISTRY
    assert issubclass(METHOD_REGISTRY["atr_volatility"], RegimeDetector)


def test_register_method_rejects_duplicate():
    class Dummy(RegimeDetector):
        name = "atr_volatility"  # collision volontaire
        REGIME_MAP = {"a": 0}

        def fit_predict(self, df):
            return df

    with pytest.raises(ValueError, match="déjà enregistrée"):
        register_method(Dummy)


def test_register_method_rejects_non_detector():
    class NotADetector:
        name = "nope"

    with pytest.raises(TypeError, match="RegimeDetector"):
        register_method(NotADetector)  # type: ignore[arg-type]
