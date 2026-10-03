"""Tests du contrat RegimeDetector."""

from __future__ import annotations

import pytest

from regime_lib.core.base import RegimeDetector


def test_rejects_empty_regime_map():
    class Bad(RegimeDetector):
        name = "bad"
        REGIME_MAP = {}

        def fit_predict(self, df):
            return df

    with pytest.raises(ValueError, match="REGIME_MAP non vide"):
        Bad()


def test_rejects_non_str_keys():
    class Bad(RegimeDetector):
        name = "bad"
        REGIME_MAP = {0: 0}  # clé non str

        def fit_predict(self, df):
            return df

    with pytest.raises(ValueError, match="clés de REGIME_MAP"):
        Bad()


def test_rejects_bool_values():
    class Bad(RegimeDetector):
        name = "bad"
        REGIME_MAP = {"a": True, "b": 1}  # bool déguisé en int

        def fit_predict(self, df):
            return df

    with pytest.raises(ValueError, match="int \\(bool exclus\\)"):
        Bad()


def test_rejects_duplicate_values():
    class Bad(RegimeDetector):
        name = "bad"
        REGIME_MAP = {"a": 0, "b": 0}

        def fit_predict(self, df):
            return df

    with pytest.raises(ValueError, match="valeurs dupliquées"):
        Bad()


def test_requires_lookahead_raises_when_disabled():
    class NeedsLA(RegimeDetector):
        name = "needs_la"
        REGIME_MAP = {"a": 0}
        requires_lookahead = True

        def fit_predict(self, df):
            return df

    with pytest.raises(ValueError, match="exige le lookahead"):
        NeedsLA(allow_lookahead=False)

    # Avec autorisation explicite, ça passe.
    NeedsLA(allow_lookahead=True)


def test_timeframe_stored_in_params():
    class Dummy(RegimeDetector):
        name = "dummy"
        REGIME_MAP = {"a": 0}

        def fit_predict(self, df):
            return df

    d = Dummy(timeframe="H1", foo=42)
    assert d.timeframe == "H1"
    assert d.params["timeframe"] == "H1"
    assert d.params["foo"] == 42
