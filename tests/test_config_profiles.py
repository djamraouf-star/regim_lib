"""Contrat d'héritage unique, validation stricte et instantanés de configuration."""
from copy import deepcopy
import inspect

import pytest
import yaml

from regime_lib.config import ProfilConfig, load_profile, profile_hash
from regime_lib.config import loader
from regime_lib.config.parameters import TIMEFRAMES
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.methods.entropy.shannon import ShannonEntropyDetector
from regime_lib.methods.statistical.hmm_gaussian import HMMGaussianDetector
from regime_lib.methods.trend.adx import ADXDetector
from regime_lib.methods.volatility.atr_volatility import ATRVolatilityDetector
from regime_lib.utils.repro import hash_params


def write_profile(tmp_path, methods, name="custom.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"nom": "custom", "methodes": methods}))
    return path


def test_profiles_preserve_layers_and_inherit_omitted_methods():
    default = load_profile("default")
    custom = load_profile("eurusd")
    assert default.methodes == {}
    assert "pca_axis" not in custom.methodes
    assert custom.for_method("pca_axis").defaults["fenetre"] == 100
    assert ADXDetector(profile=custom.for_method("adx")).seuil_range == 22
    assert ShannonEntropyDetector(timeframe="M1", profile=custom.for_method("shannon")).fenetre == 100
    assert ShannonEntropyDetector(timeframe="M1").fenetre == 200


@pytest.mark.parametrize("name", sorted(METHOD_REGISTRY))
def test_all_methods_share_python_and_loaded_profile_defaults(name):
    cls = METHOD_REGISTRY[name]
    profile = load_profile("default")
    # Les signatures restent ergonomiques ; ce garde-fou évite leur divergence
    # avec le schéma qui fait autorité sur les clés et leurs contraintes.
    arguments = set(inspect.signature(cls).parameters) - {"profile", "timeframe", "allow_lookahead"}
    assert arguments == set(cls.PARAM_SCHEMA.parameters)
    for tf in (None, *TIMEFRAMES):
        direct = cls(timeframe=tf, allow_lookahead=True)
        loaded = cls(timeframe=tf, allow_lookahead=True, profile=profile.for_method(name))
        assert direct.params == loaded.params
        assert direct.param_sources == loaded.param_sources
        assert all(getattr(direct, key) == direct.params[key] for key in arguments)


def test_priority_is_layer_first_then_timeframe(tmp_path):
    path = write_profile(tmp_path, {"atr_volatility": {
        "n_atr": 31, "fenetre": 80, "timeframes": {"H1": {"n_atr": 32}},
    }})
    profile = load_profile(path)
    h1 = ATRVolatilityDetector(timeframe="H1", profile=profile.for_method("atr_volatility"))
    m1 = ATRVolatilityDetector(timeframe="M1", profile=profile.for_method("atr_volatility"))
    explicit = ATRVolatilityDetector(timeframe="H1", profile=profile.for_method("atr_volatility"), n_atr=33)
    assert (h1.n_atr, m1.n_atr, explicit.n_atr) == (32, 31, 33)
    assert (h1.fenetre, m1.fenetre, h1.min_periods) == (80, 80, 40)
    assert h1.param_sources["n_atr"].endswith("timeframes.H1.n_atr")
    assert m1.param_sources["n_atr"].endswith("atr_volatility.n_atr")
    assert explicit.param_sources["n_atr"] == "explicit.n_atr"


@pytest.mark.parametrize("extension", ["yaml", "yml"])
def test_user_file_named_default_is_not_ignored(tmp_path, extension):
    path = write_profile(tmp_path, {"adx": {"n_adx": 21}}, f"default.{extension}")
    profile = load_profile(str(path))
    assert profile.source_path == path
    assert ADXDetector(profile=profile.for_method("adx")).n_adx == 21


@pytest.mark.parametrize("methods, message", [
    ({"adxx": {}}, "méthode inconnue"),
    ({"atr_volatility": {"n_atrr": 20}}, "n_atrr.*n_atr"),
    ({"adx": {"n_adx": True}}, "n_adx.*type int"),
    ({"adx": {"n_adx": 2.5}}, "n_adx.*type int"),
    ({"adx": {"n_adx": "14"}}, "n_adx.*type int"),
    ({"adx": {"n_adx": None}}, "n_adx.*type int"),
    ({"adx": {"seuil_range": float("nan")}}, "doit être fini"),
    ({"adx": {"seuil_range": float("inf")}}, "doit être fini"),
    ({"adx": {"seuil_range": 30}}, "seuil_range < seuil_tendance"),
    ({"shannon": {"timeframes": {"H2": {"fenetre": 20}}}}, "Timeframe inconnu"),
    ({"shannon": {"timeframes": {"M1": {"n_binns": 8}}}}, "n_binns.*n_bins"),
    ({"adx": {"timeframes": {"H1": {"n_adx": 20}}}}, "par timeframe"),
    ({"atr_volatility": {"fenetre": 20, "min_periods": 21}}, "min_periods <= fenetre"),
    ({"shannon": {"timeframes": []}}, "dictionnaire"),
])
def test_invalid_configuration_fails_on_load(tmp_path, methods, message):
    with pytest.raises(ValueError, match=message):
        load_profile(write_profile(tmp_path, methods))


def test_shannon_bins_and_hmm_states_follow_common_resolver():
    shannon = ShannonEntropyDetector(timeframe="M1", profile={"timeframes": {"M1": {"n_bins": 7}}})
    hmm = HMMGaussianDetector(allow_lookahead=True, profile={"n_states": 4}, n_states=3)
    assert shannon.n_bins == 7
    assert hmm.n_states == 3
    assert hmm.REGIME_MAP == {"STATE_0": 0, "STATE_1": 1, "STATE_2": 2, "INCONNU": 3}


def test_auto_window_and_effective_hash():
    small = ATRVolatilityDetector(timeframe="H1", fenetre=41)
    same = ATRVolatilityDetector(timeframe="H1", profile={"fenetre": 41, "min_periods": 20})
    reset = ATRVolatilityDetector(timeframe="H1", profile={"min_periods": None}, fenetre=41)
    assert small.min_periods == reset.min_periods == 20
    assert hash_params(small.params) == hash_params(same.params)
    assert hash_params(small.params) != hash_params(ATRVolatilityDetector(timeframe="H1").params)


def test_raw_profile_is_not_mutated_and_bad_keys_are_rejected():
    raw = {"timeframes": {"H1": {"fenetre": 41}}}
    before = deepcopy(raw)
    ATRVolatilityDetector(timeframe="H1", profile=raw)
    assert raw == before
    with pytest.raises(ValueError, match="n_atrr"):
        ATRVolatilityDetector(profile={"n_atrr": None})
    with pytest.raises(ValueError, match="Timeframe inconnu"):
        ADXDetector(timeframe="H2")


@pytest.fixture
def defaults_file(tmp_path, monkeypatch):
    path = tmp_path / "default.yaml"
    path.write_text((loader._PROFILES_DIR / "default.yaml").read_text())
    monkeypatch.setattr(loader, "_PROFILES_DIR", tmp_path)
    return path


def test_reload_reads_new_defaults_but_existing_snapshot_is_stable(defaults_file):
    profile = load_profile("default")
    programmatic = ProfilConfig(nom="python")
    before = profile_hash(profile)
    raw = yaml.safe_load(defaults_file.read_text())
    raw["methodes"]["adx"]["n_adx"] = 31
    defaults_file.write_text(yaml.safe_dump(raw))
    assert ADXDetector().n_adx == 31
    assert ADXDetector(profile=profile.for_method("adx")).n_adx == 14
    assert ADXDetector(profile=programmatic.for_method("adx")).n_adx == 14
    assert profile_hash(profile) == before
    assert profile_hash(load_profile("default")) != before
    detached = profile.for_method("adx")
    detached.defaults["n_adx"] = 99
    assert profile.for_method("adx").defaults["n_adx"] == 14


@pytest.mark.parametrize("problem", ["missing", "corrupt", "missing_parameter"])
def test_no_silent_fallback_to_code(defaults_file, problem):
    if problem == "missing":
        defaults_file.unlink()
        error, message = FileNotFoundError, "Profil introuvable"
    elif problem == "corrupt":
        defaults_file.write_text("methodes: [")
        error, message = ValueError, "YAML invalide"
    else:
        raw = yaml.safe_load(defaults_file.read_text())
        del raw["methodes"]["adx"]["n_adx"]
        defaults_file.write_text(yaml.safe_dump(raw))
        error, message = ValueError, "n_adx.*manquante"
    with pytest.raises(error, match=message):
        ADXDetector(n_adx=20)


def test_duplicate_yaml_key_is_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("methodes:\n  adx:\n    n_adx: 14\n    n_adx: 20\n")
    with pytest.raises(ValueError, match="dupliquée.*n_adx"):
        load_profile(path)


def test_programmatic_profile_validates_methods():
    with pytest.raises(ValueError, match="n_adxx"):
        ProfilConfig(nom="python", methodes={"adx": {"n_adxx": 20}})
    with pytest.raises(ValueError, match="méthode inconnue"):
        load_profile("default").for_method("adxx")
