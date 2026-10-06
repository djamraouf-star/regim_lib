"""Vérifications de bout en bout de la configuration CLI et des exports."""
import json

import pandas as pd
import pytest

from regime_lib import cli
from regime_lib.config import load_profile
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.methods.trend.adx import ADXDetector
from regime_lib.utils.repro import hash_params


def test_show_config_matches_python_for_every_method(capsys, monkeypatch):
    def no_data(*args, **kwargs):
        pytest.fail("show-config ne doit pas lire de données")
    monkeypatch.setattr(cli, "load_parquet", no_data)
    assert cli.main([
        "--show-config", "--profile", "eurusd", "--timeframe", "M1,H1",
        "--methods", ",".join(METHOD_REGISTRY), "--allow-lookahead", "true",
    ]) == 0
    config = json.loads(capsys.readouterr().out)
    profile = load_profile("eurusd")
    for tf, methods in config.items():
        for name, item in methods.items():
            detector = METHOD_REGISTRY[name](timeframe=tf, profile=profile.for_method(name), allow_lookahead=True)
            assert item["params"] == detector.params
            assert item["sources"] == detector.param_sources
            assert item["params_hash"] == hash_params(detector.params)


@pytest.mark.parametrize("extra, message", [
    ({"adx": {"n_adxx": 20}}, "n_adxx"),
    ({"adx": {"n_adxx": None}}, "n_adxx"),
    ({"adx": {"n_adx": True}}, "n_adx"),
    ({"adx": {"timeframes": {"H1": {"n_adx": 20}}}}, "scalaires"),
    ({"shannon": {"fenetre": 20}}, "non sélectionnées"),
])
def test_cli_rejects_invalid_overrides_before_reading_data(extra, message, capsys, monkeypatch):
    monkeypatch.setattr(cli, "load_parquet", lambda *a, **kw: pytest.fail("lecture prématurée"))
    assert cli.main([
        "--url", "absent.parquet", "--output", "unused", "--methods", "adx",
        "--timeframe", "H1", "--method-params", json.dumps(extra),
    ]) == 1
    assert message in capsys.readouterr().err


def test_none_explicit_has_same_meaning_in_python_and_cli(capsys):
    assert cli.main([
        "--show-config", "--methods", "adx", "--timeframe", "H1",
        "--method-params", '{"adx": {"n_adx": null}}',
    ]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["H1"]["adx"]["params"] == ADXDetector(timeframe="H1", n_adx=None).params


def test_export_records_effective_parameters_and_matching_hash(tmp_path, ohlcv_h1):
    source = tmp_path / "prices.parquet"
    output = tmp_path / "result"
    ohlcv_h1.to_parquet(source)
    assert cli.main([
        "--url", str(source), "--asset", "TEST", "--source-timeframe", "H1",
        "--output", str(output), "--methods", "adx", "--timeframe", "H1,H4",
        "--profile", "eurusd", "--method-params", '{"adx": {"n_adx": 7}}',
    ]) == 0
    meta = json.loads((output / "run_meta.json").read_text())
    data = pd.read_parquet(output / "regimes.parquet")
    for tf in ("H1", "H4"):
        effective = meta["effective_config"][tf]["adx"]
        assert effective["params"]["n_adx"] == 7
        assert effective["params"]["seuil_range"] == 22.0
        assert effective["sources"]["n_adx"] == "explicit.n_adx"
        assert set(data.loc[data.timeframe == tf, "params_hash"]) == {effective["params_hash"]}
