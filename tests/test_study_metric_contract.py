"""Applicabilité, ordre déclaré et effectifs des métriques."""
import numpy as np
import pandas as pd
import pytest

from regime_lib.study import Study
from regime_lib.study.evaluation import feature_type, hit_rate_directionnel, ic_spearman
from regime_lib.study.targets import calculer_cible, liste_cibles, parser_cible


@pytest.mark.parametrize("name", ["regime", "adx__regime", "ctx_hour_ny", "ctx_day_of_week"])
def test_numeric_codes_do_not_imply_order(name):
    f = pd.Series(np.arange(30), name=name)
    result = ic_spearman(f, pd.Series(np.arange(30)))
    assert feature_type(f) == "nominal"
    assert result["status"] == "not_applicable_nominal"
    assert np.isnan(result["ic"])


def test_declared_order_is_respected():
    f = pd.Series(pd.Categorical(["low"] * 10 + ["high"] * 10,
                                categories=["high", "low"], ordered=True))
    target = pd.Series([0.] * 10 + [1.] * 10)
    result = ic_spearman(f, target)
    assert result["ic"] == pytest.approx(-1)
    assert result["status"] == "ok"
    with pytest.raises(ValueError, match="explicitement ordonnée"):
        ic_spearman(f.astype(str), target, type_feature="ordinal")


def test_nonfinite_pairs_are_excluded():
    f = pd.Series(list(range(20)) + [np.inf, 22.])
    target = pd.Series(list(range(20)) + [21., np.inf])
    result = ic_spearman(f, target)
    assert result["n"] == 20
    assert result["ic"] == pytest.approx(1)
    assert np.isinf(f.iloc[-2])  # Pas de mutation de l'entrée.


@pytest.mark.parametrize("target", ["vol_fwd_5", "dd_fwd_5", "ru_fwd_5"])
def test_directional_hit_rejects_nonreturn_targets(target):
    result = hit_rate_directionnel(pd.Series([1.] * 20), pd.Series([.1] * 20),
                                  type_feature="directional", nom_cible=target)
    assert result["status"] == "not_applicable_target"
    assert result["n"] == 0
    assert np.isnan(result["hit_rate"])


def test_directional_hit_requires_explicit_signal_and_target():
    f, c = pd.Series([1.] * 20), pd.Series([.1] * 20)
    assert hit_rate_directionnel(f, c)["status"] == "not_applicable_non_directional"
    assert hit_rate_directionnel(f, c, type_feature="directional")["status"] == "not_applicable_target"


def test_abstentions_and_zero_returns_have_explicit_denominator():
    f = pd.Series([1.] * 10 + [-1.] * 10 + [0.] * 5 + [np.inf])
    c = pd.Series([.1] * 9 + [0.] + [-.1] * 10 + [.1] * 6, name="ret_fwd_1")
    result = hit_rate_directionnel(f, c, type_feature="directional")
    assert result["n"] == 20
    assert result["n_available"] == 25
    assert result["hit_rate"] == pytest.approx(19 / 20)


def test_boolean_direction_is_opt_in():
    f = pd.Series([True, False] * 10, dtype="boolean")
    c = pd.Series([.1, -.1] * 10, name="ret_fwd_1")
    assert np.isnan(hit_rate_directionnel(f, c)["hit_rate"])
    assert hit_rate_directionnel(f, c, type_feature="directional")["hit_rate"] == 1


def test_volatility_one_is_rejected_and_not_advertised():
    with pytest.raises(ValueError, match="K >= 2"):
        parser_cible("vol_fwd_1")
    with pytest.raises(ValueError, match="K >= 2"):
        calculer_cible(pd.DataFrame(), "vol_fwd_1")
    assert "vol_fwd_1" not in liste_cibles()
    assert "ret_fwd_1" in liste_cibles()


def test_study_exports_contract_and_actual_hit_counts(tmp_path):
    index = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
    study = Study("unused", "unused", tests_modalite=False,
                  feature_types={"signal": "directional"})
    study.df_features = pd.DataFrame({"adx__signal": [1.] * 20 + [0.] * 10}, index=index)
    study.df_cibles = pd.DataFrame({"ret_fwd_1": .1}, index=index)
    study._evaluer()
    study.save(tmp_path)
    row = pd.read_parquet(tmp_path / "study_results.parquet").iloc[0]
    assert row.n_test == 30
    assert row.n_hit_test == 20
    assert row.feature_type == "directional"
    assert row.hit_status_test == "ok"
    assert row.ic_status_test == "constant"
    assert "n_hit_test" in (tmp_path / "study_report.md").read_text()


def test_study_rejects_misspelled_feature_declaration():
    index = pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    study = Study("unused", "unused", feature_types={"typo": "directional"})
    study.df_features = pd.DataFrame({"signal": 1.}, index=index)
    study.df_cibles = pd.DataFrame({"ret_fwd_1": .1}, index=index)
    with pytest.raises(ValueError, match="absentes"):
        study._evaluer()
