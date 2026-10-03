from regime_lib.config import load_profile
from regime_lib.methods.entropy.shannon import ShannonEntropyDetector
from regime_lib.methods.vector.pca_axis import PCAAxisDetector


def test_default_profile_exposes_vector_and_entropy_parameters():
    profile = load_profile("default")

    assert profile.methodes["pca_axis"]["timeframes"]["H1"]["fenetre"] == 100
    assert profile.methodes["pca_axis"]["seuil_eigen_ratio"] == 0.65
    assert profile.methodes["shannon"]["timeframes"]["H1"]["fenetre"] == 100
    assert profile.methodes["minkowski_causal"]["q_c"] == 0.90


def test_custom_profile_deep_merges_and_uses_method_identifiers():
    profile = load_profile("eurusd")

    assert profile.methodes["adx"]["seuil_range"] == 22.0
    assert profile.methodes["pca_axis"]["timeframes"]["H1"]["fenetre"] == 100
    assert profile.methodes["shannon"]["timeframes"]["H1"]["fenetre"] == 100
    assert profile.methodes["shannon"]["seuil_ordonne"] == 0.3

    assert PCAAxisDetector(
        timeframe="H1", profile=profile.methodes["pca_axis"]
    ).fenetre == 100
    assert ShannonEntropyDetector(
        timeframe="H1", profile=profile.methodes["shannon"]
    ).seuil_ordonne == 0.3
