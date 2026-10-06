"""Inférence exploratoire sur les écarts de moyenne, par blocs temporels.

Bootstrap de paires en blocs mobiles ; intervalle basic et test bilatéral
centré. Hypothèse : dépendance faible et distribution suffisamment stable
sur chaque fold. Les intervalles sont ponctuels, non simultanés.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
from numbers import Integral

import numpy as np
import pandas as pd

from regime_lib.core.temporal import TIMEFRAME_SECONDS, resolve_timeframe
from regime_lib.utils.validation import validate_time_index


@dataclass(frozen=True)
class InferenceConfig:
    block_size: int
    n_resamples: int = 999
    confidence_level: float = .95
    seed: int = 0
    min_group_size: int = 10

    def __post_init__(self):
        for name, lower in (("block_size", 1), ("n_resamples", 199),
                            ("seed", 0), ("min_group_size", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
                raise ValueError(f"{name} doit être un entier >= {lower}.")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level doit être dans (0, 1).")

    def to_dict(self):
        values = asdict(self)
        for field in ("block_size", "n_resamples", "seed", "min_group_size"):
            values[field] = int(values[field])
        return {**values, "method": "moving_block_pairs_basic",
                "multiple_testing": "benjamini_yekutieli",
                "family": "all_fold_feature_target_modality_mean_differences"}


def block_indices(index, valid, block_size, *, timeframe=None, expected_index=None):
    """Blocs complets ne traversant ni trous temporels ni paires invalides."""
    validate_time_index(index)
    if expected_index is not None:
        validate_time_index(expected_index)
        positions = expected_index.get_indexer(index)
        if (positions < 0).any():
            raise ValueError("Observations absentes du calendrier attendu.")
        adjacent = np.diff(positions) == 1
    elif len(index) > 1:
        frame = pd.DataFrame(index=index)
        tf = resolve_timeframe(frame, timeframe)
        adjacent = np.asarray(index[1:] - index[:-1] == pd.Timedelta(seconds=TIMEFRAME_SECONDS[tf]))
    else:
        adjacent = np.empty(0, dtype=bool)
    segments = []
    current = []
    for i, usable in enumerate(valid):
        if not usable or (current and not adjacent[i - 1]):
            if current:
                segments.append(current)
            current = []
        if usable:
            current.append(i)
    if current:
        segments.append(current)
    # Ne pas exclure silencieusement un segment court du support estimé.
    if not segments or any(len(segment) < block_size for segment in segments):
        return np.empty((0, block_size), dtype=int)
    return np.asarray([segment[start:start + block_size] for segment in segments
                       for start in range(len(segment) - block_size + 1)], dtype=int)


def bootstrap_modalities(feature, target, table, config, *, key,
                         horizon, timeframe=None, expected_index=None):
    """Ajoute une incertitude à mean_diff, sans modifier les tests historiques."""
    if not feature.index.equals(target.index):
        raise ValueError("Alignement strict requis pour le bootstrap.")
    if config.block_size < horizon:
        raise ValueError("block_size doit être >= à l'horizon de chaque cible.")
    result = table.copy()
    for name in ("mean_diff_ci_low", "mean_diff_ci_high", "block_p", "block_mc_se"):
        result[name] = np.nan
    result["inference_status"] = "insufficient_support"
    result["bootstrap_valid"] = 0
    result["block_size"] = config.block_size
    result["n_resamples"] = config.n_resamples
    result["confidence_level"] = config.confidence_level
    result["seed"] = config.seed
    if result.empty:
        return result
    valid = feature.notna().to_numpy() & np.isfinite(target.to_numpy(dtype=float))
    if pd.api.types.is_numeric_dtype(feature):
        valid &= np.isfinite(feature.to_numpy(dtype=float))
    blocks = block_indices(feature.index, valid, config.block_size,
                           timeframe=timeframe, expected_index=expected_index)
    n = int(valid.sum())
    if not len(blocks) or n < 4 * config.block_size:
        return result
    y = target.to_numpy(dtype=float)
    for modality, row in table.iterrows():
        if min(row.n_a, row.n_b) < config.min_group_size or min(row.n_episodes_a, row.n_episodes_b) < 2:
            continue
        member = feature.eq(modality).fillna(False).to_numpy(dtype=bool)
        digest = sha256(repr((config.seed, key, modality)).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], "little"))
        estimates = []
        for _ in range(config.n_resamples):
            sampled = blocks[rng.integers(len(blocks), size=int(np.ceil(n / config.block_size)))].ravel()[:n]
            own = member[sampled]
            if min(int(own.sum()), int((~own).sum())) < 2:
                continue
            estimates.append(y[sampled][own].mean() - y[sampled][~own].mean())
        result.loc[modality, "bootstrap_valid"] = len(estimates)
        if len(estimates) < .9 * config.n_resamples:
            result.loc[modality, "inference_status"] = "insufficient_resamples"
            continue
        estimates = np.asarray(estimates)
        if np.ptp(estimates) == 0:
            result.loc[modality, "inference_status"] = "degenerate_bootstrap"
            continue
        alpha = 1 - config.confidence_level
        lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
        effect = row.mean_diff
        p = (1 + np.count_nonzero(np.abs(estimates - effect) >= abs(effect))) / (len(estimates) + 1)
        result.loc[modality, ["mean_diff_ci_low", "mean_diff_ci_high", "block_p", "block_mc_se"]] = [
            2 * effect - upper, 2 * effect - lower, p, np.sqrt(p * (1 - p) / len(estimates))]
        result.loc[modality, "inference_status"] = "ok"
    return result


def adjust_by(pvalues):
    """BY sur toute la famille ; les hypothèses non estimables comptent comme p=1."""
    values = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(values)
    if ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError("Les p-values doivent être dans [0, 1].")
    n = len(values)
    if not n:
        return values.copy()
    filled = np.where(finite, values, 1.)
    order = np.argsort(filled, kind="stable")
    raw = filled[order] * n * np.sum(1 / np.arange(1, n + 1)) / np.arange(1, n + 1)
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(1., np.minimum.accumulate(raw[::-1])[::-1])
    adjusted[~finite] = np.nan
    return adjusted


def correct_family(tables, confidence_level):
    """Une famille globale de tests bootstrap sur tous les folds et modalités."""
    entries = [(key, modality) for key, table in tables.items() for modality in table.index]
    p = [tables[key].loc[modality, "block_p"] for key, modality in entries]
    for table in tables.values():
        table["block_p_by"] = np.nan
        table["reject_by"] = pd.Series(pd.NA, index=table.index, dtype="boolean")
        table["family_size"] = len(entries)
    for (key, modality), adjusted in zip(entries, adjust_by(p)):
        tables[key].loc[modality, "block_p_by"] = adjusted
        if np.isfinite(adjusted):
            tables[key].loc[modality, "reject_by"] = adjusted <= 1 - confidence_level


def block_sensitivity(feature, target, *, block_sizes, horizon, config,
                      timeframe=None, expected_index=None):
    """Diagnostic pré-spécifié de sensibilité, sans sélectionner une taille optimale.

    Une famille BY couvre toutes les tailles et modalités de ce diagnostic.
    Ne remplace pas la famille globale de Study, ni une confirmation indépendante.
    """
    from dataclasses import replace
    from regime_lib.study.evaluation import par_modalite_tests
    sizes = list(block_sizes)
    if len(sizes) < 2 or len(set(sizes)) != len(sizes):
        raise ValueError("Déclarer au moins deux tailles de blocs distinctes.")
    tables = {}
    table = par_modalite_tests(feature, target, timeframe=timeframe, expected_index=expected_index)
    for size in sizes:
        candidate = replace(config, block_size=size)
        tables[size] = bootstrap_modalities(feature, target, table, candidate,
            key=("sensitivity", size), horizon=horizon,
            timeframe=timeframe, expected_index=expected_index)
    correct_family(tables, config.confidence_level)
    result = pd.concat(tables, names=["sensitivity_block_size", "modality"])
    result.attrs["purpose"] = "diagnostic_only_no_block_selection"
    return result
