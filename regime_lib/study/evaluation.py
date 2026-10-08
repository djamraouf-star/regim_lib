"""
Métriques et tests statistiques d'évaluation.

Métriques globales
------------------
- `ic_spearman` : corrélation de rang feature ↔ cible
- `hit_rate_directionnel` : % de bonnes directions (signal déclaré, cible rendement)

Tests statistiques (batterie à 3 tests)
---------------------------------------
Pour comparer la distribution d'une cible entre deux groupes :

- `welch_test` : t-test de Welch (moyennes, variances inégales)
- `mann_whitney_test` : Mann-Whitney U (distributions)
- `z_proportions_test` : z de proportions (fréquences de succès)

Ces trois tests sont agrégés dans `batterie_tests` et appliqués par
`par_modalite_tests` pour chaque valeur d'une feature catégorielle,
comparée au reste.

Convention "succès" pour le test z
----------------------------------
La nature du "succès" dépend du type de cible :

- `ret_fwd_K` : cible > 0 (retour positif)
- `vol_fwd_K` : cible > médiane (régime agité vs calme)
- `dd_fwd_K`  : cible > médiane (excursion négative moins profonde)
- `ru_fwd_K`  : cible > médiane (run-up fort = succès)

Sans cette adaptation, le test z est trivial sur les cibles non
signées (volatilité, drawdown, run-up).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, spearmanr


# ---------------------------------------------------------------------------
# Convention de succès pour le test z
# ---------------------------------------------------------------------------
def _convention_succes(
    cible: pd.Series,
    nom_cible: str,
    mediane: float | None = None,
) -> np.ndarray:
    """
    Détermine le masque "succès" d'une cible selon son type.

    Parameters
    ----------
    cible : pd.Series
        Série de valeurs (sous-échantillon).
    nom_cible : str
        Nom de la cible (ex. 'ret_fwd_5', 'vol_fwd_20').
    mediane : float | None
        Médiane globale à utiliser comme seuil pour les cibles non
        signées. Si None, utilise la médiane de `cible` (moins
        pertinent pour comparer deux groupes).
    """
    if nom_cible.startswith("ret_fwd_"):
        return (cible > 0).to_numpy()

    seuil = mediane if mediane is not None else cible.median()

    if nom_cible.startswith("vol_fwd_"):
        return (cible > seuil).to_numpy()
    if nom_cible.startswith("dd_fwd_"):
        return (cible > seuil).to_numpy()
    if nom_cible.startswith("ru_fwd_"):
        return (cible > seuil).to_numpy()

    from regime_lib.study.targets import parser_cible
    parser_cible(nom_cible)
    # Pour les excursions et durées : événement « supérieur à la médiane »,
    # sans interprétation de performance pour les durées.
    return (cible > seuil).to_numpy()

# ---------------------------------------------------------------------------
# Métriques globales
# ---------------------------------------------------------------------------
FEATURE_TYPES = {"nominal", "ordinal", "continuous", "directional"}


def feature_type(feature: pd.Series, declared: str | None = None) -> str:
    """Type explicite prioritaire ; aucun signal directionnel n'est inféré."""
    if declared is not None:
        if declared not in FEATURE_TYPES:
            raise ValueError(f"Type de feature inconnu : {declared!r}.")
        return declared
    name = str(feature.name).split("__")[-1]
    if name == "regime" or name.startswith("ctx_") or pd.api.types.is_bool_dtype(feature):
        return "nominal"
    if isinstance(feature.dtype, pd.CategoricalDtype):
        return "ordinal" if feature.cat.ordered else "nominal"
    return "continuous" if pd.api.types.is_numeric_dtype(feature) else "nominal"


def _metric_pairs(feature: pd.Series, cible: pd.Series) -> pd.DataFrame:
    df = pd.concat([feature, cible], axis=1)
    df.columns = ["feature", "cible"]
    for column in df:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].where(np.isfinite(df[column]))
    return df.dropna()


def ic_spearman(feature: pd.Series, cible: pd.Series, *,
                type_feature: str | None = None) -> dict:
    """Spearman pour variables ordonnées, avec effectif et statut explicites."""
    kind = feature_type(feature, type_feature)
    df = _metric_pairs(feature, cible)
    n = len(df)
    result = {"ic": np.nan, "pvalue": np.nan, "n": n, "status": "ok"}
    if kind == "nominal":
        return {**result, "status": "not_applicable_nominal"}
    f, c = df.feature, df.cible
    if isinstance(f.dtype, pd.CategoricalDtype) and f.cat.ordered and kind == "ordinal":
        f = f.cat.codes
    elif not pd.api.types.is_numeric_dtype(f):
        raise ValueError("Une feature ordonnée doit être numérique ou une catégorie explicitement ordonnée.")
    if n < 10:
        return {**result, "status": "insufficient_samples"}
    if f.nunique() < 2 or c.nunique() < 2:
        return {**result, "status": "constant"}
    rho, p = spearmanr(f, c)
    return {**result, "ic": float(rho), "pvalue": float(p)}


def hit_rate_directionnel(feature: pd.Series, cible: pd.Series, *,
                          type_feature: str | None = None,
                          nom_cible: str | None = None) -> dict:
    """Signal explicitement directionnel et rendement futur uniquement.

    Signal numérique signé : zéro = abstention. Booléen : True = hausse,
    False = baisse. Un rendement nul compte comme une direction incorrecte.
    """
    from regime_lib.study.targets import parser_cible

    kind = feature_type(feature, type_feature)
    df = _metric_pairs(feature, cible)
    result = {"hit_rate": np.nan, "n": 0, "n_available": len(df), "status": "ok"}
    if kind != "directional":
        return {**result, "status": "not_applicable_non_directional"}
    target = nom_cible or cible.name
    if not isinstance(target, str):
        return {**result, "status": "not_applicable_target"}
    family, _ = parser_cible(target)
    if family != "ret":
        return {**result, "status": "not_applicable_target"}
    f, c = df.feature, df.cible
    if pd.api.types.is_bool_dtype(f):
        pred = f.astype(int).replace({0: -1})
    elif pd.api.types.is_numeric_dtype(f):
        pred = np.sign(f)
    else:
        raise ValueError("Un signal directionnel doit être numérique ou booléen.")
    active = pred.ne(0)
    result["n"] = int(active.sum())
    if result["n"] < 10:
        return {**result, "status": "insufficient_samples"}
    return {**result, "hit_rate": float((np.sign(c[active]) == pred[active]).mean())}


# ---------------------------------------------------------------------------
# Tests statistiques individuels
# ---------------------------------------------------------------------------
def welch_test(a: np.ndarray, b: np.ndarray) -> dict:
    """
    t-test de Welch : compare les moyennes de deux échantillons
    indépendants avec variances inégales.

    Parameters
    ----------
    a, b : np.ndarray
        Échantillons (les NaN doivent être filtrés en amont).

    Returns
    -------
    dict
        Clés : `t_stat`, `pvalue`, `n_a`, `n_b`, `mean_a`, `mean_b`,
        `std_a`, `std_b`. `nan` si un échantillon a moins de 2 points.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return {
            "t_stat": np.nan, "pvalue": np.nan,
            "n_a": len(a), "n_b": len(b),
            "mean_a": np.nan, "mean_b": np.nan,
            "std_a": np.nan, "std_b": np.nan,
        }

    # Center both samples around one shared reference before computing
    # moments. This avoids loss of precision when values have a large common
    # offset but very small between/within-group differences.
    reference = a[0]
    a_centered = a - reference
    b_centered = b - reference
    mean_a_centered = float(np.mean(a_centered))
    mean_b_centered = float(np.mean(b_centered))
    mean_difference = mean_a_centered - mean_b_centered
    var_a = float(np.var(a_centered, ddof=1))
    var_b = float(np.var(b_centered, ddof=1))
    variance_a = var_a / len(a)
    variance_b = var_b / len(b)
    standard_error_squared = variance_a + variance_b

    if standard_error_squared == 0.0:
        if mean_difference == 0.0:
            t, p = 0.0, 1.0
        else:
            t, p = np.copysign(np.inf, mean_difference), 0.0
    else:
        degrees_of_freedom = standard_error_squared**2 / (
            variance_a**2 / (len(a) - 1)
            + variance_b**2 / (len(b) - 1)
        )
        t = mean_difference / np.sqrt(standard_error_squared)
        p = 2.0 * stats.t.sf(abs(t), degrees_of_freedom)

    return {
        "t_stat": float(t),
        "pvalue": float(p),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "std_a": float(np.std(a, ddof=1)),
        "std_b": float(np.std(b, ddof=1)),
    }


def mann_whitney_test(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Mann-Whitney U : test non-paramétrique de décalage de distribution.

    Returns
    -------
    dict
        Clés : `u_stat`, `pvalue`, `n_a`, `n_b`.
    """
    if len(a) < 2 or len(b) < 2:
        return {
            "u_stat": np.nan, "pvalue": np.nan,
            "n_a": len(a), "n_b": len(b),
        }
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "u_stat": float(u),
        "pvalue": float(p),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
    }


def z_proportions_test(
    n_a: int, k_a: int, n_b: int, k_b: int,
) -> dict:
    """
    Test z de proportions à deux échantillons.

    Parameters
    ----------
    n_a, n_b : int
        Effectifs totaux des deux groupes.
    k_a, k_b : int
        Nombre de "succès" dans chaque groupe.

    Returns
    -------
    dict
        Clés : `z_stat`, `pvalue`, `p_a`, `p_b`, `p_pool`,
        `n_a`, `n_b`.
    """
    if n_a < 1 or n_b < 1:
        return {
            "z_stat": np.nan, "pvalue": np.nan,
            "p_a": np.nan, "p_b": np.nan, "p_pool": np.nan,
            "n_a": n_a, "n_b": n_b,
        }

    p_a = k_a / n_a
    p_b = k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)

    denom = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if denom == 0:
        return {
            "z_stat": 0.0, "pvalue": 1.0,
            "p_a": p_a, "p_b": p_b, "p_pool": p_pool,
            "n_a": n_a, "n_b": n_b,
        }

    z = (p_a - p_b) / denom
    pvalue = 2.0 * (1.0 - norm.cdf(abs(z)))
    return {
        "z_stat": float(z),
        "pvalue": float(pvalue),
        "p_a": float(p_a),
        "p_b": float(p_b),
        "p_pool": float(p_pool),
        "n_a": int(n_a),
        "n_b": int(n_b),
    }


# ---------------------------------------------------------------------------
# Batterie et application par modalité
# ---------------------------------------------------------------------------
def batterie_tests(
    a: np.ndarray,
    b: np.ndarray,
    nom_cible: str = "ret_fwd_5",
) -> dict:
    """
    Applique les trois tests à deux échantillons.

    La convention "succès" pour le test z dépend du type de cible
    (voir `_convention_succes`).

    Parameters
    ----------
    a, b : np.ndarray
        Échantillons.
    nom_cible : str
        Nom de la cible, pour déterminer la convention de succès.

    Returns
    -------
    dict
        Agrégation des trois tests, avec préfixes :
          - `welch_*`, `mw_*`, `z_*`
          - `n_a`, `n_b`, `mean_a`, `mean_b`, `p_a_pos`, `p_b_pos`
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    res_welch = welch_test(a, b)
    res_mw = mann_whitney_test(a, b)

    # Succès selon la nature de la cible.
    # La médiane (pour les cibles non signées) est calculée sur
    # l'union a ∪ b, sinon les deux groupes auraient mécaniquement
    # 50 % de succès chacun.
    union = pd.concat([pd.Series(a), pd.Series(b)])
    mediane_globale = float(union.median())

    s_a = _convention_succes(pd.Series(a), nom_cible, mediane=mediane_globale)
    s_b = _convention_succes(pd.Series(b), nom_cible, mediane=mediane_globale)
    k_a = int(s_a.sum())
    k_b = int(s_b.sum())
    res_z = z_proportions_test(len(a), k_a, len(b), k_b)
    
    return {
        "n_a": res_welch["n_a"],
        "n_b": res_welch["n_b"],
        "mean_a": res_welch["mean_a"],
        "mean_b": res_welch["mean_b"],
        "std_a": res_welch["std_a"],
        "std_b": res_welch["std_b"],
        "p_a_pos": res_z["p_a"],
        "p_b_pos": res_z["p_b"],
        "welch_t": res_welch["t_stat"],
        "welch_p": res_welch["pvalue"],
        "mw_u": res_mw["u_stat"],
        "mw_p": res_mw["pvalue"],
        "z_stat": res_z["z_stat"],
        "z_p": res_z["pvalue"],
    }


def _episode_count(mask: pd.Series, timeframe: str | None = None,
                   expected_index: pd.DatetimeIndex | None = None) -> int:
    """Compte les séquences valides sans raccorder les trous ou valeurs exclues."""
    adjacent = pd.Series(True, index=mask.index)
    if isinstance(mask.index, pd.DatetimeIndex) and len(mask) > 1:
        from regime_lib.core.temporal import TIMEFRAME_SECONDS
        if expected_index is not None:
            positions = expected_index.get_indexer(mask.index)
            if (positions < 0).any():
                raise ValueError("Observation absente du calendrier attendu.")
            adjacent.iloc[1:] = np.diff(positions) == 1
        elif timeframe is not None:
            adjacent.iloc[1:] = (mask.index[1:] - mask.index[:-1] ==
                                  pd.Timedelta(seconds=TIMEFRAME_SECONDS[timeframe]))
        else:
            # Sans fréquence déclarée, seule une grille régulière est inférable.
            deltas = mask.index[1:] - mask.index[:-1]
            if not (deltas == deltas[0]).all():
                raise ValueError("Déclarer timeframe pour compter les épisodes sur un index irrégulier.")
    previous = mask.shift(1, fill_value=False)
    return int((mask & ~(previous & adjacent)).sum())


def par_modalite_tests(
    feature: pd.Series,
    cible: pd.Series,
    nom_cible: str = "ret_fwd_5",
    *, timeframe: str | None = None, expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """
    Pour chaque modalité d'une feature, compare la cible dans cette
    modalité vs le reste (one-vs-rest).

    Parameters
    ----------
    feature : pd.Series
        Feature catégorielle (ou numérique discrète).
    cible : pd.Series
        Cible à comparer.
    nom_cible : str
        Nom de la cible (pour la convention de succès du test z).

    Returns
    -------
    pd.DataFrame
        Indexé par modalité de la feature. Colonnes : n_a, n_b,
        mean_a, mean_b, std_a, std_b, p_a_pos, p_b_pos, welch_t,
        welch_p, mw_u, mw_p, z_stat, z_p.
    """
    df = _metric_pairs(feature, cible)
    if df.empty:
        return pd.DataFrame()

    modalites = df["feature"].unique()
    lignes = []

    for m in modalites:
        a = df.loc[df["feature"] == m, "cible"].to_numpy()
        b = df.loc[df["feature"] != m, "cible"].to_numpy()
        res = batterie_tests(a, b, nom_cible=nom_cible)
        own = feature.eq(m).fillna(False) & feature.index.isin(df.index)
        rest = feature.ne(m).fillna(False) & feature.index.isin(df.index)
        res["n_episodes_a"] = _episode_count(own, timeframe, expected_index)
        res["n_episodes_b"] = _episode_count(rest, timeframe, expected_index)
        res["mean_diff"] = float(a.mean() - b.mean()) if len(a) and len(b) else np.nan
        res["median_diff"] = float(np.median(a) - np.median(b)) if len(a) and len(b) else np.nan
        res["cliffs_delta"] = (2 * res["mw_u"] / (len(a) * len(b)) - 1
                                if len(a) and len(b) else np.nan)
        res["status"] = ("insufficient_samples" if min(len(a), len(b)) < 2 else
                         "insufficient_episodes" if min(res["n_episodes_a"], res["n_episodes_b"]) < 2
                         else "descriptive")
        res["modalite"] = m
        lignes.append(res)

    out = pd.DataFrame(lignes).set_index("modalite")
    return out.sort_index()


# ---------------------------------------------------------------------------
# Statistiques descriptives (conservé)
# ---------------------------------------------------------------------------
def stats_conditionnelles(
    feature: pd.Series, cible: pd.Series, *,
    timeframe: str | None = None, expected_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Statistiques descriptives de la cible par valeur de feature."""
    df = _metric_pairs(feature, cible)
    if df.empty:
        return pd.DataFrame()

    g = df.groupby("feature")["cible"]
    return pd.DataFrame({
        "n": g.size(),
        "n_episodes": pd.Series({m: _episode_count(
            feature.eq(m).fillna(False) & feature.index.isin(df.index), timeframe, expected_index
        ) for m in g.groups}),
        "mean": g.mean(),
        "median": g.median(),
        "std": g.std(),
        "min": g.min(),
        "max": g.max(),
    }).sort_index()


def stabilite_modalites(tests: dict, n_folds: int) -> pd.DataFrame:
    """Dispersion non pondérée des effets par modalité, sans agréger les p-values."""
    rows = []
    for (fold, feature, cible), table in tests.items():
        for modalite, row in table.iterrows():
            rows.append({"fold": fold, "feature": feature, "cible": cible,
                         "modalite": str(modalite), "effect": row["mean_diff"],
                         "status": row["status"]})
    columns = ["feature", "cible", "modalite", "n_folds_total", "n_folds_present",
               "n_folds_estimable", "n_folds_sufficient", "effect_mean", "effect_std",
               "effect_min", "effect_max", "n_positive", "n_negative", "n_zero", "sign_stability"]
    if not rows:
        return pd.DataFrame(columns=columns)
    records = []
    for key, group in pd.DataFrame(rows).groupby(["feature", "cible", "modalite"]):
        effects = group.effect.dropna()
        positive, negative, zero = (int((effects > 0).sum()), int((effects < 0).sum()), int((effects == 0).sum()))
        records.append(dict(zip(columns, [*key, n_folds, len(group), len(effects),
            int(group.status.eq("descriptive").sum()), effects.mean(), effects.std(),
            effects.min(), effects.max(), positive, negative, zero,
            max(positive, negative) / len(effects) if len(effects) else np.nan])))
    return pd.DataFrame(records, columns=columns)
