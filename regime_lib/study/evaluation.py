"""
Métriques et tests statistiques d'évaluation.

Métriques globales
------------------
- `ic_spearman` : corrélation de rang feature ↔ cible
- `hit_rate_directionnel` : % de bonnes directions

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
- `dd_fwd_K`  : cible < médiane (drawdown faible = succès)
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
        return (cible < seuil).to_numpy()
    if nom_cible.startswith("ru_fwd_"):
        return (cible > seuil).to_numpy()

    return (cible > 0).to_numpy()

# ---------------------------------------------------------------------------
# Métriques globales
# ---------------------------------------------------------------------------
def ic_spearman(feature: pd.Series, cible: pd.Series) -> dict:
    """
    Information Coefficient : corrélation de rang Spearman.

    Returns
    -------
    dict
        Clés : `ic`, `pvalue`, `n`. Si moins de 10 points valides,
        `ic = nan`.
    """
    df = pd.concat([feature, cible], axis=1).dropna()
    n = len(df)
    if n < 10:
        return {"ic": np.nan, "pvalue": np.nan, "n": n}

    f = df.iloc[:, 0]
    c = df.iloc[:, 1]

    if not pd.api.types.is_numeric_dtype(f):
        f = f.astype("category").cat.codes
        if (f < 0).any():
            return {"ic": np.nan, "pvalue": np.nan, "n": n}

    if f.nunique() < 2 or c.nunique() < 2:
        return {"ic": np.nan, "pvalue": np.nan, "n": n}

    rho, p = spearmanr(f, c)
    return {"ic": float(rho), "pvalue": float(p), "n": int(n)}


def hit_rate_directionnel(feature: pd.Series, cible: pd.Series) -> dict:
    """
    Taux de bonnes prédictions directionnelles.

    Pour une feature numérique ou booléenne (signal directionnel).
    Renvoie `nan` pour les features catégorielles.
    """
    df = pd.concat([feature, cible], axis=1).dropna()
    n = len(df)
    if n < 10:
        return {"hit_rate": np.nan, "n": n}

    f = df.iloc[:, 0]
    c = df.iloc[:, 1]

    if pd.api.types.is_bool_dtype(f):
        pred = f.astype(int).replace({1: 1, 0: -1})
    elif pd.api.types.is_numeric_dtype(f):
        pred = np.sign(f)
    else:
        return {"hit_rate": np.nan, "n": n}

    masque = pred != 0
    if masque.sum() < 10:
        return {"hit_rate": np.nan, "n": int(masque.sum())}

    correct = (np.sign(c[masque]) == pred[masque]).mean()
    return {"hit_rate": float(correct), "n": int(masque.sum())}


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


def par_modalite_tests(
    feature: pd.Series,
    cible: pd.Series,
    nom_cible: str = "ret_fwd_5",
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
    df = pd.concat([feature, cible], axis=1).dropna()
    df.columns = ["feature", "cible"]
    if df.empty:
        return pd.DataFrame()

    modalites = df["feature"].unique()
    lignes = []

    for m in modalites:
        a = df.loc[df["feature"] == m, "cible"].to_numpy()
        b = df.loc[df["feature"] != m, "cible"].to_numpy()
        res = batterie_tests(a, b, nom_cible=nom_cible)
        res["modalite"] = m
        lignes.append(res)

    out = pd.DataFrame(lignes).set_index("modalite")
    return out.sort_index()


# ---------------------------------------------------------------------------
# Statistiques descriptives (conservé)
# ---------------------------------------------------------------------------
def stats_conditionnelles(
    feature: pd.Series, cible: pd.Series,
) -> pd.DataFrame:
    """Statistiques descriptives de la cible par valeur de feature."""
    df = pd.concat([feature, cible], axis=1).dropna()
    df.columns = ["feature", "cible"]
    if df.empty:
        return pd.DataFrame()

    g = df.groupby("feature")["cible"]
    return pd.DataFrame({
        "n": g.size(),
        "mean": g.mean(),
        "median": g.median(),
        "std": g.std(),
        "min": g.min(),
        "max": g.max(),
    }).sort_index()
