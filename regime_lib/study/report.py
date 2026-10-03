"""
Génération du rapport d'étude au format Markdown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fmt_float(x: float, precision: int = 3) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:.{precision}f}"


def generer_rapport(
    resultats: pd.DataFrame,
    detail_conditionnel: dict | None = None,
    tests_modalite: dict | None = None,
    schema_split: str = "in_sample",
    noms_cibles: list[str] | None = None,
) -> str:
    """
    Produit un rapport Markdown à partir des résultats d'une étude.

    Parameters
    ----------
    resultats : pd.DataFrame
        Sortie de `Study.run()`.
    detail_conditionnel : dict | None
        Dict {(fold, feature, cible) → DataFrame conditionnel}.
    tests_modalite : dict | None
        Dict {(feature, cible) → DataFrame de tests statistiques par
        modalité}. Sortie de `Study.resultats_tests`.
    schema_split : str
        Nom du schéma de split, pour la section en-tête.
    noms_cibles : list[str] | None
        Cibles étudiées (pour le résumé).

    Returns
    -------
    str
        Contenu Markdown.
    """
    lignes = []
    A = lignes.append

    A("# Rapport d'étude")
    A("")
    A(f"- **Schéma de split** : `{schema_split}`")
    A(f"- **Nombre de lignes de résultat** : {len(resultats)}")
    if noms_cibles:
        A(f"- **Cibles** : {', '.join(noms_cibles)}")
    A("")

    # --- Vue synthétique par feature × cible (fold 0) ---
    A("## Vue synthétique (fold 0)")
    A("")
    fold0 = resultats[resultats["fold"] == 0].copy()
    if not fold0.empty:
        tab = fold0[[
            "feature", "cible", "n_test",
            "ic_test", "pvalue_test", "hit_test",
        ]].copy()
        tab = tab.sort_values(
            ["cible", "ic_test"],
            key=lambda s: s.abs() if s.name == "ic_test" else s,
            ascending=[True, False],
        )
        A(_to_markdown(tab))
    A("")

    # --- Stabilité inter-folds (si plusieurs folds) ---
    n_folds = resultats["fold"].nunique()
    if n_folds > 1:
        A(f"## Stabilité inter-folds ({n_folds} folds)")
        A("")
        stab = resultats.groupby(["feature", "cible"]).agg(
            ic_mean=("ic_test", "mean"),
            ic_std=("ic_test", "std"),
            ic_min=("ic_test", "min"),
            ic_max=("ic_test", "max"),
            hit_mean=("hit_test", "mean"),
        ).round(3)
        A(_to_markdown(stab.reset_index()))
        A("")

    # --- Détail conditionnel (fold 0, meilleur IC) ---
    if detail_conditionnel:
        fold0_res = resultats[resultats["fold"] == 0]
        if not fold0_res.empty and fold0_res["ic_test"].notna().any():
            best = fold0_res.loc[
                fold0_res["ic_test"].abs().idxmax()
            ]
            cle = (0, best["feature"], best["cible"])
            if cle in detail_conditionnel:
                A("## Détail conditionnel (fold 0, meilleur IC)")
                A("")
                A(f"Feature : `{best['feature']}`  ")
                A(f"Cible : `{best['cible']}`  ")
                A(f"IC = {_fmt_float(best['ic_test'])} "
                  f"(p = {_fmt_float(best['pvalue_test'], 4)})")
                A("")
                A(_to_markdown(detail_conditionnel[cle].reset_index()))
                A("")

    # --- Tests statistiques par modalité (fold 0) ---
    if tests_modalite:
        A("## Tests statistiques par modalité (fold 0, test)")
        A("")
        A("Trois tests appliqués one-vs-rest pour chaque modalité :")
        A("")
        A("- **Welch** (`welch_p`) : comparaison des moyennes")
        A("- **Mann-Whitney** (`mw_p`) : comparaison des distributions")
        A("- **z de proportions** (`z_p`) : comparaison des fréquences "
          "de cibles positives")
        A("")
        A("Seuils usuels : `p < 0.05` significatif, `p < 0.001` très "
          "significatif.")
        A("")

        for (feat, cible), tab in tests_modalite.items():
            if tab is None or tab.empty:
                continue
            A(f"### `{feat}` × `{cible}`")
            A("")
            # Colonnes affichées : n, moyennes, proportions positives,
            # p-values des trois tests
            cols_affichage = [
                "n_a", "n_b",
                "mean_a", "mean_b",
                "p_a_pos", "p_b_pos",
                "welch_p", "mw_p", "z_p",
            ]
            cols_presentes = [c for c in cols_affichage if c in tab.columns]
            tab_aff = tab[cols_presentes].copy()
            A(_to_markdown(tab_aff.reset_index()))
            A("")

    A("---")
    A("")
    A("*Rapport généré automatiquement par `regime_lib.study.report`.*")
    return "\n".join(lignes)


def _to_markdown(df: pd.DataFrame) -> str:
    """Rend un DataFrame en tableau Markdown."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind == "f":
            df[col] = df[col].map(lambda x: _fmt_float(x))
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, sep] + rows)
