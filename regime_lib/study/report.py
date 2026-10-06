"""
Rapports du module study.

``generer_rapport`` produit le rapport Markdown d'une étude prédictive.
``rapport_complet`` produit le rapport descriptif des runs, avec leurs labels originaux.

Usage CLI : ``python -m regime_lib.study.report --help`` ou ``regime-report``.
Usage bibliothèque : ``from regime_lib.study.report import rapport_complet``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lib.core.data_loader import load_parquet
from regime_lib.core.temporal import TIMEFRAME_SECONDS, resolve_timeframe
from regime_lib.study.validation import prepare_regimes, prepare_prices
from regime_lib.utils.validation import validate_time_index, validity_mask


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
    stabilite: pd.DataFrame | None = None,
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
        Dict {(fold, feature, cible) → DataFrame de tests statistiques par
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
    A("Exploration sur les mêmes données." if schema_split == "in_sample" else
      "Évaluation temporelle après purge ; indépendance finale à vérifier dans le protocole.")
    A("")
    A(f"- **Schéma de split** : `{schema_split}`")
    A(f"- **Nombre de lignes de résultat** : {len(resultats)}")
    if noms_cibles:
        A(f"- **Cibles** : {', '.join(noms_cibles)}")
    A("")

    # --- Vue synthétique par feature × cible (fold 0) ---
    A("## Vue synthétique — tous les folds")
    A("")
    if resultats.empty:
        A("Aucun résultat disponible ; aucune estimation.")
        return "\n".join(lignes)
    fold0 = resultats.copy()
    if not fold0.empty:
        tab = fold0[[
            "fold", "feature", "cible", "n_test",
            "ic_test", "pvalue_test", "hit_test",
        ]].copy()
        for column in ("feature_type", "ic_status_test", "hit_status_test", "n_hit_test"):
            if column in fold0:
                tab[column] = fold0[column]
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

    if detail_conditionnel:
        A("## Détails conditionnels — tous les folds")
        for (fold, feat, cible), table in detail_conditionnel.items():
            A(f"### Fold {fold} — `{feat}` × `{cible}`")
            A(_to_markdown(table.reset_index()) if not table.empty else "Support valide vide.")
            A("")
    if stabilite is not None and not stabilite.empty:
        A("## Stabilité des effets par modalité")
        A("Écarts de moyenne modalité moins reste ; dispersion non pondérée entre folds.")
        A(_to_markdown(stabilite))
        A("")

    # --- Tests statistiques par modalité (fold 0) ---
    if tests_modalite:
        A("## Tests statistiques par modalité (tous les folds, test)")
        A("")
        A("Trois tests appliqués one-vs-rest pour chaque modalité :")
        A("")
        A("- **Welch** (`welch_p`) : comparaison des moyennes")
        A("- **Mann-Whitney** (`mw_p`) : comparaison des distributions")
        A("- **z de proportions** (`z_p`) : comparaison des fréquences "
          "de succès selon la famille de cible")
        A("")
        A("Welch, Mann-Whitney et z : p-values descriptives non corrigées. "
          "Si activée, l’inférence sur mean_diff utilise des blocs mobiles ; "
          "block_p_by corrige la famille globale par Benjamini–Yekutieli. "
          "Les intervalles sont ponctuels, non simultanés, et supposent une dépendance faible "
          "et une distribution suffisamment stable dans chaque fold.")
        A("")

        for key, tab in tests_modalite.items():
            fold, feat, cible = key if len(key) == 3 else (0, *key)
            if tab is None or tab.empty:
                continue
            A(f"### Fold {fold} — `{feat}` × `{cible}`")
            A("")
            # Colonnes affichées : n, moyennes, proportions positives,
            # p-values des trois tests
            cols_affichage = [
                "n_a", "n_b", "n_episodes_a", "n_episodes_b", "status",
                "mean_diff", "median_diff", "cliffs_delta",
                "mean_diff_ci_low", "mean_diff_ci_high", "inference_status",
                "block_p", "block_p_by", "reject_by", "family_size",
                "bootstrap_valid", "block_size", "n_resamples", "confidence_level", "seed", "block_mc_se",
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


# ---------------------------------------------------------------------------
# Chargement et extraction
# ---------------------------------------------------------------------------
def charger_run(run_dir: str | Path) -> pd.DataFrame:
    """
    Charge le parquet d'un run et vérifie la présence des colonnes
    minimales.

    Parameters
    ----------
    run_dir : str | Path
        Répertoire contenant `regimes.parquet`.

    Returns
    -------
    pd.DataFrame
        DataFrame brut du parquet, non filtré.
    """
    path = Path(run_dir) / "regimes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_parquet(path)
    required = {"timestamp", "regime", "method", "timeframe", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path} : {sorted(missing)}"
        )
    return df


def detecter_timeframes(df: pd.DataFrame) -> list[str]:
    """
    Retourne la liste ordonnée des timeframes présents dans le parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Parquet d'un run, avec colonne `timeframe`.

    Returns
    -------
    list[str]
        Liste triée des timeframes (ex. ['H1', 'M15']).
    """
    tfs = sorted(df["timeframe"].unique().tolist())
    if not tfs:
        raise ValueError("Aucun timeframe trouvé dans le parquet.")
    return tfs


def extraire_etats_int(series: pd.Series) -> pd.Series:
    """
    Convertit une série de labels HMM ('STATE_0', ...) en entiers.

    Les labels non reconnus (INCONNU, ou labels sémantiques d'autres
    méthodes) sont convertis en -1.
    """
    out = series.astype(str).str.extract(r"STATE_(\d+)")[0]
    return pd.to_numeric(out, errors="coerce").fillna(-1).astype(int)


def aligner_sur_features(
    df_run: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Aligne le run sur l'index des features via intersection.

    Parameters
    ----------
    df_run : pd.DataFrame
        Run filtré sur un couple (method, timeframe), avec colonne
        `timestamp`.
    features : pd.DataFrame
        Features construites sur le même timeframe, indexées par timestamp.

    Returns
    -------
    (features_alignées, states_int, confidence_alignée). Les codes internes
    sont nominaux ; states.attrs["labels"] conserve les labels originaux.
    """
    df, valid = prepare_regimes(df_run)
    validate_time_index(features.index)
    common = df.index.intersection(features.index)
    if common.empty:
        raise ValueError("Aucun timestamp commun entre le run et les features.")
    feats = features.reindex(common).copy()
    usable = valid.reindex(common) & validity_mask(feats)
    numeric = feats.select_dtypes(include="number")
    usable &= np.isfinite(numeric).all(axis=1)
    feats = feats.where(usable, axis=0)
    labels = df["regime"].reindex(common)
    usable &= labels.notna() & labels.ne("INCONNU")
    names = sorted(labels[usable].astype(str).unique())
    codes = {name: code for code, name in enumerate(names)}
    states = labels.astype(str).map(codes).fillna(-1).astype(int).where(usable, -1)
    states.attrs["labels"] = dict(enumerate(names))
    states.attrs["timeframe"] = df.attrs["series_identity"]["timeframe"]
    conf = df["confidence"].reindex(common).astype(float)
    conf = conf.where(usable & np.isfinite(conf))
    feats.attrs["exclusions"] = df.attrs["exclusions"]
    feats.attrs["n_sans_features"] = len(df) - int(np.isfinite(numeric).all(axis=1).sum())
    return feats, states, conf


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def statistiques_par_etat(
    features: pd.DataFrame, states: pd.Series
) -> pd.DataFrame:
    """
    Statistiques descriptives des features par état/régime.

    Pour chaque label (STATE_N ou label sémantique), calcule :
      n, part_pct, vol_moy, ret_moy, ret_std, range_moy, close_pos_moy.
    """
    valid = states.ge(0) & features.notna().all(axis=1)
    total = int(valid.sum())
    grouped = features.loc[valid].assign(state=states.loc[valid]).groupby("state")
    rows = []
    for state, g in grouped:
        rows.append({
            "state": state,
            "n": len(g),
            "part_pct": 100 * len(g) / total,
            "vol_moy": g["vol_realisee"].mean(),
            "ret_moy": g["log_return"].mean(),
            "ret_std": g["log_return"].std(),
            "range_moy": g["range_norm"].mean(),
            "close_pos_moy": g["close_position"].mean(),
        })
    return pd.DataFrame(rows, columns=["state", "n", "part_pct", "vol_moy",
                                      "ret_moy", "ret_std", "range_moy", "close_pos_moy"]).set_index("state")


def _label(states: pd.Series, code) -> str:
    return states.attrs.get("labels", {}).get(code, f"STATE_{code}")


def _adjacent(states: pd.Series, timeframe: str | None) -> np.ndarray:
    """Liens entre observations consécutives sur la grille déclarée."""
    if not isinstance(states.index, pd.DatetimeIndex):
        return np.ones(max(0, len(states) - 1), dtype=bool)
    validate_time_index(states.index)
    if len(states) < 2:
        return np.empty(0, dtype=bool)
    frame = pd.DataFrame(index=states.index)
    frame.attrs["timeframe"] = states.attrs.get("timeframe")
    tf = resolve_timeframe(frame, timeframe)
    return np.asarray(states.index[1:] - states.index[:-1] ==
                      pd.Timedelta(seconds=TIMEFRAME_SECONDS[tf]))


def persistance(states: pd.Series, *, timeframe: str | None = None) -> pd.DataFrame:
    """Durées observées en barres ; inconnus et trous coupent les épisodes.

    Les épisodes aux frontières sont tronqués : aucune durée non observée
    n'est extrapolée. Sur un index irrégulier, déclarer le timeframe.
    """
    adjacent = _adjacent(states, timeframe)
    episodes = []
    previous = None
    duration = 0
    for i, value in enumerate(states):
        valid = pd.notna(value) and value >= 0
        continues = valid and duration > 0 and adjacent[i - 1] and value == previous
        if not continues:
            if duration:
                episodes.append((previous, duration))
            duration = 0
        if valid:
            previous = value
            duration += 1
    if duration:
        episodes.append((previous, duration))
    df = pd.DataFrame(episodes, columns=["state", "duration"])
    return df.groupby("state")["duration"].agg(
        episodes="size", duree_moy="mean", duree_med="median", duree_max="max",
        p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75),
    )


def matrice_transition(states: pd.Series, *, timeframe: str | None = None) -> pd.DataFrame:
    """Probabilités sur les seuls couples adjacents valides.

    Une ligne sans transition observée contient des NaN (non estimable).
    Les effectifs sont disponibles dans attrs['counts'].
    """
    adjacent = _adjacent(states, timeframe)
    labels = sorted(states[states.notna() & states.ge(0)].unique())
    index_map = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    values = states.to_numpy()
    for i, (a, b) in enumerate(zip(values[:-1], values[1:])):
        if adjacent[i] and pd.notna(a) and pd.notna(b) and a >= 0 and b >= 0:
            counts[index_map[a], index_map[b]] += 1
    sums = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts.astype(float), sums,
                              out=np.full(counts.shape, np.nan), where=sums != 0)
    names = [_label(states, value) for value in labels]
    result = pd.DataFrame(probabilities, index=names, columns=names)
    result.attrs["counts"] = pd.DataFrame(counts, index=names, columns=names)
    return result


def distribution_confidence(
    confidence: pd.Series, states: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Retourne (describe_global, stats_par_etat).
    """
    valid = states.values >= 0
    conf_valid = confidence[valid]
    global_desc = conf_valid.describe()
    par_etat = (
        pd.DataFrame({"state": states, "confidence": confidence})
        .query("state >= 0")
        .groupby("state")["confidence"]
        .agg(["mean", "std", "min", "max"])
    )
    return global_desc, par_etat


def proportions_mensuelles(
    index: pd.DatetimeIndex,
    states: pd.Series,
    min_jours: int = 18,
) -> pd.DataFrame:
    """
    Proportions mensuelles : mois intérieurs avec au moins min_jours observés.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mois = index.to_period("M")
        jours = (
            pd.Series(index, index=mois)
            .groupby(level=0)
            .apply(lambda x: x.dt.normalize().nunique())
        )
        ok = jours[
            (jours >= min_jours)
            & (jours.index != mois.min())
            & (jours.index != mois.max())
        ].index

        tmp = pd.DataFrame({"mois": mois, "state": states.values})
        tmp = tmp[tmp["mois"].isin(ok) & (tmp["state"] >= 0)]
        if len(tmp) == 0:
            return pd.DataFrame()
        tmp["mois_str"] = tmp["mois"].astype(str)
        table = pd.crosstab(
            tmp["mois_str"], tmp["state"], normalize="index"
        ) * 100
        table.columns = [_label(states, c) for c in table.columns]
    return table


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------
def _sep(char: str = "─", width: int = 78) -> str:
    return char * width


def _titre(num: int, texte: str, width: int = 78) -> str:
    return f"{_sep()}\n  {num}. {texte.upper()}\n{_sep()}"


def formater_rapport(
    df_run: pd.DataFrame,
    df_source: pd.DataFrame,
    timeframe: str,
    features: pd.DataFrame,
    states: pd.Series,
    confidence: pd.Series,
    method: str,
) -> str:
    """
    Assemble le rapport texte complet pour un couple (method, timeframe).
    """
    lignes: list[str] = []
    A = lignes.append

    A("=" * 78)
    A(f"  RAPPORT D'ANALYSE — {method} — {timeframe}")
    A("Analyse descriptive des observations ; aucune validation prédictive.")
    A("=" * 78)
    A("")
    A(f"Période        : {df_source.index.min()} → {df_source.index.max()}")
    A(f"Barres {timeframe:4}     : {len(df_source):,}")
    A(f"Barres features: {len(features):,}")
    A(f"Barres communes: {len(states):,}")
    A(f"Méthode        : {method}")
    A("")

    A("Couverture (motifs d'exclusion potentiellement chevauchants) :")
    for name, count in features.attrs.get("exclusions", {}).items():
        A(f"  {name} : {count}")
    usable = states.ge(0) & features.notna().all(axis=1)
    A(f"  n_support_descriptif : {int(usable.sum())}")
    A(f"  n_sans_features : {features.attrs.get('n_sans_features', 0)}")
    A("")
    if not usable.any():
        A("Aucune estimation : support valide vide.")
        return "\n".join(lignes)

    # 1. Statistiques
    stats = statistiques_par_etat(features, states)
    A(_titre(1, "statistiques par état"))
    A(stats.rename(index=lambda code: _label(states, code)).round(6).to_string())
    A("")

    # 2. Persistance
    pers = persistance(states, timeframe=timeframe)
    A(_titre(2, "persistance"))
    A("Durées observées en barres ; épisodes tronqués aux frontières et interruptions.")
    A(pers.rename(index=lambda code: _label(states, code)).round(1).to_string())
    A("")

    # 3. Transition
    mat = matrice_transition(states, timeframe=timeframe)
    A(_titre(3, "matrice de transition"))
    if not mat.empty:
        A(mat.round(4).to_string())
        A("Effectifs des transitions adjacentes valides :")
        A(mat.attrs["counts"].to_string())
        A("")
        for label in mat.index:
            A(f"Stabilité {label} : {mat.loc[label, label]*100:.1f} %")
    A("")

    # 4. Confidence
    conf_desc, conf_par = distribution_confidence(confidence, states)
    A(_titre(4, "distribution de la confidence"))
    A(conf_desc.round(4).to_string())
    A("")
    A("Confidence par état :")
    A(conf_par.rename(index=lambda code: _label(states, code)).round(4).to_string())
    A("")

    # 5. Proportions mensuelles
    A(_titre(5, "proportions mensuelles (mois intérieurs, au moins 18 jours observés)"))
    mensuel = proportions_mensuelles(features.index, states)
    if not mensuel.empty:
        A(mensuel.round(1).to_string())
        A("")
        A("Moyenne :")
        A(mensuel.mean().round(1).astype(str).add(" %").to_string())
    else:
        A("Aucun mois répondant au seuil de jours observés.")
    A("")

    # 6. Résumé
    A(_titre(6, "résumé"))
    A(f"Barres analysées       : {int(usable.sum()):,}")
    for state in pers.index:
        if state >= 0:
            A(f"Durée médiane {_label(states, state)}  : "
              f"{pers.loc[state,'duree_med']:.0f} barres")
    A(f"États de durée médiane ≤ 2 barres : "
      f"{int((pers['duree_med'] <= 2).sum())}/{len(pers)}")
    A("")
    A("=" * 78)
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def rapport_complet(
    run_dir: str | Path,
    url_source: str,
    method: str | None = None,
    timeframe: str | None = None,
    *,
    asset: str | None = None,
    configuration: str | None = None,
    ohlcv_asset: str | None = None,
    ohlcv_timeframe: str | None = None,
) -> str:
    """
    Produit le rapport complet pour un run.

    Parameters
    ----------
    run_dir : str | Path
        Répertoire du run (contient `regimes.parquet`).
    url_source : str
        URL ou chemin du parquet OHLCV source.
    method : str | None
        Méthode à analyser. Si None et qu'une seule méthode existe,
        elle est choisie. Si None et plusieurs méthodes, exception.
    timeframe : str | None
        Timeframe à analyser. Si None et qu'un seul timeframe existe,
        il est choisi. Si None et plusieurs timeframes, un rapport par
        timeframe est produit puis concaténé.
    asset, configuration : str | None
        Sélection explicite de l’actif et du params_hash des régimes.
    ohlcv_asset, ohlcv_timeframe : str | None
        Identité explicite de la source, contrôlée contre ses métadonnées.

    Returns
    -------
    str
        Rapport formaté (concaténation si plusieurs timeframes).
    """
    from regime_lib.methods.statistical.hmm_gaussian import _construire_features

    df_run = charger_run(run_dir)
    if asset is not None:
        if "asset" not in df_run:
            raise ValueError("Identité de série absente : asset.")
        df_run = df_run[df_run["asset"].eq(asset)]

    # Sélection de la méthode
    methods = df_run["method"].unique().tolist()
    if method is None:
        if len(methods) != 1:
            raise ValueError(
                f"Plusieurs méthodes dans le run : {methods}. "
                f"Préciser --method."
            )
        method = methods[0]
    if method not in methods:
        raise ValueError(
            f"Méthode {method!r} absente du run. Présentes : {methods}."
        )

    df_run = df_run[df_run["method"].eq(method)]
    timeframes_present = detecter_timeframes(df_run)

    # Sélection des timeframes à traiter
    if timeframe is not None:
        if timeframe not in timeframes_present:
            raise ValueError(
                f"Timeframe {timeframe!r} absent du run. "
                f"Présents : {timeframes_present}."
            )
        timeframes_a_traiter = [timeframe]
    else:
        timeframes_a_traiter = timeframes_present

    # Chargement unique de la source (avant la boucle)
    print(f"[report] Chargement source {url_source}", file=sys.stderr)
    df_raw, _ = load_parquet(url_source, asset=ohlcv_asset, timeframe=ohlcv_timeframe)

    rapports: list[str] = []
    for tf in timeframes_a_traiter:
        print(f"[report] Traitement timeframe {tf}", file=sys.stderr)
        df_run_tf = df_run[
            (df_run["method"] == method)
            & (df_run["timeframe"] == tf)
        ].copy()

        print(f"[report]   Resample → {tf}", file=sys.stderr)
        prepared, _ = prepare_regimes(df_run_tf, configuration)
        df_run_tf = prepared.reset_index()
        df_tf, _ = prepare_prices(
            df_raw, prepared.attrs["series_identity"], asset=ohlcv_asset,
            timeframe=ohlcv_timeframe, required_index=prepared.index,
        )

        print(f"[report]   Construction des features", file=sys.stderr)
        features = _construire_features(df_tf)

        features = features.reindex(df_tf.index).where(validity_mask(df_tf), axis=0)
        feats, states, conf = aligner_sur_features(df_run_tf, features)
        print(
            f"[report]   {len(states):,} barres alignées",
            file=sys.stderr,
        )

        rapports.append(
            formater_rapport(
                df_run_tf, df_tf, tf, feats, states, conf, method
            )
        )

    # Séparateur visuel entre rapports si plusieurs timeframes.
    if len(rapports) > 1:
        separateur = "\n\n" + "=" * 78 + "\n\n"
        return separateur.join(rapports)
    return rapports[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regime_lib.study.report",
        description="Génère un rapport d'analyse pour un run de régimes.",
    )
    parser.add_argument("--run-dir", required=True,
                        help="Répertoire contenant regimes.parquet.")
    parser.add_argument("--url", required=True,
                        help="Chemin ou URL du parquet OHLCV source.")
    parser.add_argument("--method", default=None,
                        help="Méthode à analyser (auto si une seule).")
    parser.add_argument(
        "--timeframe",
        default=None,
        help=(
            "Timeframe à analyser. Si absent et que le run contient "
            "plusieurs timeframes, un rapport est produit pour chacun."
        ),
    )
    parser.add_argument("--asset", default=None, help="Actif des régimes à sélectionner.")
    parser.add_argument("--configuration", default=None, help="params_hash à sélectionner.")
    parser.add_argument("--ohlcv-asset", default=None, help="Actif explicite des prix source.")
    parser.add_argument("--ohlcv-timeframe", default=None, help="Fréquence des prix source.")
    parser.add_argument("--output", default=None,
                        help="Fichier de sortie (défaut : stdout).")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rapport = rapport_complet(
            args.run_dir,
            args.url,
            method=args.method,
            timeframe=args.timeframe, asset=args.asset, configuration=args.configuration,
            ohlcv_asset=args.ohlcv_asset, ohlcv_timeframe=args.ohlcv_timeframe,
        )
    except Exception as exc:
        print(f"[erreur] {exc}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(rapport, encoding="utf-8")
        print(f"[report] Écrit dans {args.output}", file=sys.stderr)
    else:
        print(rapport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
