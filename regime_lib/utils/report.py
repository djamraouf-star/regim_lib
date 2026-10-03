"""
Génération de rapports d'analyse pour les runs de détection de régimes.

Ce module lit un parquet produit par le pipeline CLI, recharge les
features sur le même timeframe, et produit un rapport texte structuré
prêt à être collé ou archivé.

Usage CLI
---------

.. code-block:: console

    # Rapport sur un timeframe précis
    python -m regime_lib.utils.report \
        --run-dir results_hmm2_H1 \
        --url /chemin/vers/EURUSD_M1.parquet \
        --timeframe H1

    # Rapport sur tous les timeframes du run (concaténés)
    python -m regime_lib.utils.report \
        --run-dir results_multi \
        --url /chemin/vers/EURUSD_M1.parquet \
        --method atr_volatility

    # Écrire dans un fichier au lieu de stdout
    python -m regime_lib.utils.report \
        --run-dir results_hmm2_H1 \
        --url /chemin/vers/EURUSD_M1.parquet \
        --output report_hmm.txt

Usage bibliothèque
------------------

.. code-block:: python

    from regime_lib.utils.report import rapport_complet
    texte = rapport_complet(
        "results_hmm2_H1",
        url_source="...",
        timeframe="H1",
    )
    print(texte)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lib.core.data_loader import load_parquet, resample

# Note : `_construire_features` est importée depuis hmm_gaussian pour
# l'instant. Elle gagnerait à migrer dans `regime_lib/utils/features.py`
# pour être partagée proprement. TODO si un second consommateur apparaît.
from regime_lib.methods.statistical.hmm_gaussian import _construire_features


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
    (features_alignées, states_int, confidence_alignée)
    """
    df = df_run.set_index("timestamp").sort_index()
    common = features.index.intersection(df.index)
    if len(common) == 0:
        raise ValueError(
            "Aucun timestamp commun entre le run et les features. "
            "Vérifier que le timeframe du run correspond aux features."
        )
    feats = features.reindex(common)
    states = extraire_etats_int(df["regime"].reindex(common))
    conf = df["confidence"].reindex(common).astype(float)
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
    total = len(features)
    grouped = features.assign(state=states.values).groupby("state")
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
    return pd.DataFrame(rows).set_index("state")


def persistance(states: pd.Series) -> pd.DataFrame:
    """Durée des épisodes consécutifs par état."""
    v = states.values
    eps: list[tuple[int, int]] = []
    if len(v) == 0:
        return pd.DataFrame()
    start = 0
    for i in range(1, len(v)):
        if v[i] != v[i - 1]:
            eps.append((v[start], i - start))
            start = i
    eps.append((v[start], len(v) - start))

    df = pd.DataFrame(eps, columns=["state", "duration"])
    return df.groupby("state")["duration"].agg(
        episodes="size",
        duree_moy="mean",
        duree_med="median",
        duree_max="max",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    )


def matrice_transition(states: pd.Series) -> pd.DataFrame:
    """
    Matrice de transition entre états valides (exclut -1 / INCONNU).
    """
    labels = sorted(s for s in states.unique() if s >= 0)
    n = len(labels)
    if n == 0:
        return pd.DataFrame()

    index_map = {label: i for i, label in enumerate(labels)}
    valid = states.values[states.values >= 0]
    counts = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(valid[:-1], valid[1:]):
        counts[index_map[int(a)], index_map[int(b)]] += 1

    sums = counts.sum(axis=1, keepdims=True)
    p = np.divide(
        counts.astype(float), sums,
        out=np.zeros_like(counts, dtype=float),
        where=sums != 0,
    )
    return pd.DataFrame(
        p,
        index=[f"STATE_{s}" if s >= 0 else "INCONNU" for s in labels],
        columns=[f"STATE_{s}" if s >= 0 else "INCONNU" for s in labels],
    )


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
    Proportions mensuelles (mois complets uniquement).
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
        table.columns = [f"STATE_{c}" for c in table.columns]
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
    A("=" * 78)
    A("")
    A(f"Période        : {df_source.index.min()} → {df_source.index.max()}")
    A(f"Barres {timeframe:4}     : {len(df_source):,}")
    A(f"Barres features: {len(features):,}")
    A(f"Barres communes: {len(states):,}")
    A(f"Méthode        : {method}")
    A("")

    # 1. Statistiques
    stats = statistiques_par_etat(features, states)
    A(_titre(1, "statistiques par état"))
    A(stats.round(6).to_string())
    A("")

    s = stats[stats.index >= 0]
    ratio = None
    if len(s) == 2:
        calme = s["vol_moy"].idxmin()
        agite = s["vol_moy"].idxmax()
        ratio = s.loc[agite, "vol_moy"] / s.loc[calme, "vol_moy"]
        A(f"STATE_{calme} = CALME  (vol {s.loc[calme,'vol_moy']:.6f}, "
          f"{s.loc[calme,'part_pct']:.1f} %)")
        A(f"STATE_{agite} = AGITÉ  (vol {s.loc[agite,'vol_moy']:.6f}, "
          f"{s.loc[agite,'part_pct']:.1f} %)")
        A(f"Ratio de volatilité : {ratio:.2f}")
        A("")

    # 2. Persistance
    pers = persistance(states)
    A(_titre(2, "persistance"))
    A(pers.round(1).to_string())
    A("")

    # 3. Transition
    mat = matrice_transition(states)
    A(_titre(3, "matrice de transition"))
    if not mat.empty:
        A(mat.round(4).to_string())
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
    A(conf_par.round(4).to_string())
    A("")

    # 5. Proportions mensuelles
    A(_titre(5, "proportions mensuelles (mois complets)"))
    mensuel = proportions_mensuelles(features.index, states)
    if not mensuel.empty:
        A(mensuel.round(1).to_string())
        A("")
        A("Moyenne :")
        A(mensuel.mean().round(1).astype(str).add(" %").to_string())
    else:
        A("Aucun mois complet détecté.")
    A("")

    # 6. Résumé
    A(_titre(6, "résumé"))
    A(f"Barres analysées       : {len(states):,}")
    if ratio is not None:
        A(f"Ratio de volatilité    : {ratio:.2f}")
    for state in pers.index:
        if state >= 0:
            A(f"Durée médiane STATE_{state}  : "
              f"{pers.loc[state,'duree_med']:.0f} barres")
    A(f"États de bruit         : "
      f"{int((pers['duree_med'] <= 2).sum())}/{len(pers)}")
    if len(s) > 0:
        biais = abs(s["ret_moy"]).max() > 5e-4
        A(f"Biais directionnel     : {'oui' if biais else 'non'}")
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

    Returns
    -------
    str
        Rapport formaté (concaténation si plusieurs timeframes).
    """
    df_run = charger_run(run_dir)
    timeframes_present = detecter_timeframes(df_run)

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
    df_raw, _ = load_parquet(url_source)

    rapports: list[str] = []
    for tf in timeframes_a_traiter:
        print(f"[report] Traitement timeframe {tf}", file=sys.stderr)
        df_run_tf = df_run[
            (df_run["method"] == method)
            & (df_run["timeframe"] == tf)
        ].copy()

        print(f"[report]   Resample → {tf}", file=sys.stderr)
        df_tf = resample(df_raw, tf)

        print(f"[report]   Construction des features", file=sys.stderr)
        features = _construire_features(df_tf)

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
        prog="regime_lib.utils.report",
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
            timeframe=args.timeframe,
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
