"""Criblage walk-forward des régimes et du contexte temporel EURUSD."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT / "actifs/EURUSD/EURUSD_Tick_20250623_20261001.parquet"
)
DEFAULT_METHODS = [
    "atr_volatility",
    "er_kaufman",
    "adx",
    "shannon",
    "pca_axis",
    "minkowski_causal",
    "ofi",
    "divergence_pv",
]
OFFLINE_METHODS = {"hmm_gaussian", "price_action", "mss_choch"}
CONTEXT_FEATURES = [
    "ctx_session",
    "ctx_hour_ny",
    "ctx_day_of_week",
    "ctx_is_rollover",
    "ctx_is_holiday",
    "ctx_in_event",
    "ctx_event_type",
]


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crible les régimes causaux EURUSD et recherche les effets "
            "des sessions, heures NY et jours de semaine."
        ),
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help=f"Parquet source OHLCV ou tick (défaut : {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--profile", default="eurusd",
        help="Profil embarqué ou chemin YAML (défaut : eurusd).",
    )
    parser.add_argument(
        "--timeframes", type=_csv_list, default=["M5", "H1"],
        help="Timeframes séparés par des virgules (défaut : M5,H1).",
    )
    parser.add_argument(
        "--methods", type=_csv_list, default=DEFAULT_METHODS,
        help=(
            "Méthodes séparées par des virgules. "
            "Les méthodes offline sont exclues par défaut."
        ),
    )
    parser.add_argument(
        "--horizons", type=_csv_list, default=["1", "3", "6", "12", "24"],
        help="Horizons en barres (défaut : 1,3,6,12,24).",
    )
    parser.add_argument(
        "--families", type=_csv_list, default=["ret", "vol", "dd", "ru"],
        help="Familles de cibles (défaut : ret,vol,dd,ru).",
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Nombre de fenêtres walk-forward (défaut : 5).",
    )
    parser.add_argument(
        "--min-samples", type=int, default=30,
        help="Effectif minimal par groupe pour le criblage croisé (défaut : 30).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Dossier de sortie (défaut : output/screening_<UTC timestamp>).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Autorise l'écriture dans un dossier de sortie non vide.",
    )
    args = parser.parse_args()

    try:
        args.horizons = [int(value) for value in args.horizons]
    except ValueError as exc:
        parser.error(f"Les horizons doivent être des entiers : {exc}")
    if not args.horizons or any(horizon < 1 for horizon in args.horizons):
        parser.error("Les horizons doivent être des entiers strictement positifs.")
    if not args.families or set(args.families) - {"ret", "vol", "dd", "ru"}:
        parser.error("Familles valides : ret, vol, dd, ru.")
    if not args.timeframes:
        parser.error("Au moins un timeframe doit être fourni.")
    if args.folds < 1:
        parser.error("--folds doit être >= 1.")
    if args.min_samples < 2:
        parser.error("--min-samples doit être >= 2.")
    offline = sorted(set(args.methods) & OFFLINE_METHODS)
    if offline:
        parser.error(
            "Les méthodes offline sont exclues pour éviter le lookahead : "
            f"{', '.join(offline)}."
        )
    if not args.methods:
        parser.error("Au moins une méthode doit être fournie.")
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = ROOT / "analyses/EURUSD/output" / f"screening_{stamp}"
    elif not args.output.is_absolute():
        args.output = ROOT / args.output
    if not args.input.is_absolute():
        args.input = ROOT / args.input
    profile_path = Path(args.profile)
    if profile_path.suffix == ".yaml" and not profile_path.is_absolute():
        args.profile = str(ROOT / profile_path)
    return args


def _adjust_bh(pvalues: pd.Series) -> pd.Series:
    """Correction Benjamini-Hochberg, en conservant les valeurs manquantes."""
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return pd.Series(adjusted, index=pvalues.index)

    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return pd.Series(adjusted, index=pvalues.index)


def _apply_fdr(
    results: pd.DataFrame,
    group_columns: list[str],
    p_columns: list[str],
) -> pd.DataFrame:
    if results.empty:
        return results
    out = results.copy().reset_index(drop=True)
    for p_column in p_columns:
        q_column = p_column.removesuffix("_p") + "_q"
        out[q_column] = np.nan
        for _, group in out.groupby(group_columns, dropna=False).groups.items():
            out.loc[group, q_column] = _adjust_bh(
                out.loc[group, p_column]
            ).to_numpy()
    return out


def criblage_regimes_par_session(
    regimes: pd.DataFrame,
    contexte: pd.DataFrame,
    cibles: pd.DataFrame,
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]],
    min_samples: int = 30,
) -> pd.DataFrame:
    """Compare chaque régime aux autres au sein de chaque session test."""
    from regime_lib.study.evaluation import batterie_tests

    rows: list[dict] = []
    for method in sorted(regimes["method"].dropna().unique()):
        method_df = regimes.loc[regimes["method"] == method].copy()
        method_df = method_df.set_index("timestamp")
        for fold_id, (_, test_index) in enumerate(folds):
            common = (
                method_df.index.intersection(contexte.index)
                .intersection(cibles.index)
                .intersection(test_index)
            )
            if len(common) == 0:
                continue
            data = pd.concat(
                [
                    method_df.loc[common, ["regime"]],
                    contexte.loc[common, ["ctx_session"]],
                    cibles.loc[common],
                ],
                axis=1,
            )
            for target in cibles.columns:
                valid = data.dropna(subset=["regime", "ctx_session", target])
                valid = valid.loc[
                    (valid["regime"] != "INCONNU")
                    & (valid["ctx_session"] != "HORS_SESSION")
                ]
                for session in sorted(valid["ctx_session"].unique()):
                    session_rows = valid.loc[valid["ctx_session"] == session]
                    regimes_present = sorted(session_rows["regime"].unique())
                    for regime in regimes_present:
                        group = session_rows.loc[
                            session_rows["regime"] == regime, target
                        ].to_numpy(dtype=float)
                        rest = session_rows.loc[
                            session_rows["regime"] != regime, target
                        ].to_numpy(dtype=float)
                        if min(len(group), len(rest)) < min_samples:
                            continue
                        metrics = batterie_tests(group, rest, target)
                        rows.append({
                            "fold": fold_id,
                            "method": method,
                            "session": session,
                            "regime": regime,
                            "cible": target,
                            "n_session": len(session_rows),
                            **metrics,
                        })

    columns = [
        "fold", "method", "session", "regime", "cible", "n_session",
        "n_a", "n_b", "mean_a", "mean_b", "std_a", "std_b",
        "p_a_pos", "p_b_pos", "welch_t", "welch_p", "mw_u", "mw_p",
        "z_stat", "z_p",
    ]
    results = pd.DataFrame(rows, columns=columns)
    return _apply_fdr(
        results,
        ["fold", "method", "cible"],
        ["welch_p", "mw_p", "z_p"],
    )


def _run_cli(
    input_path: Path,
    profile: str,
    timeframe: str,
    methods: list[str],
    output_dir: Path,
) -> Path:
    run_dir = output_dir / timeframe / "cli"
    command = [
        sys.executable, "-m", "regime_lib",
        "--url", str(input_path),
        "--timeframe", timeframe,
        "--methods", ",".join(methods),
        "--profile", profile,
        "--with-context",
        "--output", str(run_dir),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    regimes_path = run_dir / "regimes.parquet"
    if not regimes_path.is_file():
        raise FileNotFoundError(
            f"Le CLI s'est terminé sans produire {regimes_path}."
        )
    return regimes_path


def _run_timeframe(
    args: argparse.Namespace,
    timeframe: str,
    profile,
) -> None:
    from regime_lib.core.data_loader import load_parquet, resample
    from regime_lib.context import enrichir_contexte
    from regime_lib.study import Study
    from regime_lib.study.split import split_walk_forward
    from regime_lib.study.targets import calculer_cible

    tf_dir = args.output / timeframe
    regimes_path = _run_cli(
        args.input, args.profile, timeframe, args.methods, args.output
    )
    targets = [
        f"{family}_fwd_{horizon}"
        for family in args.families
        for horizon in args.horizons
    ]
    split_kwargs = {
        "n_folds": args.folds,
        "embargo": max(args.horizons),
    }

    method_study = Study(
        regimes_path=regimes_path,
        ohlcv_path=args.input,
        features=["regime", "confidence"],
        targets=targets,
        methodes=args.methods,
        methodes_causales_only=True,
        split="walk_forward",
        **split_kwargs,
    )
    method_study.run()
    method_study.save(tf_dir / "method_screening")

    context_study = Study(
        regimes_path=regimes_path,
        ohlcv_path=args.input,
        features=CONTEXT_FEATURES,
        targets=targets,
        methodes=["adx"],
        methodes_causales_only=True,
        split="walk_forward",
        **split_kwargs,
    )
    context_study.run()
    context_study.save(tf_dir / "context_screening")

    df_ohlcv, _ = load_parquet(str(args.input))
    bars = resample(df_ohlcv, timeframe)
    bars_context = enrichir_contexte(bars, timeframe, profile.context)
    target_frame = pd.DataFrame(
        {
            target: calculer_cible(bars, target)
            for target in targets
        }
    )
    common = bars_context.index.intersection(target_frame.index)
    folds = split_walk_forward(
        common,
        n_folds=args.folds,
        embargo=max(args.horizons),
    )
    regimes = pd.read_parquet(regimes_path)
    regimes = regimes.loc[regimes["timeframe"] == timeframe]
    session_results = criblage_regimes_par_session(
        regimes,
        bars_context,
        target_frame,
        folds,
        min_samples=args.min_samples,
    )
    session_path = tf_dir / "session_regime_screening.parquet"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_results.to_parquet(session_path, index=False)
    print(f"Criblage régime × session : {session_path}")


def main() -> int:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Fichier source introuvable : {args.input}")
    if args.output.is_file():
        raise FileExistsError(
            f"Le chemin de sortie est un fichier : {args.output}"
        )
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Le dossier de sortie n'est pas vide : {args.output}. "
            "Choisir un autre --output ou passer --overwrite."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    from regime_lib.config import load_profile

    profile = load_profile(args.profile)
    for timeframe in args.timeframes:
        print(f"\n=== Criblage EURUSD {timeframe} ===")
        _run_timeframe(args, timeframe, profile)
    print(f"\nTous les résultats sont dans : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
