"""
Point d'entrée CLI.

Exemple simple :
    python -m regime_lib \\
        --url "https://.../EURUSD_M1.parquet" \\
        --timeframe H1 \\
        --methods atr_volatility,er_kaufman \\
        --output results/

Multi-timeframe (fichier unique) :
    python -m regime_lib \\
        --url "https://.../EURUSD_M1.parquet" \\
        --timeframe M1,M5,M15,H1 \\
        --methods atr_volatility,er_kaufman,hmm_gaussian \\
        --allow-lookahead-override hmm_gaussian:true \\
        --output results_multi

Avec plusieurs timeframes, TOUS les résultats sont empilés dans un seul
parquet long. La colonne `timeframe` distingue les lignes :

.. code-block:: text

    results_multi/regimes.parquet   - M1 + M5 + M15 + H1
    results_multi/run_meta.json     - liste des timeframes

Avec un seul timeframe, le comportement reste identique (un seul
timeframe dans le parquet).

Profil de configuration :
    python -m regime_lib \\
        --url "https://.../EURUSD_M1.parquet" \\
        --timeframe H1 \\
        --methods adx \\
        --profile eurusd \\
        --output results_eurusd

Le profil (`--profile`) charge un fichier YAML dans
`regime_lib/config/profiles/`. Il définit les paramètres par actif et
par timeframe. Sans `--profile`, le profil `default` est utilisé.

Contexte temporel :
    python -m regime_lib \\
        --url "https://.../EURUSD_M1.parquet" \\
        --timeframe H1 \\
        --methods adx,shannon \\
        --with-context \\
        --output results_ctx

Le flag `--with-context` enrichit le parquet long avec les colonnes
`ctx_*` (session de marché, rollover, fériés, événements macro). Voir
`doc/context.md`.

Lister les profils disponibles :
    python -m regime_lib --list-profiles

Pour une méthode offline (lookahead structurel) :
    python -m regime_lib \\
        --url "https://.../EURUSD_M1.parquet" \\
        --timeframe H1 \\
        --methods price_action \\
        --allow-lookahead-override price_action:true \\
        --output results_pa
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Iterable

from regime_lib.config import (
    list_profiles,
    load_profile,
    profile_hash,
)
from regime_lib.core.base import RegimeDetector
from regime_lib.core.data_loader import (
    TIMEFRAME_RULES,
    load_parquet,
    resample,
)
from regime_lib.core.output import export_long, export_meta, to_uniform
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.utils.repro import hash_params
from regime_lib.utils.validation import validate_alignment

# Effet de bord volontaire : importe les méthodes concrètes pour déclencher
# les décorateurs @register_method et peupler METHOD_REGISTRY.
import regime_lib.methods  # noqa: F401


# ---------------------------------------------------------------------------
# Parsers d'arguments
# ---------------------------------------------------------------------------
def _parse_methods(raw: str) -> list[str]:
    methods = [m.strip() for m in raw.split(",") if m.strip()]
    if not methods:
        raise argparse.ArgumentTypeError("--methods ne peut pas être vide.")
    unknown = [m for m in methods if m not in METHOD_REGISTRY]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Méthodes inconnues : {unknown}. "
            f"Disponibles : {sorted(METHOD_REGISTRY)}."
        )
    return methods


def _parse_timeframes(raw: str) -> list[str]:
    """
    Parse 'M1,M5,M15' → ['M1', 'M5', 'M15'].

    Normalise en majuscules, valide contre TIMEFRAME_RULES, dédoublonne
    en préservant l'ordre.
    """
    tfs = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tfs:
        raise argparse.ArgumentTypeError(
            "--timeframe ne peut pas être vide."
        )
    unknown = [t for t in tfs if t not in TIMEFRAME_RULES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Timeframes inconnus : {unknown}. "
            f"Valides : {sorted(TIMEFRAME_RULES)}."
        )
    seen: set[str] = set()
    uniques: list[str] = []
    for t in tfs:
        if t not in seen:
            uniques.append(t)
            seen.add(t)
    return uniques


def _parse_lookahead_override(raw: str) -> dict[str, bool]:
    """Parse 'hmm_gaussian:true,price_action:true'."""
    result: dict[str, bool] = {}
    if not raw:
        return result
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise argparse.ArgumentTypeError(
                f"Format invalide pour --allow-lookahead-override : {chunk!r}."
            )
        name, value = chunk.split(":", 1)
        v = value.strip().lower()
        if v not in ("true", "false", "1", "0"):
            raise argparse.ArgumentTypeError(
                f"Valeur booléenne invalide pour {name!r} : {value!r}."
            )
        result[name.strip()] = v in ("true", "1")
    return result


def _parse_method_params(raw: str) -> dict[str, dict]:
    """Parse un JSON de surcharges par méthode."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--method-params : JSON invalide ({exc})."
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "--method-params doit être un objet JSON."
        )
    for k, v in parsed.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            raise argparse.ArgumentTypeError(
                "--method-params : chaque entrée doit être "
                "'nom_methode': {param: valeur}."
            )
    return parsed


def _str2bool(v: str) -> bool:
    if v.lower() in ("true", "1", "yes", "y"):
        return True
    if v.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Booléen invalide : {v!r}.")


# ---------------------------------------------------------------------------
# Construction du parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regime_lib",
        description="Détection de régimes de marché multi-méthodes.",
    )
    # Arguments du pipeline : non requis au niveau argparse pour permettre
    # aux options de liste de fonctionner seules. La validation est faite
    # manuellement dans run().
    parser.add_argument("--url", default=None,
                        help="URL locale ou distante du parquet OHLCV.")
    parser.add_argument("--source-timeframe", choices=sorted(TIMEFRAME_RULES), default=None,
                        help="Fréquence des barres source ; requise si la grille est irrégulière.")
    parser.add_argument("--timestamp-convention", choices=["open", "close"], default=None,
                        help="Convention des barres source (métadonnées, sinon close).")
    parser.add_argument("--source-timezone", default="UTC",
                        help="Fuseau des timestamps de barres sans fuseau (défaut UTC).")
    parser.add_argument(
        "--timeframe",
        default=None,
        help=(
            "Timeframe(s) cible(s). Une valeur unique (ex. 'H1') ou "
            "plusieurs séparées par des virgules (ex. 'M1,M5,M15'). "
            "Tous les résultats sont empilés dans un seul parquet long ; "
            "la colonne `timeframe` distingue les lignes."
        ),
    )
    parser.add_argument("--methods", default=None, type=_parse_methods,
                        help="Liste de méthodes séparées par des virgules.")
    parser.add_argument("--asset", default=None,
                        help="Symbole de l'actif. Fallback : nom de fichier.")
    parser.add_argument(
        "--profile",
        default="default",
        help=(
            "Profil de configuration (nom court ou chemin YAML). "
            "Défaut : 'default'. Les profils sont dans "
            "regime_lib/config/profiles/. "
            "Voir `--list-profiles` pour la liste."
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Liste les profils disponibles et quitte.",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="Liste les méthodes enregistrées et quitte.",
    )
    parser.add_argument("--show-config", action="store_true",
                        help="Affiche paramètres effectifs, provenance et hash sans lire de données.")
    parser.add_argument("--allow-lookahead", type=_str2bool, default=False,
                        help="Autorise le lookahead globalement (défaut : false).")
    parser.add_argument("--allow-lookahead-override",
                        type=_parse_lookahead_override, default={},
                        help="Override par méthode, ex. 'hmm_gaussian:true'.")
    parser.add_argument(
        "--method-params",
        type=_parse_method_params,
        default={},
        help=(
            "Surcharges par méthode au format JSON, ex. "
            "'{\"atr_volatility\": {\"n_atr\": 20}}'. "
            "Ces surcharges écrasent les valeurs du profil."
        ),
    )
    parser.add_argument(
        "--with-context",
        action="store_true",
        help=(
            "Enrichit le parquet long avec les colonnes de contexte "
            "temporel (ctx_session, ctx_is_rollover, ctx_is_holiday, "
            "ctx_in_event, etc.). Désactivé par défaut pour ne pas "
            "modifier le format des parquets existants."
        ),
    )
    parser.add_argument("--output", default=None,
                        help="Répertoire de sortie.")
    return parser


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
def _build_detectors(args, profil, timeframe):
    """Construit les instances une seule fois, avant toute lecture des données."""
    unused = set(args.method_params) - set(args.methods)
    if unused:
        raise ValueError(f"Surcharges pour des méthodes inconnues ou non sélectionnées : {sorted(unused)}")
    result = {}
    for name in args.methods:
        cls = METHOD_REGISTRY[name]
        extra = cls.PARAM_SCHEMA.explicit_values(args.method_params.get(name, {}))
        # Vérifier les clés avant l'appel Python pour un diagnostic avec chemin.
        cls.PARAM_SCHEMA.validate_layer(extra, f"explicit.methodes.{name}")
        result[name] = cls(
            timeframe=timeframe,
            profile=profil.for_method(name),
            allow_lookahead=args.allow_lookahead_override.get(name, args.allow_lookahead),
            **extra,
        )
    return result


def _describe_detectors(detectors):
    return {
        name: {
            "interpretation": detector.interpretation(),
            "params": detector.params,
            "sources": detector.param_sources,
            "params_hash": hash_params(detector.params),
        }
        for name, detector in detectors.items()
    }


def _run_one_timeframe(
    df_raw,
    asset: str,
    timeframe: str,
    args: argparse.Namespace,
    profil,
    detectors=None,
) -> tuple[list, list[str], int]:
    """
    Exécute toutes les méthodes pour un timeframe.

    Le sous-profil de chaque méthode est extrait du profil global et
    passé via `profile=...`. Les surcharges CLI (`--method-params`)
    restent prioritaires.

    Si `args.with_context` est vrai, chaque DataFrame de sortie est
    enrichi avec les colonnes de contexte temporel (`ctx_*`) AVANT
    l'appel à `to_uniform`.

    Retourne
    --------
    (frames, methodes_avec_lookahead, n_rows_resampled)
    """
    print(f"[run] Timeframe {timeframe}", file=sys.stderr)
    df_tf = resample(df_raw, timeframe)

    # Enrichissement du contexte temporel, calculé une seule fois par
    # timeframe (indépendant des méthodes).
    df_ctx = None
    cols_ctx: list[str] = []
    if args.with_context:
        from regime_lib.context import enrichir_contexte
        df_ctx = enrichir_contexte(df_tf, timeframe, profil.context)
        cols_ctx = [c for c in df_ctx.columns if c.startswith("ctx_")]
        print(
            f"[run]   Contexte : {len(cols_ctx)} colonnes ctx_* "
            f"({', '.join(cols_ctx)})",
            file=sys.stderr,
        )

    frames: list = []
    methodes_avec_lookahead: list[str] = []

    detectors = detectors if detectors is not None else _build_detectors(args, profil, timeframe)
    for method_name in args.methods:
        cls = METHOD_REGISTRY[method_name]
        allow = args.allow_lookahead_override.get(
            method_name, args.allow_lookahead
        )

        if allow and getattr(cls, "requires_lookahead", False):
            methodes_avec_lookahead.append(method_name)

        detector = detectors[method_name]

        print(
            f"[run]   → {method_name} "
            f"(lookahead={'oui' if allow else 'non'})",
            file=sys.stderr,
        )
        out = detector.fit_predict(df_tf)
        validate_alignment(df_tf.index, out.index)

        # Propagation des colonnes ctx_* sur `out`.
        # `out` et `df_ctx` partagent le même DatetimeIndex (issu de
        # df_tf), dans le même ordre. L'assignation directe est correcte.
        if df_ctx is not None:
            for col in cols_ctx:
                validate_alignment(out.index, df_ctx.index)
                out[col] = df_ctx[col]

        uniform = to_uniform(
            out,
            method=detector.name,
            timeframe=timeframe,
            asset=asset,
            regime_map=detector.REGIME_MAP,
            params_hash=hash_params(detector.params),
        )
        frames.append(uniform)

    return frames, methodes_avec_lookahead, len(df_tf)


def run(args: argparse.Namespace) -> int:
    """
    Exécute le pipeline : chargement, resampling, méthodes, export.

    Supporte un ou plusieurs timeframes. TOUS les résultats sont empilés
    dans un seul parquet long (`regimes.parquet`), distingués par la
    colonne `timeframe`.

    Retourne un code de sortie (0 = succès).
    """
    # --- Validation des arguments requis --------------------------------
    manquants: list[str] = []
    show_config = getattr(args, "show_config", False)
    if not args.url and not show_config:
        manquants.append("--url")
    if not args.timeframe:
        manquants.append("--timeframe")
    if not args.methods:
        manquants.append("--methods")
    if not args.output and not show_config:
        manquants.append("--output")
    if manquants:
        print(
            f"[erreur] Arguments requis manquants : {', '.join(manquants)}",
            file=sys.stderr,
        )
        return 2

    timeframes = _parse_timeframes(args.timeframe)

    # --- Chargement du profil -------------------------------------------
    try:
        profil = load_profile(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[erreur] Profil : {exc}", file=sys.stderr)
        return 1

    hash_profil = profile_hash(profil)
    print(
        f"[run] Profil '{profil.nom}' chargé (hash={hash_profil})",
        file=sys.stderr,
    )

    detectors = {tf: _build_detectors(args, profil, tf) for tf in timeframes}
    effective_config = {tf: _describe_detectors(items) for tf, items in detectors.items()}
    if show_config:
        print(json.dumps(effective_config, indent=2, ensure_ascii=False))
        return 0

    # --- Chargement des données -----------------------------------------
    print(f"[run] Chargement de {args.url}", file=sys.stderr)
    df_raw, asset = load_parquet(
        args.url, asset=args.asset,
        timeframe=getattr(args, "source_timeframe", None),
        timestamp_convention=getattr(args, "timestamp_convention", None),
        source_timezone=getattr(args, "source_timezone", "UTC"),
    )
    source_type = df_raw.attrs.get("source_type", "barres")
    source_rows = int(df_raw.attrs.get("source_rows", len(df_raw)))
    if source_type == "ticks":
        print(
            f"[run] {source_rows:,} ticks agrégés en "
            f"{len(df_raw):,} barres M1 (asset={asset})",
            file=sys.stderr,
        )
    else:
        print(
            f"[run] {source_rows:,} barres chargées (asset={asset})",
            file=sys.stderr,
        )
    print(
        f"[run] {len(timeframes)} timeframe(s) : {timeframes}",
        file=sys.stderr,
    )

    # --- Exécution par timeframe ----------------------------------------
    all_frames: list = []
    all_lookahead: list[str] = []
    rows_par_timeframe: dict[str, int] = {}

    for tf in timeframes:
        frames, lookahead, n_rows = _run_one_timeframe(
            df_raw, asset, tf, args, profil, detectors[tf]
        )
        all_frames.extend(frames)
        # On dédoublonne les méthodes avec lookahead (elles apparaissent
        # une fois par timeframe).
        for m in lookahead:
            if m not in all_lookahead:
                all_lookahead.append(m)
        rows_par_timeframe[tf] = n_rows

    # --- Export ---------------------------------------------------------
    from regime_lib.utils.provenance import environment, frame_info, file_hash
    export_path = export_long(all_frames, args.output)
    export_meta(
        {
            **environment(),
            "input_prepared": frame_info(df_raw),
            "outputs": [frame_info(frame) for frame in all_frames],
            "artifacts": {export_path.name: file_hash(export_path)},
            "url": args.url,
            "source_type": source_type,
            "timeframes": timeframes,
            "asset": asset,
            "methods": args.methods,
            # Profil utilisé
            "profile_name": profil.nom,
            "profile_type": profil.type,
            "profile_hash": hash_profil,
            "profile_path": (
                str(profil.source_path) if profil.source_path else None
            ),
            "allow_lookahead": args.allow_lookahead,
            "allow_lookahead_override": args.allow_lookahead_override,
            "method_params": args.method_params,
            "effective_config": effective_config,
            "methodes_avec_lookahead": all_lookahead,
            "run_descriptif": any(d.availability != "bar_close"
                                   for group in detectors.values() for d in group.values()),
            # Contexte temporel
            "with_context": bool(args.with_context),
            "run_at": datetime.now(timezone.utc).isoformat(),
            "n_rows_input": source_rows,
            "n_rows_par_timeframe": rows_par_timeframe,
            "n_lignes_total": int(sum(len(f) for f in all_frames)),
        },
        args.output,
    )

    print(
        f"[run] Terminé. {len(timeframes)} timeframe(s), "
        f"{len(all_frames)} DataFrame(s) empilé(s).",
        file=sys.stderr,
    )
    print(
        f"[run] Export → {args.output}/regimes.parquet",
        file=sys.stderr,
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Cas particulier : lister les profils sans exécuter le pipeline.
    if args.list_profiles:
        profils = list_profiles()
        print("Profils disponibles :")
        for p in profils:
            print(f"  - {p}")
        return 0

    if args.list_methods:
        print("Méthodes disponibles :")
        for name in sorted(METHOD_REGISTRY):
            print(f"  - {name}")
        return 0

    try:
        return run(args)
    except Exception as exc:  # pragma: no cover — point d'entrée
        print(f"[erreur] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
