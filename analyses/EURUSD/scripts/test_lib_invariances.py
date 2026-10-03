"""
Tests d'invariance de regime_lib sur données EURUSD.

Test 1 — Causalité par troncature
    Pour chaque (timeframe, méthode) : on lance la détection sur la série
    complète, puis sur la série tronquée à t. Les labels de la série tronquée
    doivent être IDENTIQUES à ceux du run complet aux mêmes timestamps.
    Toute différence = la méthode utilise de l'information postérieure
    (lookahead) ou dépend de la longueur totale de la série (normalisation
    globale, refit non ancré sur le début de série).

    Lecture : si les écarts se concentrent dans les dernières barres avant la
    coupure, c'est une fenêtre qui regarde devant (lissage centré, etc.).
    S'ils sont répartis, c'est plutôt une normalisation ou un fit global.

Test 2 — Miroir
    On reflète les prix autour du centre de la plage : p' = 2K - p, avec
    high' = 2K - low et low' = 2K - high. Les différences de prix, donc ATR,
    ER, ADX, sont conservées ; la direction s'inverse. Attendu :
      - HAUSSIER <-> BAISSIER (et TENDANCE_HAUSSIERE <-> TENDANCE_BAISSIERE)
      - tous les autres labels inchangés.
    Les méthodes qui travaillent sur des log-rendements verront des écarts
    mineurs (le reflet ne conserve pas exactement les rendements relatifs) :
    d'où un seuil PASS / WARN / FAIL indicatif, pas un test d'égalité stricte.
    Seules les colonnes open/high/low/close sont reflétées ; volume, spreads
    et autres colonnes restent inchangés.

    Option --miroir log : reflet multiplicatif p' = K²/p (K = centre
    géométrique de la plage). Les log-rendements s'inversent EXACTEMENT, donc
    les méthodes basées sur des rendements relatifs (candidats : shannon,
    minkowski_causal) doivent monter vers 100 %. En contrepartie, les
    différences de prix ne sont plus conservées : ATR, ER, ADX peuvent dévier
    légèrement.
    Lecture croisée : PASS en arithmétique et WARN en log = la méthode
    travaille sur des différences de prix ; l'inverse = sur des rendements
    relatifs. Un écart qui persiste dans les DEUX variantes indique un
    traitement asymétrique dans la méthode elle-même.

Usage :
    python test_lib_invariances.py
    python test_lib_invariances.py --timeframes M15 H1 H4
    python test_lib_invariances.py --methodes adx pca_axis --timeframes H1
    python test_lib_invariances.py --miroir log --sans-troncature \
        --methodes shannon minkowski_causal --timeframes M15 H1 H4

Code de sortie : 1 si au moins un FAIL, sinon 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime_lib.core.data_loader import load_parquet, resample
from regime_lib.core.registry import METHOD_REGISTRY

import regime_lib.methods  # noqa: F401  (effet de bord : registre)


URL = Path.home() / "Projects/regime_lib/data/EURUSD_Tick_20250623_20261001.parquet"

METHODES_DEFAUT = [
    "atr_volatility",
    "er_kaufman",
    "adx",
    "minkowski_causal",
    "shannon",
    "pca_axis",
]
TIMEFRAMES_DEFAUT = ["H1", "H4"]

FRACTIONS_COUPE = (0.30, 0.50, 0.70, 0.90)
MIN_BARRES_COUPE = 400          # au-delà de la période de chauffe (~255 barres)

SEUIL_PASS = 99.0               # % d'accord miroir (indicatif)
SEUIL_WARN = 95.0

SWAP = {
    "HAUSSIER": "BAISSIER",
    "BAISSIER": "HAUSSIER",
    "TENDANCE_HAUSSIERE": "TENDANCE_BAISSIERE",
    "TENDANCE_BAISSIERE": "TENDANCE_HAUSSIERE",
}


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def lancer(methode: str, tf: str, df: pd.DataFrame) -> pd.Series:
    """Run d'une méthode, régimes indexés par timestamp (NaN -> INCONNU)."""
    det = METHOD_REGISTRY[methode](timeframe=tf)
    out = det.fit_predict(df.copy())
    if len(out) != len(df):
        raise ValueError(
            f"{methode}/{tf} : {len(out)} sorties pour {len(df)} barres "
            "(alignement ligne à ligne requis)"
        )
    return pd.Series(out["regime"].values, index=df.index).fillna("INCONNU").astype(str)


def miroir(df: pd.DataFrame, mode: str = "arith") -> pd.DataFrame:
    """
    Reflet des prix.

    arith : p' = 2K - p, K = centre arithmétique de la plage (conserve les
            différences de prix).
    log   : p' = K² / p, K = centre géométrique de la plage (inverse
            exactement les log-rendements).
    """
    m = df.copy()
    if mode == "arith":
        k = (df["low"].min() + df["high"].max()) / 2.0
        m["open"] = 2 * k - df["open"]
        m["close"] = 2 * k - df["close"]
        m["high"] = 2 * k - df["low"]
        m["low"] = 2 * k - df["high"]
    elif mode == "log":
        k2 = df["low"].min() * df["high"].max()   # K² avec K = sqrt(min * max)
        m["open"] = k2 / df["open"]
        m["close"] = k2 / df["close"]
        m["high"] = k2 / df["low"]
        m["low"] = k2 / df["high"]
    else:
        raise ValueError(f"mode de miroir inconnu : {mode}")
    return m


# ---------------------------------------------------------------------------
# Test 1 : causalité par troncature
# ---------------------------------------------------------------------------
def test_troncature(methode: str, tf: str, df: pd.DataFrame, full: pd.Series) -> dict:
    lignes = []
    for frac in FRACTIONS_COUPE:
        n_cut = int(len(df) * frac)
        if n_cut < MIN_BARRES_COUPE:
            continue
        tronque = lancer(methode, tf, df.iloc[:n_cut])
        ref = full.iloc[:n_cut]
        diff = tronque.values != ref.values
        idx = np.flatnonzero(diff)
        dist_fin = (n_cut - 1 - idx) if len(idx) else np.array([], dtype=int)
        lignes.append({
            "frac": frac,
            "n_cut": n_cut,
            "n_diff": int(diff.sum()),
            "premier": df.index[idx[0]] if len(idx) else None,
            "dist_fin_med": float(np.median(dist_fin)) if len(idx) else float("nan"),
            "dist_fin_max": int(dist_fin.max()) if len(idx) else 0,
        })
    total = sum(l["n_diff"] for l in lignes)
    return {"lignes": lignes, "total_diff": total, "verdict": "PASS" if total == 0 else "FAIL"}


# ---------------------------------------------------------------------------
# Test 2 : miroir
# ---------------------------------------------------------------------------
def test_miroir(methode: str, tf: str, df: pd.DataFrame, full: pd.Series,
                mode: str = "arith") -> dict:
    attendu = full.map(lambda lab: SWAP.get(lab, lab))
    obtenu = lancer(methode, tf, miroir(df, mode))

    classe = (attendu != "INCONNU") & (obtenu != "INCONNU")
    n_classe = int(classe.sum())
    accord = 100 * float((attendu[classe] == obtenu[classe]).mean()) if n_classe else float("nan")

    par_label = {
        lab: 100 * float((obtenu[classe & (attendu == lab)] == lab).mean())
        for lab in sorted(set(attendu[classe]))
    }
    inconnu_diff = int(((attendu == "INCONNU") != (obtenu == "INCONNU")).sum())
    directionnel = any(lab in SWAP for lab in set(full))

    if np.isnan(accord):
        verdict = "FAIL"
    elif accord >= SEUIL_PASS:
        verdict = "PASS"
    elif accord >= SEUIL_WARN:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    return {
        "accord": accord,
        "n_classe": n_classe,
        "par_label": par_label,
        "inconnu_diff": inconnu_diff,
        "directionnel": directionnel,
        "dist_orig": full[full != "INCONNU"].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "dist_miroir": obtenu[obtenu != "INCONNU"].value_counts(normalize=True).mul(100).round(1).to_dict(),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=str(URL))
    ap.add_argument("--timeframes", nargs="+", default=TIMEFRAMES_DEFAUT)
    ap.add_argument("--methodes", nargs="+", default=METHODES_DEFAUT)
    ap.add_argument("--miroir", choices=["arith", "log"], default="arith",
                    help="variante du reflet de prix (défaut : arith)")
    ap.add_argument("--sans-troncature", action="store_true",
                    help="saute le test de causalité (plus rapide)")
    args = ap.parse_args()

    print("=" * 78)
    print("  TESTS D'INVARIANCE regime_lib")
    print("=" * 78)
    print()

    print(f"Variante de miroir : {args.miroir}"
          f"{'  (troncature sautée)' if args.sans_troncature else ''}")
    print("Chargement...")
    df_m1, _ = load_parquet(args.url, batch_size=1_000_000)
    print(f"  M1 barres : {len(df_m1):,}")
    print()

    synthese: list[tuple[str, str, str, str, str]] = []
    echec = False

    for tf in args.timeframes:
        df_tf = resample(df_m1, tf)
        if df_tf.index.tz is None:
            df_tf.index = df_tf.index.tz_localize("UTC")
        print("─" * 78)
        print(f"  {tf} — {len(df_tf):,} barres")
        print("─" * 78)

        for methode in args.methodes:
            print(f"\n  [{methode}]")
            full = lancer(methode, tf, df_tf)

            # Test 1
            if args.sans_troncature:
                t1 = {"lignes": [], "total_diff": 0, "verdict": "SKIP"}
            else:
                t1 = test_troncature(methode, tf, df_tf, full)
            for l in t1["lignes"]:
                if l["n_diff"] == 0:
                    print(f"    troncature {l['frac']:.0%} ({l['n_cut']:,} barres) : identique")
                else:
                    print(f"    troncature {l['frac']:.0%} ({l['n_cut']:,} barres) : "
                          f"{l['n_diff']:,} écarts, premier {l['premier']}, "
                          f"distance à la coupure médiane={l['dist_fin_med']:.0f} "
                          f"max={l['dist_fin_max']} barres")

            # Test 2
            t2 = test_miroir(methode, tf, df_tf, full, args.miroir)
            print(f"    miroir : accord {t2['accord']:.2f} % sur {t2['n_classe']:,} barres classées "
                  f"(INCONNU différents : {t2['inconnu_diff']:,})")
            if t2["par_label"]:
                detail = "  ".join(f"{k}={v:.1f}%" for k, v in t2["par_label"].items())
                print(f"      accord par label attendu : {detail}")
            if t2["directionnel"]:
                print(f"      répartition originale : {t2['dist_orig']}")
                print(f"      répartition miroir    : {t2['dist_miroir']}")
            else:
                print("      méthode non directionnelle : les labels doivent rester inchangés")

            synthese.append((
                tf, methode,
                ("SKIP" if t1["verdict"] == "SKIP"
                 else f"{t1['verdict']} ({t1['total_diff']} écarts)"),
                f"{t2['verdict']} ({t2['accord']:.1f} %)",
                "",
            ))
            echec = echec or t1["verdict"] == "FAIL" or t2["verdict"] == "FAIL"
        print()

    print("=" * 78)
    print("  SYNTHÈSE")
    print("=" * 78)
    titre_miroir = f"miroir ({args.miroir})"
    print(f"  {'TF':4} {'méthode':18} {'troncature (causalité)':28} {titre_miroir:20}")
    for tf, methode, t1, t2, _ in synthese:
        print(f"  {tf:4} {methode:18} {t1:28} {t2:20}")
    print()
    print(f"  Seuils miroir (indicatifs) : PASS >= {SEUIL_PASS:.0f} %, WARN >= {SEUIL_WARN:.0f} %")
    return 1 if echec else 0


if __name__ == "__main__":
    sys.exit(main())
