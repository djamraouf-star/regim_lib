"""Calibration er_kaufman EURUSD M5 — voir PROTOCOLE.md (figé avant calcul).

Usage : .venv/bin/python analyses/eurusd_m5_er_kaufman/calibrer.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import regime_lib
import regime_lib.methods  # noqa: F401 — enregistre les détecteurs
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.utils.validation import validity_mask

# Données, blocs et période : strictement ceux de la calibration ATR M5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eurusd_m5_atr_volatility"))
from calibrer import BLOCS, EVAL, SHA256, SOURCE, STEP, TF, charger_barres  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
K = 12
N_ER = [5, 10, 20, 40, 80]
FENETRE = [1500, 3000, 6000]
FIXES = dict(q_chop=0.40, q_tendance=0.60)
N_MIN = 30
TENDANCE = ("TENDANCE_HAUSSIERE", "TENDANCE_BAISSIERE")


def efficience_future(close: pd.Series, k: int) -> pd.Series:
    """|close[t+k] − close[t]| / Σ |Δclose| sur t+1…t+k ; NaN si chemin nul."""
    chemin = close.diff().abs().rolling(k, min_periods=k).sum().shift(-k)
    net = (close.shift(-k) - close).abs()
    return net / chemin.where(chemin > 0)


def score(df: pd.DataFrame) -> dict:
    """S global (critère) ; S stratifié et accord directionnel (diagnostics)."""
    groupe = np.where(df.regime.isin(TENDANCE), "TENDANCE",
                      np.where(df.regime.eq("CHOP"), "CHOP", None))
    d = df.assign(groupe=groupe).dropna(subset=["groupe"])
    brut = d.groupby("groupe").cible.median()
    if "TENDANCE" not in brut or "CHOP" not in brut:
        return dict(S=np.nan)
    med = d.groupby(["heure", "groupe"]).cible.median().unstack()
    n = d.groupby(["heure", "groupe"]).size().unstack().fillna(0)
    poids = n.min(axis=1)
    retenues = poids[poids >= N_MIN].index
    stratifie = np.nan
    if len(retenues):
        d_h = np.log(med.loc[retenues, "TENDANCE"]) - np.log(med.loc[retenues, "CHOP"])
        stratifie = float(np.average(d_h, weights=poids[retenues]))
    t = df[df.regime.isin(TENDANCE) & df.ret.ne(0)]
    annonce = np.where(t.regime.eq("TENDANCE_HAUSSIERE"), 1.0, -1.0)
    accord = float((np.sign(t.ret) == annonce).mean()) if len(t) else np.nan
    return dict(
        S=float(np.log(brut["TENDANCE"]) - np.log(brut["CHOP"])),
        S_stratifie=stratifie, strates=int(len(retenues)),
        accord=accord, accord_ic95=float(1.96 * np.sqrt(0.25 / max(len(t), 1))),
        n_tendance=int((d.groupe == "TENDANCE").sum()), n_chop=int((d.groupe == "CHOP").sum()),
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bars = charger_barres()
    close = bars["close"]
    cible = efficience_future(close, K)
    ret = np.log(close.shift(-K) / close)
    idx = bars.index.to_series()
    fin_cible = idx.shift(-K)
    contigu = (fin_cible - idx) == K * STEP
    heure = (bars.index - STEP).tz_convert("America/New_York").hour
    dans_eval = (bars.index >= EVAL[0]) & (bars.index < EVAL[1])
    avant = int((bars.index < EVAL[0]).sum())
    if avant < max(FENETRE) + max(N_ER) + 1:
        raise SystemExit(f"Warmup insuffisant : {avant} barres avant 2024.")

    lignes, par_bar = [], {}
    for n_er, fenetre in itertools.product(N_ER, FENETRE):
        det = METHOD_REGISTRY["er_kaufman"](timeframe=TF, n_er=n_er, fenetre=fenetre, **FIXES)
        regimes = det.fit_predict(bars)
        valide = validity_mask(regimes) & cible.notna() & contigu
        df = pd.DataFrame({"regime": regimes.regime, "cible": cible, "ret": ret,
                           "heure": heure, "fin": fin_cible})[valide & dans_eval]
        par_bar[(n_er, fenetre)] = df
        couverture = float(valide[dans_eval].mean())
        lignes.append(dict(n_er=n_er, fenetre=fenetre, couverture=couverture, **score(df)))
    resultats = pd.DataFrame(lignes).sort_values("S", ascending=False)
    resultats.to_csv(OUT / "resultats_2024.csv", index=False)

    plis = []
    for f in range(1, len(BLOCS) - 1):
        debut, fin = BLOCS[f], BLOCS[f + 1]
        train = {c: score(df[(df.index < debut) & (df.fin <= debut)]) for c, df in par_bar.items()}
        test = {c: score(df[(df.index >= debut) & (df.index < fin)]) for c, df in par_bar.items()}
        choix = max(train, key=lambda c: train[c]["S"])
        s_test = pd.Series({c: v["S"] for c, v in test.items()})
        plis.append(dict(
            test=f"{debut:%Y-%m}..{fin - pd.Timedelta(days=1):%Y-%m}",
            n_er=choix[0], fenetre=choix[1],
            S_train=train[choix]["S"], S_test=test[choix]["S"],
            rang_test=int(s_test.rank(ascending=False)[choix]),
            regret=float(s_test.max() - s_test[choix]),
            meilleur_test=f"{s_test.idxmax()[0]}/{s_test.idxmax()[1]}",
            accord_test=test[choix]["accord"],
        ))
    plis = pd.DataFrame(plis)
    plis.to_csv(OUT / "validation_glissante.csv", index=False)

    final = tuple(int(v) for v in resultats.iloc[0][["n_er", "fenetre"]])
    df = par_bar[final]
    par_heure = (df.groupby(["heure", "regime"]).cible.agg(["size", "median"]).unstack())
    par_heure.to_csv(OUT / "par_heure_retenu.csv")

    provenance = dict(source=str(SOURCE.name), sha256=SHA256,
                      regime_lib=regime_lib.__version__, pandas=pd.__version__,
                      barres=len(bars), barres_warmup=avant, K=K,
                      retenu=dict(n_er=final[0], fenetre=final[1]), fixes=FIXES)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(resultats.to_string(index=False))
        print()
        print(plis.to_string(index=False))
        print()
        print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
