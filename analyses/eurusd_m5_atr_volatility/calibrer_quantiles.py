"""Calibration de q_low, q_high et fenetre sur le critère de fidélité (n_atr=7).

Étalon : volatilité réalisée centrée (demi-fenêtre K). Ses classes HAUT/BAS
sont définies avec les MÊMES quantiles (q_low, q_high) que le détecteur, pour
comparer des proportions identiques. Score : kappa de Cohen sur 2024.
Usage : .venv/bin/python analyses/eurusd_m5_atr_volatility/calibrer_quantiles.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrer as c  # noqa: E402
import fidelite as f  # noqa: E402
from regime_lib.core.registry import METHOD_REGISTRY  # noqa: E402
from regime_lib.core.utils import direction, true_range  # noqa: E402

N_ATR = 7
KS = [6, 9]
Q_LOW = [0.20, 0.25, 0.30, 0.33, 0.40, 0.45]
Q_HIGH = [0.55, 0.60, 0.67, 0.70, 0.75, 0.80]
FEN = [1500, 3000, 6000, 8000]


def classes(atr, s_low, s_high, dirn, pret):
    g = np.full(len(atr), "CHOP", dtype=object)
    g[atr <= s_low] = "BAS"
    g[(atr >= s_high) & (dirn != 0)] = "HAUT"
    return pd.Series(np.where(pret, g, None), index=atr.index)


def main() -> None:
    bars = c.charger_barres()
    ok = ~bars["is_partial"].to_numpy()
    dans = (bars.index >= c.EVAL[0]) & (bars.index < c.EVAL[1])
    h, l, cl = (bars[x].to_numpy(float) for x in ("high", "low", "close"))
    atr = pd.Series(true_range(h, l, cl), index=bars.index).rolling(N_ATR, min_periods=1).mean()
    dirn = direction(cl, c.FIXES["n_direction"])
    past = atr.shift(1)
    seuils = {(fe, q): past.rolling(fe, min_periods=fe // 2).quantile(q)
              for fe in FEN for q in set(Q_LOW + Q_HIGH)}

    # contrôle : la classification locale reproduit le détecteur
    det = METHOD_REGISTRY["atr_volatility"](timeframe=c.TF, n_atr=N_ATR, fenetre=6000, **c.FIXES).fit_predict(bars)
    gl = classes(atr, seuils[(6000, .33)], seuils[(6000, .67)], dirn,
                 (seuils[(6000, .33)].notna() & seuils[(6000, .67)].notna()).to_numpy())
    ref_det = det.regime.map(lambda x: "HAUT" if x in c.HAUT else ("BAS" if x == "RANGE" else "CHOP"))
    v = gl.notna() & (det.regime != "INCONNU") & ok
    assert (gl[v] == ref_det[v]).mean() > 0.999, (gl[v] == ref_det[v]).mean()

    blocs = list(zip(c.BLOCS[:-1], c.BLOCS[1:]))
    lignes = []
    for k in KS:
        ref = f.etalon(bars.close, k)
        e = ref[dans & ref.notna().to_numpy()]
        for fe, ql, qh in itertools.product(FEN, Q_LOW, Q_HIGH):
            tl, th = e.quantile(ql), e.quantile(qh)
            et = pd.Series(np.where(ref >= th, "HAUT", np.where(ref <= tl, "BAS", "CHOP")), index=bars.index)
            pret = (seuils[(fe, ql)].notna() & seuils[(fe, qh)].notna()).to_numpy()
            g = classes(atr, seuils[(fe, ql)], seuils[(fe, qh)], dirn, pret)
            m = dans & ok & pret & ref.notna().to_numpy()
            kb = [f.kappa(g[m & (bars.index >= a) & (bars.index < b)], et[m & (bars.index >= a) & (bars.index < b)])
                  for a, b in blocs]
            lignes.append(dict(k=k, fenetre=fe, q_low=ql, q_high=qh, kappa=f.kappa(g[m], et[m]),
                               kappa_bloc_min=min(kb), part_haut=float((g[m] == "HAUT").mean()),
                               part_bas=float((g[m] == "BAS").mean()),
                               bascules_jour=float((g[m] != g[m].shift()).groupby(g[m].index.date).sum().mean())))
    r = pd.DataFrame(lignes)
    r.to_csv(c.OUT / "calibration_quantiles.csv", index=False)
    pd.set_option("display.width", 200)
    for k, d in r.groupby("k"):
        print(f"\n=== k={k} : top 8 ===")
        print(d.sort_values("kappa", ascending=False).head(8).round(3).to_string(index=False))
        print(f"-- défaut (0.33/0.67, 6000) :",
              d[(d.q_low == .33) & (d.q_high == .67) & (d.fenetre == 6000)].kappa.round(3).tolist())
        print("-- moyenne κ par fenetre :", d.groupby("fenetre").kappa.mean().round(3).to_dict())
        print("-- moyenne κ par q_low :", d.groupby("q_low").kappa.mean().round(3).to_dict())
        print("-- moyenne κ par q_high :", d.groupby("q_high").kappa.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
