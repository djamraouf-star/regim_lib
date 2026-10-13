"""Balayage de la demi-fenêtre k de l'étalon : n_atr optimal en fonction de k.
Usage : .venv/bin/python analyses/eurusd_m5_atr_volatility/balayage_k.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrer as c  # noqa: E402
import fidelite as f  # noqa: E402
from regime_lib.core.registry import METHOD_REGISTRY  # noqa: E402
from regime_lib.core.utils import true_range  # noqa: E402
from regime_lib.utils.validation import validity_mask  # noqa: E402

KS = list(range(3, 13))
NS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20, 25]


def main() -> None:
    bars = c.charger_barres()
    heure = pd.Series((bars.index - c.STEP).tz_convert("America/New_York").hour, index=bars.index)
    ok = ~bars["is_partial"].to_numpy()
    dans = (bars.index >= c.EVAL[0]) & (bars.index < c.EVAL[1])
    tr = pd.Series(true_range(bars.high.to_numpy(float), bars.low.to_numpy(float),
                              bars.close.to_numpy(float)), index=bars.index)
    atr = {n: tr.rolling(n, min_periods=n).mean() for n in NS}
    atr_d = {n: f.desaisonnaliser(a, heure) for n, a in atr.items()}
    g = {}
    for n in NS:
        reg = METHOD_REGISTRY["atr_volatility"](timeframe=c.TF, n_atr=n, fenetre=6000, **c.FIXES).fit_predict(bars)
        g[n] = (reg.regime.map(lambda x: "HAUT" if x in c.HAUT else ("BAS" if x == "RANGE" else "CHOP")),
                validity_mask(reg))
    lignes = []
    for k in KS:
        ref = f.etalon(bars.close, k)
        ref_d = f.desaisonnaliser(ref, heure)
        q1, q2 = ref[dans].quantile([1 / 3, 2 / 3])
        et = pd.Series(np.where(ref >= q2, "HAUT", np.where(ref <= q1, "BAS", "CHOP")), index=bars.index)
        m0 = dans & ok & ref.notna().to_numpy()
        for n in NS:
            d = pd.DataFrame({"a": atr[n], "r": ref, "ad": atr_d[n], "rd": ref_d})[m0].dropna()
            gn, vn = g[n]
            m = m0 & vn.to_numpy()
            lignes.append(dict(k=k, n_atr=n, rho=d.a.corr(d.r, method="spearman"),
                               rho_desais=d.ad.corr(d.rd, method="spearman"),
                               kappa=f.kappa(gn[m], et[m])))
    r = pd.DataFrame(lignes)
    r.to_csv(c.OUT / "balayage_k.csv", index=False)
    pd.set_option("display.width", 200)
    out = []
    for k, d in r.groupby("k"):
        top = lambda col: d.loc[d[col].idxmax(), "n_atr"]
        plat = lambda col: list(d[d[col] >= d[col].max() - 0.01].n_atr)
        out.append(dict(k=k, fenetre_min=(2 * k + 1) * 5, n_rho=top("rho"), n_kappa=top("kappa"),
                        n_desais=top("rho_desais"), plateau_kappa=plat("kappa"),
                        kappa_max=round(d.kappa.max(), 3), kappa_n7=round(d[d.n_atr == 7].kappa.iloc[0], 3)))
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
