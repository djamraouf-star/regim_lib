"""Fidélité de atr_volatility à l'état courant du marché (EURUSD M5, 2024).

Étalon avec lookahead : volatilité réalisée close-à-close sur une fenêtre
CENTRÉE [t-k, t+k] (estimateur différent de l'ATR, basé sur high/low).
Usage : .venv/bin/python analyses/eurusd_m5_atr_volatility/fidelite.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrer as c  # noqa: E402
from regime_lib.core.registry import METHOD_REGISTRY  # noqa: E402
from regime_lib.core.utils import true_range  # noqa: E402
from regime_lib.utils.validation import validity_mask  # noqa: E402

K_LIST = [6, 12, 24]
LAGS = range(0, 13, 2)  # retard de l'indicateur, en barres
N_ATR = [3, 5, 7, 10, 14, 20, 40]


def etalon(close: pd.Series, k: int) -> pd.Series:
    r = np.log(close / close.shift(1))
    return r.rolling(2 * k + 1, center=True, min_periods=2 * k + 1).std()


def desaisonnaliser(x: pd.Series, heure) -> pd.Series:
    return x / x.groupby(heure).transform("median")


def kappa(a: pd.Series, b: pd.Series) -> float:
    t = pd.crosstab(a, b).to_numpy(dtype=float)
    n = t.sum()
    po = np.trace(t) / n
    pe = (t.sum(0) * t.sum(1)).sum() / n**2
    return float((po - pe) / (1 - pe))


def main() -> None:
    bars = c.charger_barres()
    heure = pd.Series((bars.index - c.STEP).tz_convert("America/New_York").hour, index=bars.index)
    ok = ~bars["is_partial"]
    dans = (bars.index >= c.EVAL[0]) & (bars.index < c.EVAL[1])
    tr = pd.Series(true_range(bars.high.to_numpy(float), bars.low.to_numpy(float),
                              bars.close.to_numpy(float)), index=bars.index)

    # --- 1) continu : corrélation de rang indicateur(t-lag) vs étalon(t) ---
    lignes = []
    for k in K_LIST:
        ref = etalon(bars.close, k)
        for n_atr in N_ATR:
            atr = tr.rolling(n_atr, min_periods=n_atr).mean()
            for desais in (False, True):
                a = desaisonnaliser(atr, heure) if desais else atr
                r = desaisonnaliser(ref, heure) if desais else ref
                best = {}
                for lag in LAGS:
                    d = pd.DataFrame({"a": a.shift(lag), "r": r})[dans & ok.to_numpy()].dropna()
                    best[lag] = d.a.corr(d.r, method="spearman")
                lag0 = best[0]
                lag_opt = max(best, key=best.get)
                lignes.append(dict(k=k, n_atr=n_atr, desaisonnalise=desais,
                                   rho_lag0=lag0, rho_max=best[lag_opt], lag_opt=lag_opt))
    cont = pd.DataFrame(lignes)
    cont.to_csv(c.OUT / "fidelite_continu.csv", index=False)

    # --- 2) régimes : accord avec les terciles de l'étalon ---
    lignes = []
    for k in K_LIST:
        ref = etalon(bars.close, k)
        e = ref[dans]
        q1, q2 = e.quantile([1 / 3, 2 / 3])
        et = pd.Series(np.where(ref >= q2, "HAUT", np.where(ref <= q1, "BAS", "CHOP")), index=bars.index)
        for n_atr, fen in itertools.product(N_ATR, c.FENETRE):
            det = METHOD_REGISTRY["atr_volatility"](timeframe=c.TF, n_atr=n_atr, fenetre=fen, **c.FIXES)
            reg = det.fit_predict(bars)
            g = reg.regime.map(lambda x: "HAUT" if x in c.HAUT else ("BAS" if x == "RANGE" else "CHOP"))
            m = validity_mask(reg) & ref.notna() & dans & ok.to_numpy()
            jours = (g[m] != g[m].shift()).groupby(g[m].index.date).sum().mean()
            lignes.append(dict(k=k, n_atr=n_atr, fenetre=fen, kappa=kappa(g[m], et[m]),
                               accord=float((g[m] == et[m]).mean()), bascules_jour=float(jours)))
        eth = (et[dans & ref.notna()] != et[dans & ref.notna()].shift()).groupby(
            et[dans & ref.notna()].index.date).sum().mean()
        lignes.append(dict(k=k, n_atr=0, fenetre=0, kappa=np.nan, accord=np.nan, bascules_jour=float(eth)))
    cls = pd.DataFrame(lignes)
    cls.to_csv(c.OUT / "fidelite_regimes.csv", index=False)

    pd.set_option("display.width", 200)
    print("CONTINU (rho Spearman, lag 0 / max, retard optimal)")
    print(cont.pivot_table(index=["desaisonnalise", "n_atr"], columns="k",
                           values=["rho_lag0", "rho_max", "lag_opt"]).round(3))
    print("\nRÉGIMES (kappa), fenetre=6000 ; n_atr=0 : bascules/jour de l'étalon")
    print(cls[cls.fenetre.isin([6000, 0])].pivot_table(index="n_atr", columns="k",
          values=["kappa", "accord", "bascules_jour"]).round(3))


if __name__ == "__main__":
    main()
