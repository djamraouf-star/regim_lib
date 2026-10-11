"""Calibration adx EURUSD M5 — voir PROTOCOLE.md (figé avant calcul).

Usage : .venv/bin/python analyses/eurusd_m5_adx/calibrer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import regime_lib
import regime_lib.methods  # noqa: F401 — enregistre les détecteurs
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.methods.trend.adx import _compute_adx
from regime_lib.utils.validation import validity_mask

# Données, blocs et période : strictement ceux de la calibration ATR M5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eurusd_m5_atr_volatility"))
from calibrer import BLOCS, EVAL, SHA256, SOURCE, STEP, TF, charger_barres  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
K = 12
N_ADX = [5, 7, 14, 28, 56]
Q_RANGE, Q_TENDANCE = 0.40, 0.60
N_MIN = 30
TENDANCE = ("TENDANCE_HAUSSIERE", "TENDANCE_BAISSIERE")


def efficience_future(close: pd.Series, k: int) -> pd.Series:
    """|close[t+k] − close[t]| / Σ |Δclose| sur t+1…t+k ; NaN si chemin nul."""
    chemin = close.diff().abs().rolling(k, min_periods=k).sum().shift(-k)
    net = (close.shift(-k) - close).abs()
    return net / chemin.where(chemin > 0)


def score(df: pd.DataFrame) -> dict:
    """S global TENDANCE vs RANGE (critère) ; diagnostics."""
    groupe = np.where(df.regime.isin(TENDANCE), "TENDANCE",
                      np.where(df.regime.eq("RANGE"), "RANGE", None))
    d = df.assign(groupe=groupe).dropna(subset=["groupe"])
    brut = d.groupby("groupe").cible.median()
    if "TENDANCE" not in brut or "RANGE" not in brut:
        return dict(S=np.nan)
    med = d.groupby(["heure", "groupe"]).cible.median().unstack()
    n = d.groupby(["heure", "groupe"]).size().unstack().fillna(0)
    poids = n.min(axis=1)
    retenues = poids[poids >= N_MIN].index
    stratifie = np.nan
    if len(retenues):
        d_h = np.log(med.loc[retenues, "TENDANCE"]) - np.log(med.loc[retenues, "RANGE"])
        stratifie = float(np.average(d_h, weights=poids[retenues]))
    t = df[df.regime.isin(TENDANCE) & df.ret.ne(0)]
    annonce = np.where(t.regime.eq("TENDANCE_HAUSSIERE"), 1.0, -1.0)
    parts = df.regime.value_counts(normalize=True)
    return dict(
        S=float(np.log(brut["TENDANCE"]) - np.log(brut["RANGE"])),
        S_stratifie=stratifie, strates=int(len(retenues)),
        accord=float((np.sign(t.ret) == annonce).mean()) if len(t) else np.nan,
        part_tendance=float(parts.reindex(TENDANCE).fillna(0).sum()),
        part_range=float(parts.get("RANGE", 0.0)), part_chop=float(parts.get("CHOP", 0.0)),
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
    ts = bars.index
    dans_eval = (ts >= EVAL[0]) & (ts < EVAL[1])
    non_partiel = ~bars["is_partial"].to_numpy(dtype=bool)

    def evaluer(n_adx: int, adx: np.ndarray, debut_seuil, fin_seuil) -> tuple[dict, pd.DataFrame]:
        """Seuils = quantiles de l'ADX valide sur [debut_seuil, fin_seuil)."""
        base = non_partiel & ~np.isnan(adx) & (ts >= debut_seuil) & (ts < fin_seuil)
        seuils = dict(seuil_range=round(float(np.quantile(adx[base], Q_RANGE)), 1),
                      seuil_tendance=round(float(np.quantile(adx[base], Q_TENDANCE)), 1))
        det = METHOD_REGISTRY["adx"](timeframe=TF, n_adx=n_adx, **seuils)
        regimes = det.fit_predict(bars)
        valide = validity_mask(regimes) & cible.notna() & contigu
        df = pd.DataFrame({"regime": regimes.regime, "cible": cible, "ret": ret,
                           "heure": heure, "fin": fin_cible})[valide & dans_eval]
        return seuils, df

    adx_par_n = {}
    for n_adx in N_ADX:
        adx_par_n[n_adx] = _compute_adx(bars["high"].to_numpy(float), bars["low"].to_numpy(float),
                                        close.to_numpy(float), n_adx)[2]

    lignes, par_bar = [], {}
    for n_adx in N_ADX:
        seuils, df = evaluer(n_adx, adx_par_n[n_adx], *EVAL)
        par_bar[n_adx] = df
        lignes.append(dict(n_adx=n_adx, **seuils, **score(df)))
    resultats = pd.DataFrame(lignes).sort_values("S", ascending=False)
    resultats.to_csv(OUT / "resultats_2024.csv", index=False)

    plis = []
    for f in range(1, len(BLOCS) - 1):
        debut, fin = BLOCS[f], BLOCS[f + 1]
        train, test, seuils_pli = {}, {}, {}
        for n_adx in N_ADX:
            seuils, df = evaluer(n_adx, adx_par_n[n_adx], EVAL[0], debut)
            seuils_pli[n_adx] = seuils
            train[n_adx] = score(df[(df.index < debut) & (df.fin <= debut)])
            test[n_adx] = score(df[(df.index >= debut) & (df.index < fin)])
        choix = max(train, key=lambda c: train[c]["S"])
        s_test = pd.Series({c: v["S"] for c, v in test.items()})
        plis.append(dict(
            test=f"{debut:%Y-%m}..{fin - pd.Timedelta(days=1):%Y-%m}",
            n_adx=choix, **seuils_pli[choix],
            S_train=train[choix]["S"], S_test=test[choix]["S"],
            rang_test=int(s_test.rank(ascending=False)[choix]),
            regret=float(s_test.max() - s_test[choix]), meilleur_test=int(s_test.idxmax()),
            accord_test=test[choix]["accord"], part_tendance_test=test[choix]["part_tendance"],
        ))
    plis = pd.DataFrame(plis)
    plis.to_csv(OUT / "validation_glissante.csv", index=False)

    final = resultats.iloc[0]
    provenance = dict(source=SOURCE.name, sha256=SHA256, regime_lib=regime_lib.__version__,
                      pandas=pd.__version__, barres=len(bars), K=K,
                      retenu=dict(n_adx=int(final.n_adx), seuil_range=float(final.seuil_range),
                                  seuil_tendance=float(final.seuil_tendance)))
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(resultats.to_string(index=False))
        print()
        print(plis.to_string(index=False))
        print()
        print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
