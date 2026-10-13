"""Calibration atr_volatility EURUSD M5 — voir PROTOCOLE.md (figé avant calcul).

Usage : .venv/bin/python analyses/eurusd_m5_atr_volatility/calibrer.py
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import regime_lib
import regime_lib.methods  # noqa: F401 — enregistre les détecteurs
from regime_lib.core.data_loader import agreger_tick, resample
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.study import calculer_cible
from regime_lib.utils.validation import validity_mask

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "actifs/EURUSD/EURUSD_Tick_20231120_20261010.parquet"
SHA256 = "51ef04fa5ed3ae6e0e001f5eb745340d82376880b7c950008b9af41209674df7"
OUT = Path(__file__).resolve().parent / "output"

LECTURE = (pd.Timestamp("2023-11-20", tz="UTC"), pd.Timestamp("2025-01-10", tz="UTC"))
EVAL = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC"))
BLOCS = list(pd.date_range("2024-01-01", "2025-01-01", freq="2MS", tz="UTC"))
TF, K, STEP = "M5", 12, pd.Timedelta(minutes=5)
N_ATR = [3, 5, 7, 14, 20, 40]
FENETRE = [1500, 3000, 6000]
FIXES = dict(q_low=0.33, q_high=0.67, n_direction=20)
N_MIN = 30
HAUT = ("TENDANCE_HAUSSIERE", "TENDANCE_BAISSIERE")
BAS = ("RANGE",)
HEURES_HAUT = list(range(3, 12))
HEURES_BAS = [16, 17, 18, 19, 20, 21, 22, 23, 0, 1]


def charger_barres() -> pd.DataFrame:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Source modifiée : sha256 {digest}")
    table = pq.read_table(SOURCE, filters=[("timestamp", ">=", LECTURE[0]),
                                           ("timestamp", "<", LECTURE[1])])
    ticks = table.to_pandas()
    if "timestamp" in ticks.columns:
        ticks = ticks.set_index("timestamp")
    m1 = agreger_tick(ticks, "M1", prix="bid", combler_sans_tick="60min")
    return resample(m1, TF)


def score(df: pd.DataFrame) -> dict:
    """S global (critère principal) et S stratifié par heure (diagnostic)."""
    groupe = np.where(df.regime.isin(HAUT), "HAUT", np.where(df.regime.isin(BAS), "BAS", None))
    d = df.assign(groupe=groupe).dropna(subset=["groupe"])
    brut = d.groupby("groupe").cible.median()
    if "HAUT" not in brut or "BAS" not in brut:
        return dict(S=np.nan)
    med = d.groupby(["heure", "groupe"]).cible.median().unstack()
    n = d.groupby(["heure", "groupe"]).size().unstack().fillna(0)
    poids = n.min(axis=1) if {"HAUT", "BAS"} <= set(n.columns) else pd.Series(dtype=float)
    retenues = poids[poids >= N_MIN].index
    stratifie = np.nan
    if len(retenues):
        d_h = np.log(med.loc[retenues, "HAUT"]) - np.log(med.loc[retenues, "BAS"])
        stratifie = float(np.average(d_h, weights=poids[retenues]))
    return dict(
        S=float(np.log(brut["HAUT"]) - np.log(brut["BAS"])),
        S_stratifie=stratifie, strates=int(len(retenues)),
        n_haut=int((d.groupe == "HAUT").sum()), n_bas=int((d.groupe == "BAS").sum()),
    )


def horloge(heure: np.ndarray) -> np.ndarray:
    """Référence sans prix : HAUT 3 h–11 h, BAS 16 h–1 h (New York), CHOP sinon."""
    return np.where(np.isin(heure, HEURES_HAUT), HAUT[0],
                    np.where(np.isin(heure, HEURES_BAS), BAS[0], "CHOP"))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    bars = charger_barres()
    cible = calculer_cible(bars, f"vol_fwd_{K}", timeframe=TF)
    idx = bars.index.to_series()
    fin_cible = idx.shift(-K)
    contigu = (fin_cible - idx) == K * STEP
    heure = (bars.index - STEP).tz_convert("America/New_York").hour
    dans_eval = (bars.index >= EVAL[0]) & (bars.index < EVAL[1])
    avant = int((bars.index < EVAL[0]).sum())
    if avant < max(FENETRE) + 1:
        raise SystemExit(f"Warmup insuffisant : {avant} barres avant 2024.")

    lignes, par_bar = [], {}
    for n_atr, fenetre in itertools.product(N_ATR, FENETRE):
        det = METHOD_REGISTRY["atr_volatility"](timeframe=TF, n_atr=n_atr, fenetre=fenetre, **FIXES)
        regimes = det.fit_predict(bars)
        valide = validity_mask(regimes) & cible.notna() & contigu
        df = pd.DataFrame({"regime": regimes.regime, "cible": cible, "heure": heure,
                           "fin": fin_cible})[valide & dans_eval]
        par_bar[(n_atr, fenetre)] = df
        couverture = float(valide[dans_eval].mean())
        lignes.append(dict(n_atr=n_atr, fenetre=fenetre, couverture=couverture, **score(df)))
    ref = pd.DataFrame({"regime": horloge(np.asarray(heure)), "cible": cible, "heure": heure,
                        "fin": fin_cible})[cible.notna() & contigu & dans_eval & ~bars["is_partial"]]
    S_horloge = score(ref)["S"]
    resultats = pd.DataFrame(lignes).sort_values("S", ascending=False)
    resultats["exces_horloge"] = resultats["S"] - S_horloge
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
            n_atr=choix[0], fenetre=choix[1],
            S_train=train[choix]["S"], S_test=test[choix]["S"],
            rang_test=int(s_test.rank(ascending=False)[choix]),
            regret=float(s_test.max() - s_test[choix]),
            meilleur_test=str(s_test.idxmax()),
            S_horloge_test=score(ref[(ref.index >= debut) & (ref.index < fin)])["S"],
        ))
    plis = pd.DataFrame(plis)
    plis.to_csv(OUT / "validation_glissante.csv", index=False)

    final = tuple(resultats.iloc[0][["n_atr", "fenetre"]].astype(int))
    df = par_bar[final]
    groupe = np.where(df.regime.isin(HAUT), "HAUT", np.where(df.regime.isin(BAS), "BAS", "CHOP"))
    par_heure = (df.assign(groupe=groupe).groupby(["heure", "groupe"]).cible
                 .agg(["size", "median"]).unstack())
    par_heure.to_csv(OUT / "par_heure_retenu.csv")

    provenance = dict(source=str(SOURCE.relative_to(ROOT)), sha256=SHA256,
                      regime_lib=regime_lib.__version__, pandas=pd.__version__,
                      barres=len(bars), barres_warmup=avant, S_horloge_2024=S_horloge,
                      retenu=dict(n_atr=final[0], fenetre=final[1]), fixes=FIXES)
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(resultats.to_string(index=False))
        print()
        print(plis.to_string(index=False))
        print()
        print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
