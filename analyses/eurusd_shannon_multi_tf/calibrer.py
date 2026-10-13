"""Calibration shannon EURUSD M1 / M15 / H1 — voir eurusd_<tf>_shannon/PROTOCOLE.md.

Usage : .venv/bin/python analyses/eurusd_shannon_multi_tf/calibrer.py M15
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import regime_lib
import regime_lib.methods  # noqa: F401 — enregistre les détecteurs
from regime_lib.core.data_loader import agreger_tick, resample
from regime_lib.core.registry import METHOD_REGISTRY
from regime_lib.methods.entropy.shannon import _shannon_windowed
from regime_lib.utils.validation import validity_mask

# Période, blocs et source : strictement ceux de la calibration ATR M5.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eurusd_m5_atr_volatility"))
from calibrer import BLOCS, EVAL, LECTURE, SHA256, SOURCE  # noqa: E402

K = 12
Q_ORDONNE, Q_DESORDRE = 0.40, 0.60
N_MIN = 30
CONFIG = {
    "M1": dict(step=pd.Timedelta(minutes=1), fenetre=[50, 100, 200, 400], n_bins=[5, 10, 20]),
    "M15": dict(step=pd.Timedelta(minutes=15), fenetre=[50, 100, 200, 400], n_bins=[5, 10, 20]),
    "H1": dict(step=pd.Timedelta(hours=1), fenetre=[24, 50, 100, 200], n_bins=[5, 10]),
}


def charger_barres(tf: str) -> pd.DataFrame:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Source modifiée : sha256 {digest}")
    table = pq.read_table(SOURCE, filters=[("timestamp", ">=", LECTURE[0]),
                                           ("timestamp", "<", LECTURE[1])])
    ticks = table.to_pandas()
    if "timestamp" in ticks.columns:
        ticks = ticks.set_index("timestamp")
    m1 = agreger_tick(ticks, "M1", prix="bid", combler_sans_tick="60min")
    return m1 if tf == "M1" else resample(m1, tf)


def efficience_future(close: pd.Series, k: int) -> pd.Series:
    """|close[t+k] − close[t]| / Σ |Δclose| sur t+1…t+k ; NaN si chemin nul."""
    chemin = close.diff().abs().rolling(k, min_periods=k).sum().shift(-k)
    net = (close.shift(-k) - close).abs()
    return net / chemin.where(chemin > 0)


def score(df: pd.DataFrame) -> dict:
    """S global ORDONNE vs DESORDRE (critère) ; diagnostics."""
    d = df[df.regime.isin(["ORDONNE", "DESORDRE"])]
    brut = d.groupby("regime").cible.median()
    if len(brut) < 2:
        return dict(S=np.nan)
    med = d.groupby(["heure", "regime"]).cible.median().unstack()
    n = d.groupby(["heure", "regime"]).size().unstack().fillna(0)
    poids = n.min(axis=1)
    retenues = poids[poids >= N_MIN].index
    stratifie = np.nan
    if len(retenues):
        d_h = np.log(med.loc[retenues, "ORDONNE"]) - np.log(med.loc[retenues, "DESORDRE"])
        stratifie = float(np.average(d_h, weights=poids[retenues]))
    vol = d.groupby("regime").vol.median()
    parts = df.regime.value_counts(normalize=True)
    return dict(
        S=float(np.log(brut["ORDONNE"]) - np.log(brut["DESORDRE"])),
        S_stratifie=stratifie, strates=int(len(retenues)),
        rapport_vol=float(vol["DESORDRE"] / vol["ORDONNE"]),
        part_ordonne=float(parts.get("ORDONNE", 0.0)),
        part_desordre=float(parts.get("DESORDRE", 0.0)),
    )


def main(tf: str) -> None:
    cfg = CONFIG[tf]
    step = cfg["step"]
    out_dir = Path(__file__).resolve().parents[1] / f"eurusd_{tf.lower()}_shannon" / "output"
    out_dir.mkdir(exist_ok=True)
    bars = charger_barres(tf)
    close = bars["close"]
    cible = efficience_future(close, K)
    vol = np.log(close).diff().rolling(K, min_periods=K).std().shift(-K)
    idx = bars.index.to_series()
    fin_cible = idx.shift(-K)
    contigu = (fin_cible - idx) == K * step
    heure = (bars.index - step).tz_convert("America/New_York").hour
    ts = bars.index
    dans_eval = (ts >= EVAL[0]) & (ts < EVAL[1])
    partiel = bars["is_partial"].to_numpy(dtype=bool)

    lr = np.full(len(bars), np.nan)
    c = close.to_numpy(float)
    lr[1:] = np.log(c[1:] / c[:-1])

    fenetre_max = max(cfg["fenetre"])
    avant_eval = int((ts < EVAL[0]).sum())
    if avant_eval < fenetre_max:
        raise SystemExit(f"Warmup insuffisant : {avant_eval} barres < {fenetre_max}")

    h_par, valide_par = {}, {}
    for fen, nb in itertools.product(cfg["fenetre"], cfg["n_bins"]):
        h = _shannon_windowed(lr, fen, nb)
        h[partiel] = np.nan
        h_par[fen, nb] = h
        det = METHOD_REGISTRY["shannon"](timeframe=tf, fenetre=fen, n_bins=nb,
                                         seuil_ordonne=0.5, seuil_desordre=0.9)
        regimes = det.fit_predict(bars)
        valide_par[fen, nb] = validity_mask(regimes).to_numpy() & ~np.isnan(h)

    def evaluer(cle, debut_seuil, fin_seuil) -> tuple[dict, pd.DataFrame]:
        """Seuils = quantiles de H valide sur [debut_seuil, fin_seuil)."""
        h = h_par[cle]
        base = valide_par[cle] & (ts >= debut_seuil) & (ts < fin_seuil)
        seuils = dict(seuil_ordonne=round(float(np.quantile(h[base], Q_ORDONNE)), 2),
                      seuil_desordre=round(float(np.quantile(h[base], Q_DESORDRE)), 2))
        regime = np.where(h < seuils["seuil_ordonne"], "ORDONNE",
                          np.where(h > seuils["seuil_desordre"], "DESORDRE", "NEUTRE"))
        df = pd.DataFrame({"regime": regime, "cible": cible, "vol": vol,
                           "heure": heure, "fin": fin_cible}, index=ts)
        keep = valide_par[cle] & cible.notna().to_numpy() & contigu.to_numpy() & dans_eval
        return seuils, df[keep]

    lignes = []
    for cle in h_par:
        seuils, df = evaluer(cle, *EVAL)
        lignes.append(dict(fenetre=cle[0], n_bins=cle[1], **seuils, **score(df)))
    resultats = pd.DataFrame(lignes).sort_values("S", ascending=False)
    resultats.to_csv(out_dir / "resultats_2024.csv", index=False)

    plis = []
    for f in range(1, len(BLOCS) - 1):
        debut, fin = BLOCS[f], BLOCS[f + 1]
        train, test, seuils_pli = {}, {}, {}
        for cle in h_par:
            seuils, df = evaluer(cle, EVAL[0], debut)
            seuils_pli[cle] = seuils
            train[cle] = score(df[(df.index < debut) & (df.fin <= debut)])
            test[cle] = score(df[(df.index >= debut) & (df.index < fin)])
        choix = max(train, key=lambda k: train[k]["S"])
        s_test = pd.Series({k: v["S"] for k, v in test.items()})
        plis.append(dict(
            test=f"{debut:%Y-%m}..{fin - pd.Timedelta(days=1):%Y-%m}",
            fenetre=choix[0], n_bins=choix[1], **seuils_pli[choix],
            S_train=train[choix]["S"], S_test=test[choix]["S"],
            rang_test=int(s_test.rank(ascending=False)[choix]),
            regret=float(s_test.max() - s_test[choix]),
            meilleur_test=str(tuple(int(x) for x in s_test.idxmax())),
            rapport_vol_test=test[choix]["rapport_vol"],
        ))
    plis = pd.DataFrame(plis)
    plis.to_csv(out_dir / "validation_glissante.csv", index=False)

    final = resultats.iloc[0]
    cle = (int(final.fenetre), int(final.n_bins))
    det = METHOD_REGISTRY["shannon"](timeframe=tf, fenetre=cle[0], n_bins=cle[1],
                                     seuil_ordonne=float(final.seuil_ordonne),
                                     seuil_desordre=float(final.seuil_desordre))
    reg_det = det.fit_predict(bars).regime
    _, df_ctrl = evaluer(cle, *EVAL)
    assert (reg_det.loc[df_ctrl.index] == df_ctrl.regime).all(), "écart détecteur/script"

    provenance = dict(source=SOURCE.name, sha256=SHA256, regime_lib=regime_lib.__version__,
                      pandas=pd.__version__, timeframe=tf, barres=len(bars),
                      barres_eval=int(dans_eval.sum()), K=K,
                      retenu=dict(fenetre=cle[0], n_bins=cle[1],
                                  seuil_ordonne=float(final.seuil_ordonne),
                                  seuil_desordre=float(final.seuil_desordre)))
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(resultats.to_string(index=False))
        print()
        print(plis.to_string(index=False))
        print()
        print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIG:
        raise SystemExit(f"Usage : calibrer.py {{{'|'.join(CONFIG)}}}")
    main(sys.argv[1])
