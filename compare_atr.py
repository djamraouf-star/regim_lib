"""
Test de non-auto-référence du seuil.

Perturber high[i] change atr[i] (légitime). La question est :
le SEUIL à la barre i change-t-il aussi ?
  - original : oui (atr[i] est dans la fenêtre du seuil)
  - nouveau  : non (seuil utilise atr.shift(1), fenêtre = atr[i-500 : i])
"""
import numpy as np
import pandas as pd

from regime_lib.core.data_loader import load_parquet, resample


def compute_atr(df, n_atr=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n_atr, min_periods=1).mean()


def seuil_original(atr, fenetre=500, q=0.33):
    # atr[i] est DANS la fenêtre → seuil[i] contaminé par atr[i]
    return atr.rolling(fenetre, min_periods=fenetre // 2).quantile(q)


def seuil_nouveau(atr, fenetre=500, q=0.33):
    # atr[i] est EXCLU (shift) → seuil[i] non contaminé
    return atr.shift(1).rolling(fenetre, min_periods=fenetre // 2).quantile(q)


URL = "/home/djamal/Projects/EURUSD_startegie/EURUSD_M1_202507_202608.parquet"
df_raw, _ = load_parquet(URL)
df_tf = resample(df_raw, "H1")

atr_base = compute_atr(df_tf)
seuil_orig_base = seuil_original(atr_base)
seuil_new_base = seuil_nouveau(atr_base)

idx_test = list(range(600, 650))  # bien après la chauffe

n_orig = 0
n_new = 0
for i in idx_test:
    df_mod = df_tf.copy()
    df_mod.iloc[i, df_mod.columns.get_loc("high")] *= 10  # TR[i] explose
    atr_mod = compute_atr(df_mod)

    s_orig = seuil_original(atr_mod).iloc[i]
    s_new = seuil_nouveau(atr_mod).iloc[i]

    if not np.isclose(s_orig, seuil_orig_base.iloc[i], equal_nan=True):
        n_orig += 1
    if not np.isclose(s_new, seuil_new_base.iloc[i], equal_nan=True):
        n_new += 1

print(f"Original : {n_orig}/{len(idx_test)} seuils[i] changent "
      f"quand on perturbe TR[i]")
print(f"Nouveau  : {n_new}/{len(idx_test)} seuils[i] changent "
      f"quand on perturbe TR[i]")
