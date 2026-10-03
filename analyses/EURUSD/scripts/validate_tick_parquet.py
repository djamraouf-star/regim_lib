"""
Validation du parquet tick EURUSD.

Produit un rapport de qualité : distributions, continuité, cohérence.
"""

import pandas as pd
from regime_lib.core.data_loader import load_parquet

URL = "~/Projects/regime_lib/data/EURUSD_Tick_20250623_20261001.parquet"
PIP = 1e4  # EURUSD : 1 pip = 0.0001
WEEKEND_GAP = pd.Timedelta("40h")

print("Chargement (streaming)...")
df_m1, _ = load_parquet(URL, batch_size=1_000_000)

print(f"Source       : {df_m1.attrs['source_type']}")
print(f"Lignes src   : {df_m1.attrs['source_rows']:,}")
print(f"Côté prix    : {df_m1.attrs['price_side']}")
print(f"M1 barres    : {len(df_m1):,}")
print()

# ---------------------------------------------------------------------------
# Intégrité de l'index
# ---------------------------------------------------------------------------
print("=== Intégrité de l'index ===")
print(f"  Doublons      : {df_m1.index.duplicated().sum():,}")
print(f"  Monotone      : {df_m1.index.is_monotonic_increasing}")
print(f"  Timezone      : {df_m1.index.tz}")
print()

# ---------------------------------------------------------------------------
# Distribution des barres
# ---------------------------------------------------------------------------
print("=== Distribution des barres M1 ===")
print(f"Première : {df_m1.index.min()}")
print(f"Dernière : {df_m1.index.max()}")
print(f"Durée    : {df_m1.index.max() - df_m1.index.min()}")
print()

print("=== Statistiques des colonnes ===")
cols = ["open", "high", "low", "close", "volume",
        "spread_mean", "spread_max", "tick_count"]
print(df_m1[cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(6))
print()

# ---------------------------------------------------------------------------
# Cohérence OHLC / spread
# ---------------------------------------------------------------------------
print("=== Cohérence OHLC ===")
o, h, l, c = (df_m1[k] for k in ("open", "high", "low", "close"))
oc_max = pd.concat([o, c], axis=1).max(axis=1)
oc_min = pd.concat([o, c], axis=1).min(axis=1)
checks = {
    "high < max(open, close)":  (h < oc_max).sum(),
    "low  > min(open, close)":  (l > oc_min).sum(),
    "high < low":               (h < l).sum(),
    "prix <= 0":                (df_m1[["open", "high", "low", "close"]] <= 0).any(axis=1).sum(),
    "spread_mean < 0":          (df_m1["spread_mean"] < 0).sum(),
    "spread_max < spread_mean": (df_m1["spread_max"] < df_m1["spread_mean"]).sum(),
    "tick_count <= 0":          (df_m1["tick_count"] <= 0).sum(),
    "NaN (toutes colonnes)":    df_m1.isna().any(axis=1).sum(),
}
for k, v in checks.items():
    print(f"  {k:<26}: {int(v):,}")
print()

# ---------------------------------------------------------------------------
# Continuité : trous entre barres consécutives
# ---------------------------------------------------------------------------
print("=== Trous (écart > 1 min) ===")
gaps = df_m1.index.to_series().diff().dropna()
gaps = gaps[gaps > pd.Timedelta("1min")]
weekend = gaps[gaps >= WEEKEND_GAP]
anomal = gaps[gaps < WEEKEND_GAP]
print(f"  Total trous           : {len(gaps):,}")
print(f"  Type week-end (>=40h) : {len(weekend):,}")
print(f"  Intra-semaine (<40h)  : {len(anomal):,}")
print("  Top 15 intra-semaine (début du trou = timestamp précédent) :")
top = anomal.sort_values(ascending=False).head(15)
for ts, dur in top.items():
    print(f"    {ts - dur}  ->  {ts}   ({dur})")
print()

# ---------------------------------------------------------------------------
# Répartition horaire (UTC) avec ratio de couverture
# ---------------------------------------------------------------------------
print("=== Répartition horaire (UTC) ===")
idx = df_m1.index
jours_par_heure = (
    pd.Series(1, index=idx)
    .groupby([idx.normalize(), idx.hour])
    .size()
    .groupby(level=1)
    .size()
)
par_heure = df_m1.groupby(idx.hour).agg(
    barres=("close", "size"),
    tick_count_moy=("tick_count", "mean"),
    spread_pips=("spread_mean", lambda s: s.mean() * PIP),
)
par_heure["jours_actifs"] = jours_par_heure
par_heure["couverture_%"] = (
    100 * par_heure["barres"] / (par_heure["jours_actifs"] * 60)
)
print(par_heure.round(4).to_string())
print()

# ---------------------------------------------------------------------------
# Répartition par jour de semaine
# ---------------------------------------------------------------------------
print("=== Répartition par jour de semaine ===")
jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
par_jour = df_m1.groupby(idx.dayofweek).agg(
    barres=("close", "size"),
    spread_pips=("spread_mean", lambda s: s.mean() * PIP),
)
par_jour.index = [jours[i] for i in par_jour.index]
print(par_jour.round(4).to_string())
print()

# ---------------------------------------------------------------------------
# Dérive mensuelle (changement de source / broker / régime)
# ---------------------------------------------------------------------------
print("=== Spread et activité par mois ===")
tz_safe = idx.tz_localize(None) if idx.tz is not None else idx
mensuel = df_m1.groupby(tz_safe.to_period("M")).agg(
    barres=("close", "size"),
    tick_moy=("tick_count", "mean"),
    spread_pips=("spread_mean", lambda s: s.mean() * PIP),
    spread_max_pips=("spread_max", lambda s: s.max() * PIP),
).round(3)
print(mensuel.to_string())
print()

# ---------------------------------------------------------------------------
# Spread autour du rollover (21h-22h UTC)
# ---------------------------------------------------------------------------
print("=== Spread autour du rollover (pips, moyenne par heure) ===")
roll = df_m1[df_m1.index.hour.isin([20, 21, 22, 23])]
print(
    roll.groupby(roll.index.hour)["spread_mean"]
    .agg(moyenne=lambda s: s.mean() * PIP, p99=lambda s: s.quantile(0.99) * PIP)
    .round(3)
    .to_string()
)
print()

# ---------------------------------------------------------------------------
# Barres partielles
# ---------------------------------------------------------------------------
print("=== Barres partielles (détail) ===")
print(df_m1[df_m1["is_partial"]].index.tolist())
print()

print("=== Trous intra-semaine hors fêtes, par durée ===")
g = gaps[gaps < WEEKEND_GAP].copy()
debut = g.index - g.values
fetes = (g.index >= "2025-12-24") & (g.index <= "2026-01-02")
g = g[~fetes]
bins = pd.cut(g.dt.total_seconds() / 60, [1, 2, 3, 5, 10, 30, 120, 10_000])
print(g.groupby(bins, observed=True).size().to_string())
print("Par heure UTC (fin du trou) :")
print(g.groupby(g.index.hour).size().to_string())
print()

print("=== Durées des trous week-end (h) ===")
print((weekend.dt.total_seconds() / 3600).round(1).value_counts().sort_index().to_string())
