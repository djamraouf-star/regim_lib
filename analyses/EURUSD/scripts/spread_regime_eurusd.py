"""
Étude : le spread diffère-t-il selon le régime de marché ?

Source   : parquet tick EURUSD (Dukascopy)
Filtrage : exclusion de la barre H1 [17:00, 18:00) heure NY (rollover,
           indépendant de l'heure d'été) et des barres partielles
Méthodes : adx, atr_volatility, er_kaufman (causales uniquement)

Corrections par rapport à la v1 :
  1. Détection des régimes sur la série H1 COMPLÈTE, filtrage ensuite
     (pas de fenêtres d'indicateurs recollées par-dessus le rollover).
  2. Rollover filtré en heure NY (17h NY = 21h UTC été / 22h UTC hiver).
  3. Spread normalisé par la médiane de sa propre heure NY (spread_rel) :
     neutralise le profil de session, confondant majeur régime × spread.
  4. Test causal : régime[t] -> spread de la barre t+1 (spread_rel_next),
     en plus du test contemporain.
  5. Taille d'effet (epsilon²), stratification par session, information
     mutuelle régime/heure, stabilité mensuelle du signe de l'écart.

Limites : les barres H1 sont autocorrélées, donc les p-values du
Kruskal-Wallis sont trop optimistes. Se fier à epsilon², à la
stratification et à la stabilité mensuelle, pas au seul p.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from regime_lib.core.data_loader import load_parquet, resample
from regime_lib.core.registry import METHOD_REGISTRY

import regime_lib.methods  # noqa: F401  (effet de bord : registre)


URL = Path.home() / "Projects/regime_lib/data/EURUSD_Tick_20250623_20261001.parquet"
TIMEFRAME = "H1"
BAR = pd.Timedelta("1h")
METHODES = ["adx", "atr_volatility", "er_kaufman"]

NY = "America/New_York"
HEURE_ROLLOVER_NY = 17      # barre dont le DÉBUT est à 17h NY
PIP = 1e4
N_MIN_GROUPE = 30           # taille mini d'un groupe pour le Kruskal-Wallis
N_MIN_MOIS = 10             # taille mini d'un groupe par mois (stabilité)

COLS_BASE = ["open", "high", "low", "close", "volume",
             "spread_mean", "spread_max", "tick_count"]


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def session_ny(heures: np.ndarray) -> np.ndarray:
    """Session selon l'heure NY de début de barre."""
    h = np.asarray(heures)
    return np.where((h >= 3) & (h <= 7), "LONDRES",
                    np.where((h >= 8) & (h <= 16), "NY", "ASIE"))


def kw(df: pd.DataFrame, col: str, cible: str):
    """Kruskal-Wallis + epsilon² de `cible` selon `col`. None si < 2 groupes."""
    groupes = [
        g[cible].dropna().values
        for _, g in df.groupby(col)
        if g[cible].notna().sum() >= N_MIN_GROUPE
    ]
    k = len(groupes)
    if k < 2:
        return None
    h_stat, p = kruskal(*groupes)
    n = sum(len(g) for g in groupes)
    eps2 = (h_stat - k + 1) / (n - k)
    return h_stat, p, eps2, k, n


def taille_effet(eps2: float) -> str:
    if eps2 < 0.01:
        return "négligeable"
    if eps2 < 0.06:
        return "faible"
    if eps2 < 0.14:
        return "moyen"
    return "fort"


def info_mutuelle_norm(x: pd.Series, y: pd.Series) -> float:
    """I(X;Y) / H(X) : part de l'entropie de X expliquée par Y."""
    tab = pd.crosstab(x, y).values.astype(float)
    pxy = tab / tab.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    produit = px @ py
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / produit[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    return mi / hx if hx > 0 else float("nan")


def afficher_kw(titre: str, res) -> None:
    if res is None:
        print(f"    {titre:<26} : groupes insuffisants")
        return
    h_stat, p, eps2, k, n = res
    print(f"    {titre:<26} : H={h_stat:9.2f}  p={p:.2e}  "
          f"eps²={eps2:.4f} ({taille_effet(eps2)})  k={k} n={n:,}")


# ---------------------------------------------------------------------------
# 1. Chargement et agrégation
# ---------------------------------------------------------------------------
print("=" * 78)
print("  ÉTUDE SPREAD × RÉGIME — EURUSD (v2)")
print("=" * 78)
print()

print("[1/5] Chargement du parquet tick...")
df_m1, asset = load_parquet(str(URL), batch_size=1_000_000)
print(f"  Source       : {df_m1.attrs['source_type']}")
print(f"  Lignes src   : {df_m1.attrs['source_rows']:,}")
print(f"  Côté prix    : {df_m1.attrs['price_side']}")
print(f"  M1 barres    : {len(df_m1):,}")
print()

df_h1 = resample(df_m1, TIMEFRAME)
if df_h1.index.tz is None:
    df_h1.index = df_h1.index.tz_localize("UTC")
print(f"  {TIMEFRAME} barres : {len(df_h1):,}")
print(f"  Colonnes {TIMEFRAME} : {df_h1.columns.tolist()}")

# resample() peut ne pas propager les colonnes de spread / activité :
# on les reconstruit depuis le M1 (spread moyen pondéré par tick_count),
# avec la même convention que le reste du pipeline (label droite, closed gauche)
manquantes = [c for c in ("spread_mean", "spread_max", "tick_count", "volume")
              if c not in df_h1.columns]
if manquantes:
    m1 = pd.DataFrame({
        "w": df_m1["spread_mean"] * df_m1["tick_count"],
        "tick_count": df_m1["tick_count"],
        "spread_max": df_m1["spread_max"],
        "volume": df_m1["volume"],
    })
    agg = m1.resample("1h", label="right", closed="left").agg(
        {"w": "sum", "tick_count": "sum", "spread_max": "max", "volume": "sum"}
    )
    agg["spread_mean"] = agg["w"] / agg["tick_count"]
    df_h1 = df_h1.join(agg[manquantes])
    n_sans = int(df_h1[manquantes].isna().any(axis=1).sum())
    print(f"  Colonnes reconstruites depuis le M1 : {manquantes} "
          f"({n_sans} barres sans correspondance)")
print()


# ---------------------------------------------------------------------------
# 2. Détection des régimes sur la série COMPLÈTE (avant tout filtrage)
# ---------------------------------------------------------------------------
print("[2/5] Détection des régimes sur la série H1 complète...")
regimes: dict[str, pd.Series] = {}

for methode in METHODES:
    cls = METHOD_REGISTRY[methode]
    det = cls(timeframe=TIMEFRAME)
    out = det.fit_predict(df_h1)
    if len(out) != len(df_h1):
        raise ValueError(
            f"{methode} : {len(out)} sorties pour {len(df_h1)} barres "
            "(alignement ligne à ligne requis)"
        )
    regimes[methode] = (
        pd.Series(out["regime"].values, index=df_h1.index)
        .fillna("INCONNU")
        .astype(str)
    )
    print(f"  → {methode:20} ({len(out)} barres)")

for methode, serie in regimes.items():
    df_h1[f"regime_{methode}"] = serie
print()


# ---------------------------------------------------------------------------
# 3. Filtrage rollover (heure NY), partielles, et variables dérivées
# ---------------------------------------------------------------------------
print("[3/5] Filtrage et variables dérivées...")

# Barres étiquetées à droite : le début de la barre = étiquette - 1 barre
debut_ny = (df_h1.index - BAR).tz_convert(NY)
df_h1["h_ny"] = np.asarray(debut_ny.hour)
df_h1["session"] = session_ny(df_h1["h_ny"].values)

masque_rollover = df_h1["h_ny"] == HEURE_ROLLOVER_NY
if "is_partial" in df_h1.columns:
    masque_partiel = df_h1["is_partial"].fillna(False).astype(bool)
else:
    masque_partiel = pd.Series(False, index=df_h1.index)
masque_ok = ~masque_rollover & ~masque_partiel

n_roll = int(masque_rollover.sum())
n_part = int(masque_partiel.sum())
print(f"  Barres rollover exclues : {n_roll:,} "
      f"({100 * n_roll / len(df_h1):.2f} %)")
print(f"  Barres partielles       : {n_part:,}")
print(f"  Barres gardées          : {int(masque_ok.sum()):,}")
print()
print("  Contrôle du filtrage (médiane du spread) :")
print(f"    Avant : {df_h1['spread_mean'].median() * PIP:.3f} pips "
      f"(moyenne {df_h1['spread_mean'].mean() * PIP:.3f})")
print(f"    Après : {df_h1.loc[masque_ok, 'spread_mean'].median() * PIP:.3f} pips "
      f"(moyenne {df_h1.loc[masque_ok, 'spread_mean'].mean() * PIP:.3f})")
print()

# Spread relatif à la médiane de sa propre heure NY (neutralise la session)
med_heure = df_h1.loc[masque_ok].groupby("h_ny")["spread_mean"].median()
df_h1["spread_rel"] = df_h1["spread_mean"] / df_h1["h_ny"].map(med_heure)

# Spread relatif de la barre SUIVANTE : seulement si elle est strictement
# consécutive (pas de trou week-end) et elle-même exploitable
idx_s = df_h1.index.to_series()
suivante_consecutive = ((idx_s.shift(-1) - idx_s) == BAR).values
suivante_ok = masque_ok.shift(-1, fill_value=False).values
df_h1["spread_rel_next"] = (
    df_h1["spread_rel"].shift(-1).where(suivante_consecutive & suivante_ok)
)

df_analyse = df_h1.loc[masque_ok].copy()
print(f"  Barres analysées : {len(df_analyse):,}")
print()


# ---------------------------------------------------------------------------
# 4. Croisement spread × régime
# ---------------------------------------------------------------------------
print("[4/5] Croisement spread × régime")
print()

for methode in METHODES:
    col = f"regime_{methode}"
    d = df_analyse[df_analyse[col] != "INCONNU"].copy()

    print("─" * 78)
    print(f"  {methode.upper()}  ({len(d):,} barres hors période de chauffe)")
    print("─" * 78)

    # Distribution
    dist = d[col].value_counts(normalize=True).mul(100).round(2)
    print("  Distribution :")
    for regime, pct in dist.items():
        print(f"    {regime:24} : {pct:6.2f} %")
    print()

    # Statistiques par régime
    croise = d.groupby(col).agg(
        n=("spread_mean", "size"),
        spread_med_pips=("spread_mean", lambda s: s.median() * PIP),
        spread_moy_pips=("spread_mean", lambda s: s.mean() * PIP),
        spread_rel_med=("spread_rel", "median"),
        next_rel_med=("spread_rel_next", "median"),
        tick_count_moy=("tick_count", "mean"),
    ).round(3)
    print("  Statistiques par régime :")
    print(croise.to_string())
    print()

    # Part de l'entropie du régime expliquée par l'heure
    mi = info_mutuelle_norm(d[col], d["h_ny"])
    print(f"  Info mutuelle régime/heure NY : {100 * mi:.1f} % de l'entropie du régime")
    print()

    # Tests : contemporain et causal, global puis par session
    for cible, label in [("spread_rel", "CONTEMPORAIN (spread_rel[t])"),
                         ("spread_rel_next", "CAUSAL (spread_rel[t+1])")]:
        print(f"  Kruskal-Wallis — {label}")
        dd = d.dropna(subset=[cible])
        afficher_kw("Toutes sessions", kw(dd, col, cible))
        for sess in ["ASIE", "LONDRES", "NY"]:
            afficher_kw(f"Session {sess}", kw(dd[dd["session"] == sess], col, cible))
        print()

    # Amplitude (médianes) entre régimes extrêmes
    med_pips = d.groupby(col)["spread_mean"].median() * PIP
    med_next = d.groupby(col)["spread_rel_next"].median()
    if len(med_pips) >= 2:
        print(f"  Amplitude (médianes) : {med_pips.max() - med_pips.min():.3f} pips "
              f"sur spread_mean ; {med_next.max() - med_next.min():.3f} "
              f"sur spread_rel_next (1.0 = médiane de l'heure)")
        print()

    # Stabilité mensuelle du signe : régime « haut » vs « bas » (ordre global)
    if len(med_next.dropna()) >= 2:
        haut, bas = med_next.idxmax(), med_next.idxmin()
        d["mois"] = d.index.tz_localize(None).to_period("M")
        lignes = []
        for mois, g in d.groupby("mois"):
            a = g.loc[g[col] == haut, "spread_rel_next"].dropna()
            b = g.loc[g[col] == bas, "spread_rel_next"].dropna()
            if len(a) >= N_MIN_MOIS and len(b) >= N_MIN_MOIS:
                lignes.append((str(mois), len(a), len(b), a.median() - b.median()))
        print(f"  Stabilité mensuelle : médiane spread_rel_next[{haut}] − [{bas}]")
        if lignes:
            for mois, na, nb, ecart in lignes:
                print(f"    {mois}  n_haut={na:4d}  n_bas={nb:4d}  écart={ecart:+.3f}")
            n_pos = sum(1 for l in lignes if l[3] > 0)
            print(f"    → signe positif {n_pos}/{len(lignes)} mois "
                  f"(ordre haut/bas défini sur l'échantillon complet, descriptif)")
        else:
            print("    aucun mois avec des groupes suffisants")
    print()


# ---------------------------------------------------------------------------
# 5. Sortie
# ---------------------------------------------------------------------------
print("[5/5] Résumé et écriture")
print(f"Barres analysées        : {len(df_analyse):,}")
print(f"Rollover exclu          : barre débutant à {HEURE_ROLLOVER_NY}h NY")
print()

output_dir = Path.home() / "Projects/regime_lib/analyses/output"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "spread_regime_eurusd.parquet"

cols_a_garder = (
    COLS_BASE
    + ["h_ny", "session", "spread_rel", "spread_rel_next"]
    + [f"regime_{m}" for m in METHODES]
)
df_analyse[cols_a_garder].to_parquet(output_path)
print(f"Résultat écrit : {output_path}")
