"""
Vérifie que les conventions de frontière de tick_loader et de
data_loader.resample sont cohérentes :

    H1 construit directement depuis les ticks
        ==
    H1 construit en passant par le M1 puis data_loader.resample

Usage (venv actif, depuis la racine du projet, celle qui contient le
package regime_lib) :

    python chemin/vers/verif_conventions.py XAUUSD_optimized.parquet
    python chemin/vers/verif_conventions.py XAUUSD_optimized.parquet 2000000

Placer ce fichier dans le même dossier que tick_loader.py.
Le 2e argument (optionnel) limite le nombre de ticks lus
(défaut : 5 000 000) pour ne pas charger les ~94M de lignes en mémoire.
"""

from __future__ import annotations

import os
import sys
import tempfile

# La racine du projet (cwd) doit être importable pour `regime_lib`.
sys.path.insert(0, os.getcwd())

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from regime_lib.core.data_loader import resample
#from tick_loader import agreger_tick, charger_tick
from regime_lib.core.tick_loader import agreger_tick, charger_tick

COLS = ["open", "high", "low", "close", "volume"]


def lire_n_premiers_ticks(chemin: str, n: int) -> pd.DataFrame:
    """Lit seulement les n premières lignes du parquet (par lots)."""
    pf = pq.ParquetFile(chemin)
    lots, total = [], 0
    for lot in pf.iter_batches(batch_size=1_000_000):
        lots.append(lot)
        total += lot.num_rows
        if total >= n:
            break
    table = pa.Table.from_batches(lots).slice(0, n)
    return table.to_pandas()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    chemin = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5_000_000

    print(f"Lecture des {n:,} premiers ticks de {chemin} ...")
    brut = lire_n_premiers_ticks(chemin, n)
    print(f"Colonnes du parquet : {list(brut.columns)}"
          f" | index : {type(brut.index).__name__}")

    # charger_tick relit un parquet : on écrit l'extrait dans un temporaire.
    with tempfile.TemporaryDirectory() as tmp:
        extrait = os.path.join(tmp, "extrait.parquet")
        brut.to_parquet(extrait)
        try:
            ticks = charger_tick(extrait)
        except ValueError as exc:
            print(f"\ncharger_tick a refusé le fichier : {exc}")
            print("Vérifiez les noms de colonnes attendus "
                  "(bidPrice, askPrice, bidVolume, askVolume).")
            return 1

    print(f"{len(ticks):,} ticks chargés "
          f"({ticks.index.min()} -> {ticks.index.max()})")

    m1 = agreger_tick(ticks, "M1")
    h1_direct = agreger_tick(ticks, "H1")
    h1_via_m1 = resample(m1[COLS], "H1")

    # Combien de ticks sont exactement posés sur une frontière de minute ?
    sur_frontiere = ticks.index[ticks.index == ticks.index.floor("1min")]
    barres_touchees = sur_frontiere.unique().size / max(len(m1), 1)
    print(f"Ticks exactement sur une frontière de minute : "
          f"{len(sur_frontiere):,} ({len(sur_frontiere) / len(ticks):.3%} "
          f"des ticks, ~{barres_touchees:.1%} des barres M1 concernées)")

    try:
        pd.testing.assert_frame_equal(
            h1_direct[COLS], h1_via_m1[COLS], check_freq=False
        )
    except AssertionError as exc:
        print("\nRÉSULTAT : DIFFÉRENCE entre H1 direct et H1 via M1")
        print(str(exc)[:1500])
        return 1

    print(f"\nRÉSULTAT : OK - {len(h1_direct)} barres H1 identiques "
          f"(direct vs via M1 + resample).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
