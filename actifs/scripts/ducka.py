"""
Téléchargement de données tick Dukascopy.

Usage
-----
    # EURUSD, période par défaut
    python ducka.py --output actifs/EURUSD/EURUSD_Tick_20250623_20261001.parquet

    # XAUUSD, période explicite
    python ducka.py \
        --output actifs/XAUUSD/XAUUSD_Tick_20250623_20261001.parquet \
        --instrument XAU_USD \
        --start 2025-06-23 \
        --end 2026-10-01

Le script télécharge uniquement des données **tick** (pas de timeframe).
Il exporte un parquet compressé avec l'index timestamp matérialisé en
colonne.

Le nom de fichier attendu par convention est :
    <ACTIF>_Tick_<YYYYMMDD>_<YYYYMMDD>.parquet
mais le script n'impose pas ce format — il écrit à l'emplacement fourni.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import dukascopy_python
from dukascopy_python.instruments import (
    INSTRUMENT_FX_MAJORS_EUR_USD,
)


# ---------------------------------------------------------------------------
# Registre des instruments supportés
# ---------------------------------------------------------------------------
# Ajouter ici les instruments au fur et à mesure des besoins.
# La clé est le nom court utilisé dans --instrument.
INSTRUMENTS = {
    "EUR_USD": INSTRUMENT_FX_MAJORS_EUR_USD,
    # Pour XAUUSD, décommenter l'import ci-dessus et ajouter :
    # "XAU_USD": INSTRUMENT_FX_METALS_XAU_USD,
}


# ---------------------------------------------------------------------------
# Parsing des arguments
# ---------------------------------------------------------------------------
def _parse_date(s: str) -> datetime:
    """Parse une date au format YYYY-MM-DD."""
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Date invalide : {s!r}. Format attendu : YYYY-MM-DD."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ducka",
        description=(
            "Téléchargement de données tick Dukascopy et export Parquet."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help=(
            "Chemin du fichier Parquet de sortie. "
            "Exemple : actifs/EURUSD/EURUSD_Tick_20250623_20261001.parquet"
        ),
    )
    parser.add_argument(
        "--instrument", "-i",
        default="EUR_USD",
        choices=sorted(INSTRUMENTS.keys()),
        help=(
            f"Instrument à télécharger. Défaut : EUR_USD. "
            f"Disponibles : {sorted(INSTRUMENTS.keys())}."
        ),
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=datetime(2025, 6, 23),
        help="Date de début (YYYY-MM-DD). Défaut : 2025-06-23.",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=datetime(2026, 10, 1),
        help="Date de fin (YYYY-MM-DD). Défaut : 2026-10-01.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="Compression Parquet. Défaut : zstd.",
    )
    return parser


# ---------------------------------------------------------------------------
# Téléchargement et export
# ---------------------------------------------------------------------------
def telecharger_tick(
    instrument_key: str,
    start: datetime,
    end: datetime,
):
    """Télécharge les ticks pour l'instrument et la période donnés."""
    instrument = INSTRUMENTS[instrument_key]

    print(f"Téléchargement {instrument_key} (tick)...")
    print(f"  Période : {start.date()} → {end.date()}")

    df = dukascopy_python.fetch(
        instrument=instrument,
        interval=dukascopy_python.INTERVAL_TICK,
        offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=start,
        end=end,
        debug=False,
    )

    print(f"  Colonnes reçues : {df.columns.tolist()}")
    print(f"  Index           : {df.index.name} ({df.index.dtype})")
    print(f"  Ticks           : {len(df):,}")

    return df


def exporter_parquet(df, output: Path, compression: str) -> None:
    """Matérialise l'index et exporte en Parquet."""
    # L'index du DataFrame contient les timestamps UTC.
    # On le matérialise en colonne avant l'export.
    df = df.reset_index()
    if "index" in df.columns:
        df = df.rename(columns={"index": "timestamp"})

    print(f"  Colonnes après reset : {df.columns.tolist()}")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, compression=compression, index=False)

    taille_mo = output.stat().st_size / 1e6
    print(f"  Exporté : {output}")
    print(f"  Taille  : {taille_mo:.1f} Mo ({compression})")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    output = Path(args.output)

    # Vérification : ne pas écraser silencieusement
    if output.exists():
        print(f"⚠ Le fichier existe déjà : {output}")
        reponse = input("Écraser ? [y/N] ").strip().lower()
        if reponse not in ("y", "yes", "o", "oui"):
            print("Annulé.")
            return 1

    df = telecharger_tick(
        args.instrument,
        args.start,
        args.end,
    )
    exporter_parquet(df, output, args.compression)

    print("Terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
