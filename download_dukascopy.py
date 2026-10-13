"""Télécharge les ticks Dukascopy en Parquet, sur [début, fin) en UTC."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import re
import tempfile

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "dukascopy"
# Délai de publication des données par Dukascopy : en deçà, un lot peut être incomplet.
PUBLICATION_DELAY = timedelta(hours=6)


def apply_umask(path) -> None:
    """mkstemp crée en 0600 : rétablit les permissions d'un fichier ordinaire."""
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(path, 0o666 & ~umask)


def cached_ticks(instrument, start, end, cache_dir, max_retries, strict=False):
    """Réutilise un lot exact validé ; publie les nouveaux lots atomiquement."""
    import dukascopy_python
    import pyarrow as pa
    import pyarrow.parquet as pq

    identity = json.dumps({
        "format": 1, "library": version("dukascopy-python"),
        "instrument": instrument, "interval": "TICK", "side": "B",
        "start": start.isoformat(), "end": end.isoformat(),
    }, sort_keys=True).encode()
    key = hashlib.sha256(identity).hexdigest()
    path = cache_dir / f"{start:%Y%m%dT%H%M%S}-{key}.parquet"
    # Une période encore en cours ne doit jamais devenir un cache définitif.
    cacheable = end <= datetime.now(timezone.utc) - PUBLICATION_DELAY
    if cacheable and path.exists():
        try:
            table = pq.read_table(path)
            if (table.schema.metadata or {}).get(b"dukascopy_cache") != identity:
                raise ValueError("Identité du lot incorrecte")
            original = table.to_pandas()
            frame = normalize_ticks(original, start, end)
            if frame.empty or len(frame) != len(original):
                raise ValueError("Lot vide ou horodatages hors période")
            logging.info("Cache réutilisé : %s → %s", start.isoformat(), end.isoformat())
            return frame
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logging.warning("Cache invalide, nouveau téléchargement : %s (%s)", path, exc)

    frame = normalize_ticks(dukascopy_python.fetch(
        instrument=instrument, interval=dukascopy_python.INTERVAL_TICK,
        offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=start, end=end, max_retries=max_retries,
    ), start, end, drop_crossed=not strict)
    # Une réponse vide peut être transitoire : ne pas la mémoriser.
    if cacheable and not frame.empty:
        cache_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=True)
        table = table.replace_schema_metadata({
            **(table.schema.metadata or {}), b"dukascopy_cache": identity,
        })
        fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".tmp", dir=cache_dir)
        os.close(fd)
        try:
            pq.write_table(table, temporary, compression="zstd")
            apply_umask(temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return frame


def utc_date(value: str) -> datetime:
    """Accepte une date ISO ou un horodatage ; sans fuseau signifie UTC."""
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Date ISO attendue : YYYY-MM-DD") from exc
    return (result.replace(tzinfo=timezone.utc) if result.tzinfo is None
            else result.astimezone(timezone.utc))


def resolve_instrument(value: str, instruments) -> str:
    """Résout EURUSD, EUR/USD ou le nom d'une constante de la bibliothèque."""
    known = {key: val for key, val in vars(instruments).items()
             if key.startswith("INSTRUMENT_") and isinstance(val, str)}
    if value in known:
        return known[value]
    compact = lambda text: re.sub(r"[^A-Z0-9]", "", text.upper())
    matches = {val for val in known.values() if compact(val) == compact(value)}
    if len(matches) != 1:
        raise ValueError(f"Instrument inconnu ou ambigu : {value}")
    return matches.pop()


def normalize_ticks(frame, start: datetime, end: datetime, drop_crossed: bool = False):
    """Valide les ticks ; avec drop_crossed, écarte les quotes croisés (ask < bid)."""
    import numpy as np
    import pandas as pd

    columns = ["bidPrice", "askPrice", "bidVolume", "askVolume"]
    if frame.empty:  # week-end ou jour férié : l'index renvoyé peut être non temporel
        return pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in columns},
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Index temporel Dukascopy invalide")
    frame = frame.loc[:, columns].copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.index.name = "timestamp"
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    frame = frame.astype("float64").sort_index(kind="stable")
    if drop_crossed:
        crossed = frame.askPrice < frame.bidPrice
        if crossed.any():
            logging.warning("%s quotes croisés écartés (ask < bid)", int(crossed.sum()))
            frame = frame.loc[~crossed]
    values = frame.to_numpy()
    if (not np.isfinite(values).all() or (values[:, :2] <= 0).any()
            or (values[:, 2:] < 0).any()
            or (frame.askPrice < frame.bidPrice).any()):
        raise ValueError("Prix ou volumes Dukascopy incohérents")
    # Plusieurs ticks peuvent partager la même milliseconde : les conserver.
    return frame


def publish(temporary, output: Path) -> None:
    """Lien physique sans écrasement ; repli si le système de fichiers l'interdit."""
    try:
        os.link(temporary, output)
    except FileExistsError:
        raise
    except OSError:
        # Pas de lien physique (FAT, montage réseau…) : création exclusive puis remplacement.
        with open(output, "xb"):
            pass
        os.replace(temporary, output)


def download(instrument: str, start: datetime, end: datetime, output: Path,
             max_retries: int = 3, cache_dir: Path | None = None,
             strict: bool = False) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Dates avec fuseau requises et début strictement avant fin")
    if max_retries < 0:
        raise ValueError("max_retries doit être positif ou nul")
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    if output.exists():
        raise FileExistsError(f"Le fichier existe déjà : {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.stem}-", suffix=".parquet",
                                     dir=output.parent)
    os.close(fd)
    total = 0
    writer = None
    try:
        cursor = start
        while cursor < end:
            midnight = cursor.replace(hour=0, minute=0, second=0, microsecond=0)
            stop = min(midnight + timedelta(days=1), end)
            frame = cached_ticks(instrument, cursor, stop, cache_dir, max_retries, strict)
            logging.info("%s → %s : %s ticks", cursor.isoformat(), stop.isoformat(), len(frame))
            if not frame.empty:
                table = pa.Table.from_pandas(frame, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                total += len(frame)
            cursor = stop
        if total == 0:
            raise ValueError("Aucun tick reçu sur cette période ; aucun fichier créé")
        writer.close()
        writer = None
        apply_umask(temporary)
        # Publication atomique sans écraser un fichier créé pendant le téléchargement.
        publish(temporary, output)
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            Path(temporary).unlink(missing_ok=True)
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="EURUSD", help="EURUSD, XAUUSD, EUR/USD…")
    parser.add_argument("--start", required=True, type=utc_date, help="Début inclus (UTC)")
    parser.add_argument("--end", required=True, type=utc_date, help="Fin exclue (UTC)")
    parser.add_argument("--output", type=Path, help="Chemin du fichier Parquet")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--strict", action="store_true",
                        help="Échouer sur un quote croisé (ask < bid) au lieu de l'écarter")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Dossier du cache persistant (défaut : .cache/dukascopy)")
    args = parser.parse_args(argv)
    if args.start >= args.end:
        parser.error("--start doit précéder --end")
    if args.max_retries < 0:
        parser.error("--max-retries doit être positif ou nul")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        from dukascopy_python import instruments

        instrument = resolve_instrument(args.instrument, instruments)
        symbol = re.sub(r"[^A-Z0-9]", "", instrument.upper())
        def stamp(date):
            return date.strftime("%Y%m%d" if date.time() == datetime.min.time()
                                 else "%Y%m%dT%H%M%S%fZ")
        output = args.output or (Path(__file__).resolve().parent / "actifs" / symbol
                                 / f"{symbol}_Tick_{stamp(args.start)}_{stamp(args.end)}.parquet")
        count = download(instrument, args.start, args.end, output, args.max_retries,
                         cache_dir=args.cache_dir, strict=args.strict)
    except ImportError as exc:
        parser.exit(1, f"Dépendance manquante ({exc}). Installer : pip install -e '.[download]'\n")
    except Exception as exc:
        parser.exit(1, f"Échec du téléchargement : {exc}\n")
    logging.info("%s ticks enregistrés dans %s", count, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
