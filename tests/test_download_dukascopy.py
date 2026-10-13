"""Contrats du téléchargement, sans accès réseau."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from download_dukascopy import download, normalize_ticks, resolve_instrument, utc_date


def ticks(dates):
    return pd.DataFrame(
        {"bidPrice": 1.1, "askPrice": 1.2, "bidVolume": 2., "askVolume": 3.},
        index=pd.to_datetime(dates, utc=True),
    )


class DownloadTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache_dir = Path(directory.name)
        cache_patch = patch("download_dukascopy.DEFAULT_CACHE_DIR", self.cache_dir)
        cache_patch.start()
        self.addCleanup(cache_patch.stop)

    def test_utc_and_instruments(self):
        from dukascopy_python import instruments
        self.assertEqual(utc_date("2024-01-01T01:00:00+01:00"), utc_date("2024-01-01"))
        self.assertEqual(resolve_instrument("EURUSD", instruments), "EUR/USD")
        self.assertEqual(resolve_instrument("XAUUSD", instruments), "XAU/USD")
        with self.assertRaises(ValueError):
            resolve_instrument("unknown", instruments)

    def test_boundaries_duplicates_and_parquet(self):
        start, end = utc_date("2024-01-01"), utc_date("2024-01-03")
        responses = [ticks(["2024-01-01", "2024-01-01", "2024-01-02"]),
                     ticks(["2024-01-02", "2024-01-03"])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ticks.parquet"
            with patch("dukascopy_python.fetch", side_effect=responses) as fetch:
                self.assertEqual(download("EUR/USD", start, end, path), 3)
                self.assertEqual(fetch.call_count, 2)
                self.assertEqual(fetch.call_args.kwargs["interval"], "TICK")
            frame = pd.read_parquet(path)
            self.assertEqual(len(frame), 3)
            self.assertEqual(frame.index.name, "timestamp")
            self.assertEqual(str(frame.index.tz), "UTC")
            self.assertEqual(frame.index.duplicated().sum(), 1)
            with patch("dukascopy_python.fetch") as fetch:
                with self.assertRaises(FileExistsError):
                    download("EUR/USD", start, end, path)
                fetch.assert_not_called()

    def test_error_does_not_publish_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ticks.parquet"
            with patch("dukascopy_python.fetch", side_effect=[ticks(["2024-01-01"]), RuntimeError("réseau")]):
                with self.assertRaises(RuntimeError):
                    download("EUR/USD", utc_date("2024-01-01"), utc_date("2024-01-03"), path)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(len(list(self.cache_dir.glob("*.parquet"))), 1)
            with patch("dukascopy_python.fetch", return_value=ticks(["2024-01-02"])) as fetch:
                self.assertEqual(download("EUR/USD", utc_date("2024-01-01"), utc_date("2024-01-03"), path), 2)
                fetch.assert_called_once()
                self.assertEqual(fetch.call_args.kwargs["start"], utc_date("2024-01-02"))

    def test_cache_reuse_and_corruption(self):
        start, end = utc_date("2024-01-01"), utc_date("2024-01-02")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dukascopy_python.fetch", return_value=ticks(["2024-01-01", "2024-01-01"])):
                download("EUR/USD", start, end, root / "first.parquet")
            with patch("dukascopy_python.fetch") as fetch:
                download("EUR/USD", start, end, root / "second.parquet")
                fetch.assert_not_called()
            pd.testing.assert_frame_equal(pd.read_parquet(root / "first.parquet"),
                                          pd.read_parquet(root / "second.parquet"))
            cached = next(self.cache_dir.glob("*.parquet"))
            cached.write_bytes(b"interrupted file")
            with patch("dukascopy_python.fetch", return_value=ticks(["2024-01-01"])) as fetch:
                download("EUR/USD", start, end, root / "repaired.parquet")
                fetch.assert_called_once()
            self.assertEqual(len(pd.read_parquet(cached)), 1)

    def test_cache_isolates_instruments_and_partial_days(self):
        start, end = utc_date("2024-01-01"), utc_date("2024-01-02")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dukascopy_python.fetch", return_value=ticks(["2024-01-01T12:00:00Z"])) as fetch:
                download("EUR/USD", start, end, root / "eur.parquet")
                download("XAU/USD", start, end, root / "xau.parquet")
                download("EUR/USD", utc_date("2024-01-01T12:00:00Z"), end, root / "partial.parquet")
                self.assertEqual(fetch.call_count, 3)
            self.assertEqual(len(list(self.cache_dir.glob("*.parquet"))), 3)

    def test_interruption_keeps_completed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ticks.parquet"
            with patch("dukascopy_python.fetch", side_effect=[ticks(["2024-01-01"]), KeyboardInterrupt]):
                with self.assertRaises(KeyboardInterrupt):
                    download("EUR/USD", utc_date("2024-01-01"), utc_date("2024-01-03"), path)
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(len(list(self.cache_dir.glob("*.parquet"))), 1)

    def test_empty_and_invalid_data(self):
        start, end = utc_date("2024-01-01"), utc_date("2024-01-02")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ticks.parquet"
            with patch("dukascopy_python.fetch", return_value=ticks([])):
                with self.assertRaisesRegex(ValueError, "Aucun tick"):
                    download("EUR/USD", start, end, path)
            self.assertFalse(path.exists())
            self.assertEqual(list(self.cache_dir.iterdir()), [])
        frame = ticks(["2024-01-01"])
        frame["askPrice"] = 1.0
        with self.assertRaises(ValueError):
            normalize_ticks(frame, start, end)

    def test_empty_day_with_non_temporal_index(self):
        start, end = utc_date("2024-01-06"), utc_date("2024-01-07")
        result = normalize_ticks(pd.DataFrame(), start, end)
        self.assertTrue(result.empty)
        self.assertEqual(result.index.name, "timestamp")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ticks.parquet"
            responses = [pd.DataFrame(), ticks(["2024-01-07"])]
            with patch("dukascopy_python.fetch", side_effect=responses):
                self.assertEqual(download("EUR/USD", start, utc_date("2024-01-08"), path), 1)

    def test_crossed_quotes_dropped_unless_strict(self):
        start, end = utc_date("2024-01-01"), utc_date("2024-01-02")
        frame = ticks(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"])
        frame.iloc[1, 1] = 1.0
        self.assertEqual(len(normalize_ticks(frame, start, end, drop_crossed=True)), 1)
        with tempfile.TemporaryDirectory() as directory:
            with patch("dukascopy_python.fetch", return_value=frame):
                self.assertEqual(download("EUR/USD", start, end, Path(directory) / "a.parquet"), 1)
                with self.assertRaises(ValueError):
                    download("EUR/USD", start, end, Path(directory) / "b.parquet",
                             cache_dir=self.cache_dir / "strict", strict=True)

    def test_recent_period_not_cached(self):
        from datetime import datetime, timedelta, timezone
        end = datetime.now(timezone.utc) - timedelta(hours=1)
        start = end - timedelta(hours=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch("dukascopy_python.fetch", return_value=ticks([start.isoformat()])):
                download("EUR/USD", start, end, Path(directory) / "t.parquet")
        self.assertEqual(list(self.cache_dir.iterdir()), [])

    def test_link_fallback_and_permissions(self):
        start, end = utc_date("2024-01-01"), utc_date("2024-01-02")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ticks.parquet"
            with patch("dukascopy_python.fetch", return_value=ticks(["2024-01-01"])), \
                    patch("download_dukascopy.os.link", side_effect=OSError("no link")):
                download("EUR/USD", start, end, path)
            self.assertEqual(len(pd.read_parquet(path)), 1)
            self.assertNotEqual(path.stat().st_mode & 0o044, 0)


if __name__ == "__main__":
    unittest.main()
