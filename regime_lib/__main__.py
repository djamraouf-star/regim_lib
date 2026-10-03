"""
Point d'entrée pour `python -m regime_lib`.

Délègue simplement à `regime_lib.cli.main`.
"""

from regime_lib.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
