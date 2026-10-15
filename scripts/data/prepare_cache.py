"""Prepare cache command."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pure.cli.prepare_cache import main

if __name__ == "__main__":
    main()
