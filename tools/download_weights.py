#!/usr/bin/env python3
"""Download the official, hash-verified tensor-only inference weight bundle."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from takotsubo_ami_ai_ecg.download import main


if __name__ == "__main__":
    sys.exit(main())
