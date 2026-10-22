"""ConvNeXtV2 inference from the selected ordinary model; no TTA by default."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from takotsubo_ami_ai_ecg.cli import main

if __name__ == "__main__":
    raise SystemExit(main(fixed_model="convnextv2"))

