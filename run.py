from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecowaste_pccm.pipeline import run_main_experiment


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ECO-Waste-PCCM main workbook-completion experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.json",
        help="Path to a JSON configuration file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    result = run_main_experiment(ROOT, config)
    print("ECO-Waste-PCCM main experiment completed.")
    print(f"Output directory: {result.output_dir}")
    print(f"Normalized MAE: {result.metrics['normalized_mae']:.4f}")
    print(f"Normalized RMSE: {result.metrics['normalized_rmse']:.4f}")
    print(f"Closure violation: {result.metrics['closure_violation']:.6f}")


if __name__ == "__main__":
    main()
