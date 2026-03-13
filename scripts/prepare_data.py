from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gcdta.data.preprocess import prepare_all, prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and preprocess DTI benchmarks into ./data")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "davis", "kiba", "core2016", "test71", "test105", "pdbbind_v2016"],
        help="Dataset to prepare.",
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "all":
        prepare_all(
            data_root=args.data_root,
            force_download=args.force_download,
            force_preprocess=args.force_preprocess,
            seed=args.seed,
        )
        print("Prepared datasets: davis, kiba, core2016, test71, test105, pdbbind_v2016")
    else:
        out_path = prepare_dataset(
            dataset=args.dataset,
            data_root=args.data_root,
            force_download=args.force_download,
            force_preprocess=args.force_preprocess,
            seed=args.seed,
        )
        print(f"Prepared dataset '{args.dataset}': {out_path}")


if __name__ == "__main__":
    main()
