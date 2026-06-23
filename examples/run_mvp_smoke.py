"""Run the dependency-light AR + diffusion MVP smoke benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from mindrl.mvp_benchmarks import (
    build_ar_grpo_smoke_report,
    build_diffusion_ddpo_smoke_report,
    write_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="outputs/mvp_smoke",
        help="Directory for JSONL and Markdown reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = (build_ar_grpo_smoke_report(), build_diffusion_ddpo_smoke_report())
    paths = write_reports(Path(args.output_dir), reports)
    print(f"wrote {paths['jsonl']}")
    print(f"wrote {paths['markdown']}")


if __name__ == "__main__":
    main()
