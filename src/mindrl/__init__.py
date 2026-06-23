"""MindRL: transparent RL infrastructure for AR LLM and diffusion pilots."""

from pathlib import Path

from mindrl.mvp_benchmarks import (
    build_ar_grpo_smoke_report,
    build_diffusion_ddpo_smoke_report,
    write_reports,
)


def main() -> None:
    output_dir = Path("outputs/mvp_smoke")
    paths = write_reports(
        output_dir,
        (build_ar_grpo_smoke_report(), build_diffusion_ddpo_smoke_report()),
    )
    print(f"mindrl smoke reports written to {paths['markdown']}")
