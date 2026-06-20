"""Run a toy flow surrogate anchor/clip tradeoff pilot."""

from __future__ import annotations

from mindrl_repo.flow_surrogate import FlowSurrogateSetting, evaluate_flow_surrogate


def main() -> None:
    settings = [
        ("loose", FlowSurrogateSetting(1.0, 0.35, 0.60, anchor_strength=0.0, clip_range=0.2)),
        ("anchored", FlowSurrogateSetting(1.0, 0.35, 0.60, anchor_strength=2.0, clip_range=0.2)),
        ("anchored_clipped", FlowSurrogateSetting(1.0, 0.35, 0.60, anchor_strength=2.0, clip_range=0.1)),
    ]
    print("setting\teffective_update\tresidual_drift\trisk_adjusted_objective")
    for name, setting in settings:
        result = evaluate_flow_surrogate(setting)
        print(
            f"{name}\t{result.effective_update:.4f}\t"
            f"{result.residual_drift:.4f}\t{result.risk_adjusted_objective:.4f}"
        )


if __name__ == "__main__":
    main()
