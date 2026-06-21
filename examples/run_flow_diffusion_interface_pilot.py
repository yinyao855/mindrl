"""Run flow/diffusion interface adapter pilot."""

from __future__ import annotations

from mindrl_repo.flow_diffusion_interface import (
    FlowDiffusionTrace,
    compare_flow_strategies,
)


def main() -> None:
    traces = {
        "low_drift": FlowDiffusionTrace(
            reward=1.0,
            surrogate_scores=(0.72, 0.74, 0.73, 0.75),
            anchor_distances=(0.10, 0.12, 0.11),
            smoothness_costs=(0.05, 0.04),
        ),
        "high_drift": FlowDiffusionTrace(
            reward=1.0,
            surrogate_scores=(0.55, 0.90, 0.35, 0.82),
            anchor_distances=(0.48, 0.60, 0.55),
            smoothness_costs=(0.20, 0.25),
            collision_or_feasibility_failures=2,
        ),
    }
    print("trace\tstrategy\tadapter\tbranch_weight\tclip\tanchor\trisk_adjusted_reward")
    for name, trace in traces.items():
        for strategy, result in compare_flow_strategies(trace).items():
            decision = result.decision
            print(
                f"{name}\t{strategy}\t{decision.adapter}\t"
                f"{decision.branch_weight:.3f}\t{decision.clip_range:.3f}\t"
                f"{decision.anchor_strength:.3f}\t{result.risk_adjusted_reward:.4f}"
            )


if __name__ == "__main__":
    main()
