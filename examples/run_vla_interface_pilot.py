"""Run AR+flow/VLA interface controller pilot."""

from __future__ import annotations

from mindrl.flow_diffusion_interface import FlowDiffusionTrace
from mindrl.vla_interface import (
    ARBranchTrace,
    VLABranchTrace,
    compare_vla_strategies,
)


def main() -> None:
    trace = VLABranchTrace(
        task_success_proxy=0.65,
        ar=ARBranchTrace(token_reward=0.30, exact_logprob_available=True),
        flow=FlowDiffusionTrace(
            reward=0.80,
            surrogate_scores=(0.40, 0.85, 0.55, 0.90),
            anchor_distances=(0.50, 0.62, 0.58),
            smoothness_costs=(0.22, 0.28),
            collision_or_feasibility_failures=2,
        ),
        semantic_staleness=0.15,
        world_uncertainty=0.20,
    )
    print(
        "strategy\tar_adapter\tflow_adapter\tsemantic_refresh\t"
        "action_granularity\trisk_adjusted_score"
    )
    for strategy, result in compare_vla_strategies(trace).items():
        print(
            f"{strategy}\t{result.ar_decision.adapter}\t"
            f"{result.flow_decision.adapter}\t{result.semantic_refresh_budget:.3f}\t"
            f"{result.action_chunk_granularity}\t{result.risk_adjusted_score:.4f}"
        )


if __name__ == "__main__":
    main()
