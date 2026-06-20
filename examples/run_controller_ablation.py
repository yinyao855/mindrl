"""Run a toy MINDRL controller ablation."""

from __future__ import annotations

from mindrl_repo.controller_ablation import BranchScenario, run_controller_ablation
from mindrl_repo.interface_controller import BarrierProfile, PolicySpec


def main() -> None:
    scenarios = [
        BranchScenario(
            spec=PolicySpec(
                name="reasoning_tokens",
                modality="text",
                structure="ar",
                score_availability="exact",
                default_granularity=1,
            ),
            profile=BarrierProfile(),
            reward=1.0,
        ),
        BranchScenario(
            spec=PolicySpec(
                name="masked_block",
                modality="text",
                structure="parallel_discrete",
                score_availability="marginal",
                default_granularity=16,
            ),
            profile=BarrierProfile(nctc=0.45),
            reward=1.0,
        ),
        BranchScenario(
            spec=PolicySpec(
                name="action_chunk",
                modality="continuous_action",
                structure="flow",
                score_availability="surrogate",
                default_granularity=8,
            ),
            profile=BarrierProfile(surrogate_variance=0.35, drift=0.55, smoothness=0.10),
            reward=1.0,
        ),
    ]
    print("strategy\teffective_reward\trisk_cost\tupdate_budget\tadapters")
    for strategy in ["uniform", "score_routing", "barrier_gated"]:
        result = run_controller_ablation(scenarios, strategy)  # type: ignore[arg-type]
        adapters = ",".join(decision.adapter for decision in result.decisions)
        print(
            f"{result.strategy}\t{result.effective_reward:.4f}\t"
            f"{result.risk_cost:.4f}\t{result.update_budget:.4f}\t{adapters}"
        )


if __name__ == "__main__":
    main()
