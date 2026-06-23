"""Run a dependency-only MINDRL discrete-interface smoke experiment.

This example does not require model checkpoints. It verifies the theory-side
signal used by the full language reproduction: stronger within-block dependence
raises total correlation, and refining a parallel block lowers the measurable
factorization barrier.
"""

from mindrl.synthetic_barriers import (
    binary_pair_distribution,
    total_correlation,
    within_block_barrier,
)


def main() -> None:
    print("Binary pair dependence sweep")
    for diagonal_prob in (0.25, 0.35, 0.45, 0.49):
        ctc = total_correlation(binary_pair_distribution(diagonal_prob))
        print(f"diag_cell_prob={diagonal_prob:.2f} ctc={ctc:.4f}")

    joint = {
        (0, 0, 0): 0.46,
        (1, 1, 1): 0.46,
        (0, 0, 1): 0.08 / 6,
        (0, 1, 0): 0.08 / 6,
        (0, 1, 1): 0.08 / 6,
        (1, 0, 0): 0.08 / 6,
        (1, 0, 1): 0.08 / 6,
        (1, 1, 0): 0.08 / 6,
    }
    coarse = within_block_barrier(joint, blocks=[(0, 1, 2)])
    refined = within_block_barrier(joint, blocks=[(0,), (1, 2)])
    singleton = within_block_barrier(joint, blocks=[(0,), (1,), (2,)])
    print("\nBlock refinement")
    print(f"coarse=(0,1,2) barrier={coarse:.4f}")
    print(f"refined=(0),(1,2) barrier={refined:.4f}")
    print(f"singleton=(0),(1),(2) barrier={singleton:.4f}")


if __name__ == "__main__":
    main()
