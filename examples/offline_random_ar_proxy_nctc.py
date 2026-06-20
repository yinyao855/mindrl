"""Offline AR proxy nCTC smoke test with a randomly initialized tiny GPT-2.

This verifies the model-scoring code path without downloading any checkpoints.
Use `hf_ar_proxy_nctc.py` when network/model cache is available.
"""

from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from mindrl_repo.ar_proxy_nctc import pair_normalized_dependency_gap


def main() -> None:
    config = GPT2Config(
        vocab_size=32,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=1,
        bos_token_id=None,
        eos_token_id=None,
    )
    model = GPT2LMHeadModel(config)
    model.eval()

    prompt_ids = torch.tensor([1, 2, 3])
    completion_ids = torch.tensor([4, 5, 6, 7])
    with torch.no_grad():
        full = torch.cat([prompt_ids, completion_ids]).unsqueeze(0)
        log_probs = torch.log_softmax(model(full).logits[0], dim=-1)
        joint = [
            float(log_probs[len(prompt_ids) + offset - 1, int(token_id)])
            for offset, token_id in enumerate(completion_ids)
        ]
        prompt_log_probs = torch.log_softmax(model(prompt_ids.unsqueeze(0)).logits[0, -1], dim=-1)
        marginals = [float(prompt_log_probs[int(token_id)]) for token_id in completion_ids]

    print(f"joint_logprob={sum(joint):.4f}")
    print(f"prompt_only_marginal_sum={sum(marginals):.4f}")
    print(f"pair_normalized_dependency_gap={pair_normalized_dependency_gap(joint, marginals):.6f}")


if __name__ == "__main__":
    main()
