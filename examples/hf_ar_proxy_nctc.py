"""Compute an AR-scorer dependency-gap smoke test with a Hugging Face model.

This is not the full MINDRL paired AR/dLLM protocol. It is a low-cost proxy for
checking whether a causal LM can provide a dependence signal before investing in
paired dLLM checkpoints.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindrl.ar_proxy_nctc import pair_normalized_dependency_gap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--prompt", default="Question: What is 2 plus 2?\nAnswer:")
    parser.add_argument("--completion", default=" The answer is 4.")
    parser.add_argument("--max-block-tokens", type=int, default=16)
    args = parser.parse_args()

    device = _best_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    prompt_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    completion_ids = tokenizer(
        args.completion,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids[0][: args.max_block_tokens]

    if len(prompt_ids) == 0 or len(completion_ids) == 0:
        raise ValueError("prompt and completion must tokenize to at least one token")

    joint, marginals = score_completion_block(model, prompt_ids, completion_ids, device)
    proxy_nctc = pair_normalized_dependency_gap(joint, marginals)

    print(f"model={args.model}")
    print(f"device={device}")
    print(f"block_tokens={len(completion_ids)}")
    print(f"joint_logprob={sum(joint):.4f}")
    print(f"prompt_only_marginal_sum={sum(marginals):.4f}")
    print(f"pair_normalized_dependency_gap={proxy_nctc:.6f}")


@torch.no_grad()
def score_completion_block(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)

    logits = model(full_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    prompt_len = prompt_ids.shape[0]

    joint_logprobs: list[float] = []
    for offset, token_id in enumerate(completion_ids.tolist()):
        predictor_position = prompt_len + offset - 1
        joint_logprobs.append(float(log_probs[predictor_position, token_id].cpu()))

    prompt_logits = model(prompt_ids.unsqueeze(0)).logits[0, -1]
    prompt_log_probs = torch.log_softmax(prompt_logits, dim=-1)
    marginal_logprobs = [
        float(prompt_log_probs[token_id].cpu()) for token_id in completion_ids.tolist()
    ]
    return joint_logprobs, marginal_logprobs


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
