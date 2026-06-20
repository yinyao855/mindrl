"""Hugging Face causal-LM scorer for AR proxy nCTC pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DeviceSpec = Literal["auto", "cpu", "cuda", "mps"]


@dataclass
class CompletionBlockScores:
    """Teacher-forced logprob terms for one prompt/completion block."""

    joint_chain_logprobs: list[float]
    prompt_only_marginal_logprobs: list[float]
    token_count: int


class HFCausalLMScorer:
    """Teacher-forcing scorer for causal LMs."""

    def __init__(
        self,
        model_name: str,
        device: DeviceSpec = "auto",
        local_files_only: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def score_text_block(
        self,
        prompt: str,
        completion: str,
        max_block_tokens: int,
    ) -> CompletionBlockScores:
        """Tokenize and score a completion block under the prompt."""

        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        completion_ids = self.tokenizer(
            completion,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0][:max_block_tokens]

        if len(prompt_ids) == 0:
            raise ValueError("prompt must tokenize to at least one token")
        if len(completion_ids) == 0:
            raise ValueError("completion must tokenize to at least one token")

        joint, marginals = score_completion_token_ids(
            self.model,
            prompt_ids,
            completion_ids,
            self.device,
        )
        return CompletionBlockScores(
            joint_chain_logprobs=joint,
            prompt_only_marginal_logprobs=marginals,
            token_count=len(completion_ids),
        )


@torch.no_grad()
def score_completion_token_ids(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """Score joint AR terms and prompt-only one-token marginal terms."""

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


def resolve_device(device: DeviceSpec) -> torch.device:
    """Resolve a user-facing device flag into a torch device."""

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(device)
