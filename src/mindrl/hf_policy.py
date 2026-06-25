"""Hugging Face causal-LM rollout adapters for real AR smoke runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindrl.core import RolloutBatch, RolloutSample, TeacherSignal
from mindrl.hf_scorer import DeviceSpec, resolve_device, score_completion_token_ids


@dataclass(frozen=True)
class GenerationRecord:
    """One generated completion plus its teacher-forced token logprobs."""

    sample_id: str
    prompt: str
    response: str
    token_logprobs: tuple[float, ...]


def records_to_rollout_batch(records: tuple[GenerationRecord, ...]) -> RolloutBatch:
    """Convert generated records into an AR rollout batch."""

    samples: list[RolloutSample] = []
    group_counts: dict[str, int] = {}
    for record in records:
        group_index = group_counts.get(record.prompt, 0)
        group_counts[record.prompt] = group_index + 1
        samples.append(
            RolloutSample(
                sample_id=record.sample_id,
                prompt=record.prompt,
                response=record.response,
                branch="ar",
                metadata={
                    "prompt_id": record.prompt,
                    "group_index": group_index,
                    "token_count": len(record.token_logprobs),
                },
            )
        )
    return RolloutBatch(samples=tuple(samples))


def format_privileged_context(prompt: str, answer: str) -> str:
    """Build a teacher context with privileged answer information."""

    return f"Correct answer: {answer}\nProblem: {prompt}\nStudent response:"


class HFCausalLMGroupPolicy:
    """Group rollout policy backed by a Hugging Face causal LM."""

    def __init__(
        self,
        model_name: str,
        device: DeviceSpec = "auto",
        local_files_only: bool = True,
        cache_dir: str | None = None,
        max_new_tokens: int = 16,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        self.device = resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._records: tuple[GenerationRecord, ...] = ()

    @torch.no_grad()
    def generate_records(self, prompts: tuple[str, ...], group_size: int = 1) -> tuple[GenerationRecord, ...]:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        records: list[GenerationRecord] = []
        for prompt_index, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)
            for group_index in range(group_size):
                output_ids = self.model.generate(
                    prompt_ids,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )[0]
                completion_ids = output_ids[prompt_ids.shape[1] :]
                response = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                token_logprobs = self._score_completion_ids(prompt_ids[0], completion_ids)
                records.append(
                    GenerationRecord(
                        sample_id=f"hf-{prompt_index}-{group_index}",
                        prompt=prompt,
                        response=response,
                        token_logprobs=tuple(token_logprobs),
                    )
                )
        self._records = tuple(records)
        return self._records

    def rollout(self, prompts: tuple[str, ...], group_size: int = 1) -> RolloutBatch:
        return records_to_rollout_batch(self.generate_records(prompts, group_size))

    def logprob_ratios(self, batch: RolloutBatch) -> dict[str, float]:
        # Without a reference model loaded, this real smoke uses ratio 1.0 and
        # still records per-sample logprob diagnostics separately.
        return {sample_id: 1.0 for sample_id in batch.sample_ids}

    def kl(self, batch: RolloutBatch) -> dict[str, float]:
        return {sample_id: 0.0 for sample_id in batch.sample_ids}

    def score(self, batch: RolloutBatch) -> dict[str, tuple[float, ...]]:
        records = {record.sample_id: record for record in self._records}
        scores: dict[str, tuple[float, ...]] = {}
        for sample_id in batch.sample_ids:
            if sample_id not in records:
                raise ValueError(f"missing generation record for {sample_id}")
            scores[sample_id] = records[sample_id].token_logprobs
        return scores

    def sequence_logprobs(self) -> dict[str, float]:
        return {
            record.sample_id: sum(record.token_logprobs)
            for record in self._records
        }

    def _score_completion_ids(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> list[float]:
        if completion_ids.numel() == 0:
            return []
        joint, _ = score_completion_token_ids(
            self.model,
            prompt_ids.detach().cpu(),
            completion_ids.detach().cpu(),
            self.device,
        )
        return joint


class HFCausalLMTeacherSignalAdapter:
    """Teacher adapter that scores student responses under privileged contexts."""

    def __init__(
        self,
        model_name: str,
        answers: dict[str, str],
        device: DeviceSpec = "auto",
        local_files_only: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        self.answers = answers
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.entropies: dict[str, tuple[float, ...]] = {}

    @torch.no_grad()
    def signals_for(self, batch: RolloutBatch) -> tuple[TeacherSignal, ...]:
        signals: list[TeacherSignal] = []
        for sample in batch.samples:
            answer = self.answers[str(sample.metadata["prompt_id"])]
            teacher_prompt = format_privileged_context(sample.prompt, answer)
            prompt_ids = self.tokenizer(
                teacher_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            completion_ids = self.tokenizer(
                sample.response,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            if completion_ids.numel() == 0:
                token_logprobs: tuple[float, ...] = ()
                entropies: tuple[float, ...] = ()
            else:
                token_logprobs = tuple(
                    score_completion_token_ids(
                        self.model,
                        prompt_ids,
                        completion_ids,
                        self.device,
                    )[0]
                )
                entropies = tuple(
                    self._completion_entropies(prompt_ids, completion_ids)
                )
            self.entropies[sample.sample_id] = entropies
            signals.append(
                TeacherSignal(
                    sample_id=sample.sample_id,
                    token_logprobs=token_logprobs,
                )
            )
        return tuple(signals)

    def _completion_entropies(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> list[float]:
        prompt_ids = prompt_ids.to(self.device)
        completion_ids = completion_ids.to(self.device)
        full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)
        logits = self.model(full_ids).logits[0]
        prompt_len = prompt_ids.shape[0]
        entropies: list[float] = []
        for offset in range(int(completion_ids.shape[0])):
            predictor_position = prompt_len + offset - 1
            probs = torch.softmax(logits[predictor_position], dim=-1)
            log_probs = torch.log_softmax(logits[predictor_position], dim=-1)
            entropies.append(float(-(probs * log_probs).sum().cpu()))
        return entropies
