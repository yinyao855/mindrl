"""Minimal PEFT LoRA update smoke for real AR models."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindrl.core import AlgorithmConfig, TrainReport
from mindrl.grpo import NumericAnswerRewardAdapter
from mindrl.hf_policy import DeviceSpec, format_privileged_context, resolve_device


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    target: str


@dataclass(frozen=True)
class PeftUpdateResult:
    report: TrainReport
    before_loss: float
    after_loss: float
    before_reward: float
    after_reward: float


@dataclass(frozen=True)
class OPDRolloutRecord:
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    teacher_logprobs: torch.Tensor
    teacher_entropies: torch.Tensor


@dataclass(frozen=True)
class PeftOPDUpdateResult:
    report: TrainReport
    before_loss: float
    after_loss: float
    before_reward: float
    after_reward: float


def build_sft_text(example: SFTExample, eos_token: str = "") -> str:
    return f"{example.prompt}{example.target}{eos_token}"


def summarize_update(
    run_name: str,
    model_name: str,
    before_loss: float,
    after_loss: float,
    before_reward: float,
    after_reward: float,
    trainable_parameters: int,
) -> TrainReport:
    return TrainReport(
        run_name=run_name,
        algorithm=AlgorithmConfig(name="peft_sft", branch="ar"),
        metrics={
            "before_loss": before_loss,
            "after_loss": after_loss,
            "loss_delta": after_loss - before_loss,
            "before_reward": before_reward,
            "after_reward": after_reward,
            "reward_delta": after_reward - before_reward,
            "trainable_parameters": float(trainable_parameters),
        },
        artifacts={"model": model_name},
    )


def clipped_opd_loss_tensor(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    per_token_clip: float | None,
) -> torch.Tensor:
    """Differentiable clipped OPD token loss for one rollout."""

    if student_logprobs.shape != teacher_logprobs.shape:
        raise ValueError("student and teacher logprobs must have matching shape")
    if student_logprobs.numel() == 0:
        return student_logprobs.sum()
    token_losses = torch.abs(student_logprobs - teacher_logprobs)
    if per_token_clip is not None:
        token_losses = torch.clamp(token_losses, max=per_token_clip)
    return token_losses.mean()


def summarize_opd_update(
    run_name: str,
    model_name: str,
    before_loss: float,
    after_loss: float,
    raw_objective: float,
    clipped_objective: float,
    clipped_token_ratio: float,
    mean_teacher_entropy: float,
    before_reward: float,
    after_reward: float,
    trainable_parameters: int,
) -> TrainReport:
    return TrainReport(
        run_name=run_name,
        algorithm=AlgorithmConfig(name="peft_opd", branch="ar"),
        metrics={
            "before_loss": before_loss,
            "after_loss": after_loss,
            "loss_delta": after_loss - before_loss,
            "raw_objective": raw_objective,
            "clipped_objective": clipped_objective,
            "clipped_token_ratio": clipped_token_ratio,
            "mean_teacher_entropy": mean_teacher_entropy,
            "before_reward": before_reward,
            "after_reward": after_reward,
            "reward_delta": after_reward - before_reward,
            "trainable_parameters": float(trainable_parameters),
        },
        artifacts={"model": model_name},
    )


def run_peft_sft_update(
    model_name: str,
    examples: tuple[SFTExample, ...],
    device: DeviceSpec = "auto",
    local_files_only: bool = True,
    cache_dir: str | None = None,
    learning_rate: float = 1e-3,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    max_steps: int = 5,
    target_modules: tuple[str, ...] | None = None,
    dtype: str = "auto",
) -> PeftUpdateResult:
    """Run a tiny real LoRA SFT update and report before/after metrics."""

    if not examples:
        raise ValueError("examples must not be empty")
    device_obj = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    torch_dtype = _resolve_torch_dtype(dtype, device_obj)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    ).to(device_obj)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    modules = target_modules or _default_lora_targets(model)
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=list(modules),
    )
    model = get_peft_model(model, peft_config)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    batch = _encode_examples(tokenizer, examples, device_obj)
    before_loss = _loss(model, batch)
    before_reward = _mean_numeric_reward(model, tokenizer, examples, device_obj)
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, labels=batch["input_ids"]).loss
        loss.backward()
        optimizer.step()
    after_loss = _loss(model, batch)
    after_reward = _mean_numeric_reward(model, tokenizer, examples, device_obj)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    report = summarize_update(
        run_name=f"peft-sft-{model_name}",
        model_name=model_name,
        before_loss=before_loss,
        after_loss=after_loss,
        before_reward=before_reward,
        after_reward=after_reward,
        trainable_parameters=trainable_parameters,
    )
    return PeftUpdateResult(report, before_loss, after_loss, before_reward, after_reward)


def run_peft_opd_update(
    model_name: str,
    examples: tuple[SFTExample, ...],
    teacher_model_name: str | None = None,
    device: DeviceSpec = "auto",
    local_files_only: bool = True,
    cache_dir: str | None = None,
    learning_rate: float = 1e-4,
    lora_rank: int = 4,
    lora_alpha: int = 8,
    max_steps: int = 1,
    max_new_tokens: int = 12,
    per_token_clip: float | None = 0.25,
    target_modules: tuple[str, ...] | None = None,
    dtype: str = "auto",
) -> PeftOPDUpdateResult:
    """Run a one-batch PEFT LoRA update against privileged OPD token targets."""

    if not examples:
        raise ValueError("examples must not be empty")
    device_obj = resolve_device(device)
    teacher_model_name = teacher_model_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
    )
    torch_dtype = _resolve_torch_dtype(dtype, device_obj)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    ).to(device_obj)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
    ).to(device_obj)
    teacher_model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    modules = target_modules or _default_lora_targets(model)
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=list(modules),
    )
    model = get_peft_model(model, peft_config)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    rollouts = _build_opd_rollouts(
        model,
        teacher_model,
        tokenizer,
        examples,
        device_obj,
        max_new_tokens,
    )
    before_loss_tensor = _peft_opd_batch_loss(model, rollouts, device_obj, per_token_clip)
    before_loss = float(before_loss_tensor.detach().cpu())
    diagnostics = _opd_batch_diagnostics(model, rollouts, device_obj, per_token_clip)
    before_reward = _mean_numeric_reward(model, tokenizer, examples, device_obj)
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _peft_opd_batch_loss(model, rollouts, device_obj, per_token_clip)
        loss.backward()
        optimizer.step()
    after_loss_tensor = _peft_opd_batch_loss(model, rollouts, device_obj, per_token_clip)
    after_loss = float(after_loss_tensor.detach().cpu())
    after_reward = _mean_numeric_reward(model, tokenizer, examples, device_obj)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    report = summarize_opd_update(
        run_name=f"peft-opd-{model_name}",
        model_name=model_name,
        before_loss=before_loss,
        after_loss=after_loss,
        raw_objective=diagnostics["raw_objective"],
        clipped_objective=before_loss,
        clipped_token_ratio=diagnostics["clipped_token_ratio"],
        mean_teacher_entropy=diagnostics["mean_teacher_entropy"],
        before_reward=before_reward,
        after_reward=after_reward,
        trainable_parameters=trainable_parameters,
    )
    return PeftOPDUpdateResult(report, before_loss, after_loss, before_reward, after_reward)


def _encode_examples(tokenizer, examples: tuple[SFTExample, ...], device: torch.device) -> dict[str, torch.Tensor]:
    texts = [
        build_sft_text(example, eos_token=tokenizer.eos_token or "")
        for example in examples
    ]
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _build_opd_rollouts(
    model,
    teacher_model,
    tokenizer,
    examples: tuple[SFTExample, ...],
    device: torch.device,
    max_new_tokens: int,
) -> tuple[OPDRolloutRecord, ...]:
    model.eval()
    records: list[OPDRolloutRecord] = []
    for example in examples:
        prompt_ids = tokenizer(
            example.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].to(device)
        output_ids = model.generate(
            prompt_ids.unsqueeze(0),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        completion_ids = output_ids[prompt_ids.shape[0] :].to(device)
        teacher_prompt = format_privileged_context(example.prompt, example.target)
        teacher_prompt_ids = tokenizer(
            teacher_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0].to(device)
        teacher_logprobs, teacher_entropies = _teacher_completion_terms(
            teacher_model,
            teacher_prompt_ids,
            completion_ids,
            device,
        )
        records.append(
            OPDRolloutRecord(
                prompt_ids=prompt_ids.detach(),
                completion_ids=completion_ids.detach(),
                teacher_logprobs=teacher_logprobs.detach(),
                teacher_entropies=teacher_entropies.detach(),
            )
        )
    model.train()
    return tuple(records)


def _peft_opd_batch_loss(
    model,
    rollouts: tuple[OPDRolloutRecord, ...],
    device: torch.device,
    per_token_clip: float | None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for rollout in rollouts:
        student_logprobs = _completion_logprob_tensor(
            model,
            rollout.prompt_ids,
            rollout.completion_ids,
            device,
        )
        losses.append(
            clipped_opd_loss_tensor(
                student_logprobs,
                rollout.teacher_logprobs.to(device),
                per_token_clip,
            )
        )
    return torch.stack(losses).mean()


def _completion_logprob_tensor(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if completion_ids.numel() == 0:
        return torch.empty(0, device=device)
    prompt_ids = prompt_ids.to(device)
    completion_ids = completion_ids.to(device)
    full_ids = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)
    logits = model(full_ids).logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    prompt_len = prompt_ids.shape[0]
    values = []
    for offset, token_id in enumerate(completion_ids.tolist()):
        predictor_position = prompt_len + offset - 1
        values.append(log_probs[predictor_position, int(token_id)])
    return torch.stack(values)


@torch.no_grad()
def _teacher_completion_terms(
    teacher_model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logprobs = _completion_logprob_tensor(teacher_model, prompt_ids, completion_ids, device)
    if completion_ids.numel() == 0:
        return logprobs.detach(), torch.empty(0, device=device)
    full_ids = torch.cat([prompt_ids.to(device), completion_ids.to(device)], dim=0).unsqueeze(0)
    logits = teacher_model(full_ids).logits[0]
    prompt_len = prompt_ids.shape[0]
    entropies = []
    for offset in range(int(completion_ids.shape[0])):
        predictor_position = prompt_len + offset - 1
        probs = torch.softmax(logits[predictor_position], dim=-1)
        log_probs = torch.log_softmax(logits[predictor_position], dim=-1)
        entropies.append(-(probs * log_probs).sum())
    return logprobs.detach(), torch.stack(entropies).detach()


@torch.no_grad()
def _opd_batch_diagnostics(
    model,
    rollouts: tuple[OPDRolloutRecord, ...],
    device: torch.device,
    per_token_clip: float | None,
) -> dict[str, float]:
    gaps = []
    entropies = []
    for rollout in rollouts:
        student_logprobs = _completion_logprob_tensor(
            model,
            rollout.prompt_ids,
            rollout.completion_ids,
            device,
        )
        teacher_logprobs = rollout.teacher_logprobs.to(device)
        entropies.extend(float(value) for value in rollout.teacher_entropies.detach().cpu())
        token_gaps = torch.abs(student_logprobs - teacher_logprobs)
        gaps.extend(float(value) for value in token_gaps.detach().cpu())
    if not gaps:
        return {"raw_objective": 0.0, "clipped_token_ratio": 0.0, "mean_teacher_entropy": 0.0}
    clipped = sum(1 for gap in gaps if per_token_clip is not None and gap > per_token_clip)
    return {
        "raw_objective": sum(gaps) / len(gaps),
        "clipped_token_ratio": clipped / len(gaps),
        "mean_teacher_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
    }


@torch.no_grad()
def _loss(model, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    value = float(model(**batch, labels=batch["input_ids"]).loss.detach().cpu())
    model.train()
    return value


def _default_lora_targets(model) -> tuple[str, ...]:
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    for candidates in [
        ("q_proj", "v_proj"),
        ("c_attn",),
        ("query_key_value",),
    ]:
        if all(candidate in module_names for candidate in candidates):
            return candidates
    raise ValueError("could not infer LoRA target modules for model")


def _resolve_torch_dtype(dtype: str, device: torch.device):
    if dtype == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"unknown dtype: {dtype}")


@torch.no_grad()
def _mean_numeric_reward(model, tokenizer, examples: tuple[SFTExample, ...], device: torch.device) -> float:
    model.eval()
    adapter = NumericAnswerRewardAdapter({example.prompt: example.target for example in examples})
    rewards: list[float] = []
    for index, example in enumerate(examples):
        input_ids = tokenizer(
            example.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
        completion = tokenizer.decode(output_ids[input_ids.shape[1] :], skip_special_tokens=True)
        from mindrl.core import RolloutBatch, RolloutSample

        batch = RolloutBatch(
            samples=(
                RolloutSample(
                    sample_id=f"eval-{index}",
                    prompt=example.prompt,
                    response=completion,
                    branch="ar",
                    metadata={"prompt_id": example.prompt},
                ),
            )
        )
        rewards.append(adapter.score(batch).sample_rewards[f"eval-{index}"])
    model.train()
    return sum(rewards) / len(rewards)
