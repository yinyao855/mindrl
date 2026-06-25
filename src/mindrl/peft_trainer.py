"""Minimal PEFT LoRA update smoke for real AR models."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mindrl.core import AlgorithmConfig, TrainReport
from mindrl.grpo import NumericAnswerRewardAdapter
from mindrl.hf_policy import DeviceSpec, resolve_device


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


def _encode_examples(tokenizer, examples: tuple[SFTExample, ...], device: torch.device) -> dict[str, torch.Tensor]:
    texts = [
        build_sft_text(example, eos_token=tokenizer.eos_token or "")
        for example in examples
    ]
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


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
