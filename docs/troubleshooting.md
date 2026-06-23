# Troubleshooting

## Smoke Tests

Run the dependency-light path first:

```bash
uv run python examples/run_mvp_smoke.py
uv run python -m unittest discover -s tests -p "test_*.py"
```

If this fails, fix the framework layer before trying real models.

## Model Downloads

Use mock smoke tests when Hugging Face cache or network access is unavailable.
Only switch to real Qwen/Llama checkpoints after the smoke reports are stable.

## 7B/13B Runs

The MVP exposes objective and report APIs, but does not implement distributed
Ray/vLLM/FSDP scheduling. For 7B/13B, prefer LoRA/QLoRA and plug the rollout
engine into TRL, verl, or OpenRLHF.

## Diffusion Runs

The first DDPO module is an adapter target, not a full diffusers trainer. Real
diffusion experiments should populate `DiffusionTrajectory` from a pipeline and
keep reward functions small enough to inspect for reward hacking.
