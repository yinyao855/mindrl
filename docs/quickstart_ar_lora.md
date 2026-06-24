# Quickstart: AR LoRA Plan

The current MVP creates validated LoRA/QLoRA training plans before launching
real PEFT or distributed jobs.

```python
from mindrl.ar_trainer import qwen_lora_preset, summarize_ar_trainer_plan

plan = qwen_lora_preset("qwen-7b")
report = summarize_ar_trainer_plan("ar-lora-plan-qwen-7b", plan)
```

Preset scales:

- `qwen-0.5b`
- `qwen-1.5b`
- `qwen-7b`
- `qwen-13b`

The report records:

- effective batch size
- rough VRAM estimate
- LoRA rank and alpha
- quantization mode
- checkpoint/training-step knobs

This layer is intentionally dependency-light. The next implementation step is a
real adapter that passes this plan to PEFT/TRL/verl without changing the public
configuration objects.
