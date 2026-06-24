"""Diffusers-style adapters for DDPO rollouts and image reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mindrl.diffusion_training import DiffusionTrajectory


@dataclass(frozen=True)
class DiffusionGeneration:
    """One generated image plus its DDPO trajectory metadata."""

    sample_id: str
    prompt: str
    image: str
    caption: str
    trajectory: DiffusionTrajectory


@dataclass(frozen=True)
class DiffusionRolloutBatch:
    """Collected diffusion rollouts for reward and reporting."""

    trajectories: tuple[DiffusionTrajectory, ...]
    images: dict[str, str]
    captions: dict[str, str]


class DiffusionPipelineAdapter(Protocol):
    """Minimal protocol implemented by diffusers or mock pipelines."""

    def generate(self, prompt: str, sample_id: str) -> DiffusionGeneration:
        ...


class MockDiffusionPipeline:
    """Deterministic pipeline used by DDPO adapter smoke tests."""

    def __init__(
        self,
        images: dict[str, str],
        captions: dict[str, str],
        step_logprobs: dict[str, tuple[float, ...]],
        anchor_distances: dict[str, float],
    ) -> None:
        self.images = images
        self.captions = captions
        self.step_logprobs = step_logprobs
        self.anchor_distances = anchor_distances

    def generate(self, prompt: str, sample_id: str) -> DiffusionGeneration:
        if prompt not in self.images:
            raise ValueError(f"missing mock image for prompt {prompt}")
        caption = self.captions.get(prompt, "")
        trajectory = DiffusionTrajectory(
            sample_id=sample_id,
            prompt=prompt,
            step_logprobs=self.step_logprobs[prompt],
            anchor_distance=self.anchor_distances.get(prompt, 0.0),
            image_caption=caption,
        )
        return DiffusionGeneration(
            sample_id=sample_id,
            prompt=prompt,
            image=self.images[prompt],
            caption=caption,
            trajectory=trajectory,
        )


def collect_diffusion_rollouts(
    prompts: tuple[str, ...],
    pipeline: DiffusionPipelineAdapter,
) -> DiffusionRolloutBatch:
    """Collect generated images and DDPO trajectory metadata."""

    generations = [
        pipeline.generate(prompt, sample_id=f"diff-{index}")
        for index, prompt in enumerate(prompts)
    ]
    return DiffusionRolloutBatch(
        trajectories=tuple(generation.trajectory for generation in generations),
        images={generation.sample_id: generation.image for generation in generations},
        captions={generation.sample_id: generation.caption for generation in generations},
    )


def build_image_grid_manifest(batch: DiffusionRolloutBatch) -> str:
    """Build a markdown manifest for image-grid reports."""

    lines = ["# Diffusion Samples", ""]
    for trajectory in batch.trajectories:
        lines.append(
            f"- `{trajectory.sample_id}` prompt=`{trajectory.prompt}` "
            f"caption=`{batch.captions.get(trajectory.sample_id, '')}`"
        )
    return "\n".join(lines) + "\n"
