"""
Entropic Atlas: Centralized Configuration

All configurable parameters in one place.
Environment variables override defaults.

Model tier layout (post 2026-04-11 retune):
  - fast     (gpt-4.1-mini)        cheap classification, parsing, formatting
  - standard (gpt-4.1)              code generation and mid-complexity analysis
  - strong   (claude-opus-4-6)      spatial reasoning, reflection, hard MLE tasks
  - vision   (gpt-4.1)              multimodal image/PDF/video description

Previously 'standard' and 'strong' both pointed at gpt-4.1, so the cost router
was effectively 2-tier. Splitting strong onto Anthropic Claude Opus 4.6 gives
a real frontier tier on the tasks that actually move the score needle
(FieldWorkArena reflection, MLE-Bench code refinement).
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # === Model Tiers ===
    fast_model: str = field(
        default_factory=lambda: os.getenv("ATLAS_FAST_MODEL", "openai/gpt-4.1-mini")
    )
    standard_model: str = field(
        default_factory=lambda: os.getenv("ATLAS_STANDARD_MODEL", "openai/gpt-4.1")
    )
    strong_model: str = field(
        default_factory=lambda: os.getenv(
            "ATLAS_STRONG_MODEL", "anthropic/claude-opus-4-6"
        )
    )
    vision_model: str = field(
        default_factory=lambda: os.getenv("ATLAS_VISION_MODEL", "openai/gpt-4.1")
    )

    # === Cost Budgets ===
    max_tokens_per_task: int = 150_000
    max_reflection_rounds: int = 2

    # === FieldWork-specific ===
    max_video_frames: int = 30
    spatial_precision: int = 2  # decimal places for coordinates

    # === MLE-Bench-specific ===
    code_execution_timeout: int = 600  # seconds per code execution
    max_code_iterations: int = 3  # error-recovery retries on the first successful build
    # Score-driven iterations AFTER the first successful run. Each iteration
    # asks the strong model to propose an improved pipeline given the prior
    # code and its validation score, re-runs it, and keeps the best one.
    # Set to 0 to disable refinement entirely.
    max_refinement_iterations: int = 2
    # Hard wall-clock ceiling across all refinement iterations. Protects the
    # MLE-Bench per-task budget when a pipeline is slow to train.
    refinement_wall_time_seconds: int = 900

    @property
    def model_tiers(self) -> dict[str, str]:
        return {
            "fast": self.fast_model,
            "standard": self.standard_model,
            "strong": self.strong_model,
            "vision": self.vision_model,
        }
