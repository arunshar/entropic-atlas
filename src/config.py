"""
Entropic Atlas — Centralized Configuration

All configurable parameters in one place.
Environment variables override defaults.
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
        default_factory=lambda: os.getenv("ATLAS_STRONG_MODEL", "openai/gpt-4.1")
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
    max_code_iterations: int = 3

    @property
    def model_tiers(self) -> dict[str, str]:
        return {
            "fast": self.fast_model,
            "standard": self.standard_model,
            "strong": self.strong_model,
            "vision": self.vision_model,
        }
