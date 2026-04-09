from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceRequest:
    input_image: str
    output_path: str
    prompt: str
    negative_prompt: str | None = None
    seed: int = 42
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    image_guidance_scale: float = 1.5
    true_cfg_scale: float = 4.0
    max_sequence_length: int = 512
    edit_image_auto_resize: bool = True
    zero_cond_t: bool = False


def parse_dtype(dtype_name: str):
    import torch

    name = dtype_name.strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[name]


def load_rgb_image(path: str):
    from PIL import Image

    return Image.open(path).convert("RGB")


def prepare_output_path(path: str) -> Path:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
