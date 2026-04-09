from __future__ import annotations

import re
from pathlib import Path

from .base import InferenceRequest, load_rgb_image, parse_dtype, prepare_output_path


def _resolve_lora_checkpoint(path: str | None) -> str | None:
    if path is None:
        return None
    checkpoint_path = Path(path).resolve()
    if checkpoint_path.is_file():
        if (checkpoint_path.parent / "adapter_config.json").exists():
            return str(checkpoint_path.parent)
        raise FileNotFoundError(
            f"LongCat LoRA expects a PEFT checkpoint directory, but got file: {checkpoint_path}"
        )
    if (checkpoint_path / "adapter_config.json").exists():
        return str(checkpoint_path)
    candidates = sorted(
        checkpoint_path.glob("checkpoints-*"),
        key=lambda item: int(re.search(r"checkpoints-(\d+)", item.name).group(1))
        if re.search(r"checkpoints-(\d+)", item.name)
        else -1,
    )
    if not candidates:
        raise FileNotFoundError(f"No LongCat LoRA checkpoint found in {checkpoint_path}")
    return str(candidates[-1])


class LongCatEditRunner:
    def __init__(
        self,
        model_name_or_path: str = "meituan-longcat/LongCat-Image-Edit-Turbo",
        device: str = "cuda",
        dtype: str = "bf16",
        lora_path: str | None = None,
        offload: bool = False,
    ) -> None:
        import torch
        from diffusers import LongCatImageEditPipeline, LongCatImageTransformer2DModel

        self.device = device
        self.dtype = parse_dtype(dtype)
        resolved_lora = _resolve_lora_checkpoint(lora_path)

        pipeline_kwargs = {"torch_dtype": self.dtype}
        if resolved_lora:
            from peft import PeftModel

            transformer = LongCatImageTransformer2DModel.from_pretrained(
                model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.dtype,
                use_safetensors=True,
            ).to(device)
            transformer = PeftModel.from_pretrained(transformer, resolved_lora)
            transformer = transformer.merge_and_unload()
            transformer.to(device=device, dtype=self.dtype)
            pipeline_kwargs["transformer"] = transformer

        self.pipeline = LongCatImageEditPipeline.from_pretrained(model_name_or_path, **pipeline_kwargs)

        if offload:
            if not device.startswith("cuda"):
                raise ValueError("LongCat CPU offload currently expects a CUDA device")
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(device)

        self._torch = torch

    def run(self, request: InferenceRequest) -> str:
        image = load_rgb_image(request.input_image)
        generator = self._torch.Generator("cpu").manual_seed(request.seed)
        result = self.pipeline(
            image,
            request.prompt,
            negative_prompt=request.negative_prompt or "",
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
        ).images[0]
        output_path = prepare_output_path(request.output_path)
        result.save(output_path)
        return str(output_path)
