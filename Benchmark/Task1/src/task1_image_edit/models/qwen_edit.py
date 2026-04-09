from __future__ import annotations

import re
from pathlib import Path

from .base import InferenceRequest, load_rgb_image, prepare_output_path

QWEN_EDIT_REPO_ID = "Qwen/Qwen-Image-Edit"
QWEN_IMAGE_REPO_ID = "Qwen/Qwen-Image"


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    import math

    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def _resolve_inference_size(image_size: tuple[int, int], edit_image_auto_resize: bool) -> tuple[int, int]:
    if not edit_image_auto_resize:
        return image_size
    width, height = image_size
    return _calculate_dimensions(1024 * 1024, width / height)


def _resolve_lora_checkpoint(path: str | None) -> str | None:
    if path is None:
        return None
    checkpoint_path = Path(path).resolve()
    if checkpoint_path.is_file():
        return str(checkpoint_path)
    candidates = sorted(
        checkpoint_path.glob("epoch-*.safetensors"),
        key=lambda item: int(re.search(r"epoch-(\d+)", item.name).group(1)) if re.search(r"epoch-(\d+)", item.name) else -1,
    )
    if not candidates:
        raise FileNotFoundError(f"No DiffSynth LoRA checkpoint found in {checkpoint_path}")
    return str(candidates[-1])


def _resolve_model_name_or_path(path_like: str) -> tuple[str, Path | None]:
    candidate = Path(path_like).expanduser()
    if candidate.exists():
        resolved = candidate.resolve()
        return str(resolved), resolved
    return path_like, None


def _local_component_exists(local_root: Path | None, relative_path: str) -> bool:
    if local_root is None:
        return False
    normalized = relative_path.rstrip("/")
    if "*" in normalized or "?" in normalized or "[" in normalized:
        return any(local_root.glob(normalized))
    return (local_root / normalized).exists()


def _resolve_local_component_path(
    local_root: Path | None,
    relative_path: str,
    *,
    required: bool = False,
) -> str | list[str] | None:
    if local_root is None:
        return None

    normalized = relative_path.rstrip("/")
    is_glob = any(token in normalized for token in "*?[")
    if relative_path.endswith("/"):
        component_path = (local_root / normalized).resolve()
        if component_path.exists():
            return str(component_path)
        if required:
            raise FileNotFoundError(
                f"Missing required local Qwen component directory: {component_path}"
            )
        return None

    if is_glob:
        matches = sorted(local_root.glob(normalized))
        if matches:
            return [str(match.resolve()) for match in matches]
        component_parent = (local_root / normalized.split("*", 1)[0]).parent
        if required or component_parent.exists():
            raise FileNotFoundError(
                "Local Qwen checkpoint looks incomplete. "
                f"Expected files matching '{relative_path}' under {local_root}, but none were found."
            )
        return None

    component_path = (local_root / normalized).resolve()
    if component_path.exists():
        return str(component_path)
    if required or component_path.parent.exists():
        raise FileNotFoundError(
            "Local Qwen checkpoint looks incomplete. "
            f"Expected file '{component_path}' but it was not found."
        )
    return None


def _build_model_config(
    model_config_cls,
    *,
    local_root: Path | None,
    relative_path: str,
    fallback_repo_id: str,
    required_local: bool = False,
    **kwargs,
):
    local_path = _resolve_local_component_path(
        local_root,
        relative_path,
        required=required_local,
    )
    if local_path is not None:
        return model_config_cls(path=local_path, **kwargs)
    return model_config_cls(
        model_id=fallback_repo_id,
        origin_file_pattern=relative_path,
        **kwargs,
    )


class QwenEditRunner:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen-Image-Edit",
        device: str = "cuda",
        dtype: str = "bf16",
        lora_path: str | None = None,
        offload: bool = False,
        low_vram: bool = False,
        zero_cond_t: bool = False,
    ) -> None:
        import torch
        from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

        self.device = device
        resolved_model_name_or_path, local_model_root = _resolve_model_name_or_path(model_name_or_path)
        self.model_name_or_path = resolved_model_name_or_path
        self.zero_cond_t = zero_cond_t or ("2511" in resolved_model_name_or_path)
        torch_dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16}.get(dtype.lower())
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype for DiffSynth qwen-edit: {dtype}")

        common_model_kwargs = {}
        pipeline_kwargs = {}
        if low_vram or offload:
            if not device.startswith("cuda"):
                raise ValueError("DiffSynth low-vram mode currently expects a CUDA device")
            common_model_kwargs = {
                "offload_dtype": "disk",
                "offload_device": "disk",
                "onload_dtype": torch.float8_e4m3fn,
                "onload_device": "cpu",
                "preparing_dtype": torch.float8_e4m3fn,
                "preparing_device": device,
                "computation_dtype": torch_dtype,
                "computation_device": device,
            }
            pipeline_kwargs["vram_limit"] = torch.cuda.mem_get_info(device)[1] / (1024**3) - 0.5

        model_configs = [
            _build_model_config(
                ModelConfig,
                local_root=local_model_root,
                relative_path="transformer/diffusion_pytorch_model*.safetensors",
                fallback_repo_id=resolved_model_name_or_path,
                required_local=local_model_root is not None,
                **common_model_kwargs,
            ),
            _build_model_config(
                ModelConfig,
                local_root=local_model_root,
                relative_path="text_encoder/model*.safetensors",
                fallback_repo_id=QWEN_IMAGE_REPO_ID,
                **common_model_kwargs,
            ),
            _build_model_config(
                ModelConfig,
                local_root=local_model_root,
                relative_path="vae/diffusion_pytorch_model.safetensors",
                fallback_repo_id=QWEN_IMAGE_REPO_ID,
                **common_model_kwargs,
            ),
        ]
        tokenizer_config = _build_model_config(
            ModelConfig,
            local_root=local_model_root,
            relative_path="tokenizer/",
            fallback_repo_id=QWEN_IMAGE_REPO_ID,
        )
        processor_config = _build_model_config(
            ModelConfig,
            local_root=local_model_root,
            relative_path="processor/",
            fallback_repo_id=QWEN_EDIT_REPO_ID,
        )

        self.pipeline = QwenImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
            **pipeline_kwargs,
        )

        resolved_lora = _resolve_lora_checkpoint(lora_path)
        if resolved_lora:
            self.pipeline.load_lora(self.pipeline.dit, resolved_lora)

    def run(self, request: InferenceRequest) -> str:
        image = load_rgb_image(request.input_image)
        width, height = _resolve_inference_size(image.size, request.edit_image_auto_resize)
        effective_zero_cond_t = self.zero_cond_t or request.zero_cond_t
        edit_image = [image] if effective_zero_cond_t else image
        result = self.pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or "",
            edit_image=edit_image,
            edit_image_auto_resize=request.edit_image_auto_resize,
            cfg_scale=request.true_cfg_scale,
            seed=request.seed,
            num_inference_steps=request.num_inference_steps,
            height=height,
            width=width,
            zero_cond_t=effective_zero_cond_t,
        )
        output_path = prepare_output_path(request.output_path)
        result.save(output_path)
        return str(output_path)
