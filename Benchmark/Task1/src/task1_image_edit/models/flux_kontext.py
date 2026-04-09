from __future__ import annotations

from .base import InferenceRequest, load_rgb_image, parse_dtype, prepare_output_path


class FluxKontextRunner:
    def __init__(
        self,
        model_name_or_path: str = "black-forest-labs/FLUX.1-Kontext-dev",
        device: str = "cuda",
        dtype: str = "bf16",
        lora_path: str | None = None,
        offload: bool = False,
    ) -> None:
        import torch
        from diffusers import FluxKontextPipeline

        self.device = device
        self.dtype = parse_dtype(dtype)
        self.pipeline = FluxKontextPipeline.from_pretrained(model_name_or_path, torch_dtype=self.dtype)

        if lora_path:
            self.pipeline.load_lora_weights(lora_path)

        if offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(device)

        self._torch = torch

    def run(self, request: InferenceRequest) -> str:
        image = load_rgb_image(request.input_image)
        generator_device = "cuda" if self.device.startswith("cuda") else "cpu"
        generator = self._torch.Generator(generator_device).manual_seed(request.seed)
        result = self.pipeline(
            image=image,
            prompt=request.prompt,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            max_sequence_length=request.max_sequence_length,
            generator=generator,
        ).images[0]
        output_path = prepare_output_path(request.output_path)
        result.save(output_path)
        return str(output_path)
