# Task1-ImageEdit

This directory provides 3 unified image-edit baselines:

- `Qwen/Qwen-Image-Edit`
- `meituan-longcat/LongCat-Image-Edit-Turbo`
- `black-forest-labs/FLUX.1-Kontext-dev`

**Task**: Given a model outfit image, generate a `flat-lay view + white background` product image for each garment item.

## 1. Directory Structure

```text
.
├── configs/                    # Accelerate distributed training configs
├── environment.yml
├── examples/                   # Example manifest
├── hf_datasets/                # HuggingFace dataset scripts
├── requirements/               # Python dependencies
├── scripts/
│   ├── infer.py                # Unified inference CLI
│   ├── train_qwen_edit.py      # Qwen-Edit LoRA training (DiffSynth-Studio)
│   ├── train_longcat_edit.py   # LongCat-Image edit LoRA training
│   ├── train_flux_kontext.py   # FLUX Kontext LoRA training (diffusers)
│   ├── eval_outputs.py         # Evaluation metrics
│   ├── enrich_manifest.py      # Auto-fill model-specific prompts in manifest
│   ├── prepare_dataset.py      # Generate train/val/test manifest from raw data
│   └── setup_env.sh            # One-click environment setup
└── src/task1_image_edit/       # Core source code
```

## 2. Environment Setup

Requires Linux + CUDA 12.4 + Python 3.11. `FLUX.1-Kontext-dev` is a gated model — run `hf auth login` and request access before training/inference.

### 2.1 Clone External Dependencies

The following external repos are required. Clone them into the `external/` directory:

```bash
mkdir -p external
git clone https://github.com/huggingface/diffusers.git external/diffusers
git clone https://github.com/modelscope/DiffSynth-Studio.git external/DiffSynth-Studio
git clone https://github.com/meituan-longcat/LongCat-Image.git external/LongCat-Image
```

### 2.2 Conda

```bash
conda env create -f environment.yml
conda activate task1-image-edit
```

### 2.3 One-Click Script

After cloning the external repos above:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
hf auth login
```

This script will:

1. Create `.venv`
2. Install PyTorch 2.5.1 CUDA 12.4
3. Install project dependencies
4. Install `diffusers`, `LongCat-Image`, `DiffSynth-Studio`, and this project as editable packages
5. Install `dreambooth` example dependencies

## 3. Data Format

Two data entry points are supported:

1. Raw directory `data/`
2. Unified `manifest.jsonl`

### 3.1 Raw Directory `data/`

```text
data/
├── Male_1-300/
│   ├── bid_pid_dict.npy
│   ├── label.csv
│   ├── look.csv
│   └── photos/
├── Female_1-500/
└── Child_1-200/
```

Generate manifest from raw data:

```bash
python scripts/prepare_dataset.py \
  --data-root data \
  --output-dir output/manifests
```

### 3.2 Unified `manifest.jsonl`

Each line is one sample with at minimum:

```json
{
  "sample_id": "look_0001",
  "source_image": "path/to/model.jpg",
  "target_image": "path/to/flatlay.png",
  "items": ["cropped denim jacket"]
}
```

Auto-fill model-specific prompts:

```bash
python scripts/enrich_manifest.py \
  --input examples/train_manifest.example.jsonl \
  --output data/manifest/train_enriched.jsonl
```

## 4. Inference

### 4.1 Qwen-Image-Edit

```bash
python scripts/infer.py \
  --model qwen_edit \
  --input-image path/to/look.jpg \
  --output outputs/qwen/look_0001.png \
  --device cuda \
  --dtype bf16 \
  --steps 28 \
  --true-cfg-scale 4.0
```

With LoRA:

```bash
python scripts/infer.py \
  --model qwen_edit \
  --input-image path/to/look.jpg \
  --output outputs/qwen_lora/look_0001.png \
  --lora-path checkpoints/qwen_edit_lora_flatlay/epoch-4.safetensors
```

### 4.2 LongCat-Image-Edit-Turbo

```bash
python scripts/infer.py \
  --model longcat \
  --input-image path/to/look.jpg \
  --output outputs/longcat/look_0001.png \
  --device cuda \
  --dtype bf16 \
  --steps 8 \
  --guidance-scale 1.0
```

With LoRA:

```bash
python scripts/infer.py \
  --model longcat \
  --input-image path/to/look.jpg \
  --output outputs/longcat_lora/look_0001.png \
  --lora-path checkpoints/longcat_edit_lora_flatlay/checkpoints-1000
```

### 4.3 FLUX Kontext

```bash
python scripts/infer.py \
  --model flux_kontext \
  --input-image path/to/look.jpg \
  --output outputs/flux/look_0001.png \
  --device cuda \
  --dtype bf16 \
  --steps 28 \
  --guidance-scale 3.5
```

### 4.4 Batch Inference

From raw data directory:

```bash
python scripts/infer.py \
  --model flux_kontext \
  --data-root data \
  --split test \
  --output-dir output/flux_test
```

From manifest:

```bash
python scripts/infer.py \
  --model flux_kontext \
  --manifest data/manifest/train_enriched.jsonl \
  --output-dir outputs/flux_batch
```

## 5. LoRA Training

### 5.1 FLUX Kontext

```bash
python scripts/train_flux_kontext.py \
  --data-root data \
  --split train \
  --output-dir checkpoints/flux_kontext_flatlay \
  --accelerate-config configs/accelerate_single_gpu.yaml \
  --gradient-checkpointing \
  --cache-latents \
  --use-8bit-adam
```

### 5.2 LongCat-Image Edit LoRA

```bash
python scripts/train_longcat_edit.py \
  --data-root data \
  --split train \
  --output-dir checkpoints/longcat_edit_lora_flatlay \
  --longcat-root external/LongCat-Image \
  --accelerate-config configs/accelerate_zero2_longcat.yaml
```

### 5.3 Qwen-Image-Edit LoRA

```bash
python scripts/train_qwen_edit.py \
  --data-root data \
  --split train \
  --output-dir checkpoints/qwen_edit_lora_flatlay \
  --diffsynth-root external/DiffSynth-Studio \
  --accelerate-config configs/accelerate_zero2offload_qwen_diffsynth.yaml \
  --use-gradient-checkpointing \
  --find-unused-parameters
```

For `Qwen/Qwen-Image-Edit-2511`:

```bash
python scripts/train_qwen_edit.py \
  --data-root data \
  --split train \
  --output-dir checkpoints/qwen_edit_2511_lora_flatlay \
  --pretrained-model-name-or-path Qwen/Qwen-Image-Edit-2511 \
  --zero-cond-t
```

## 6. Evaluation

```bash
python scripts/eval_outputs.py \
  --outputs-root output/flux_test \
  --data-root data \
  --split test \
  --output-dir eval_res/flux
```

Metrics include retrieval accuracy (ResNet50), paired feature cosine similarity, and distribution metrics (FID/KID).

## 7. References

- Qwen-Image-Edit: <https://huggingface.co/Qwen/Qwen-Image-Edit>
- LongCat-Image: <https://github.com/meituan-longcat/LongCat-Image>
- LongCat-Image-Edit-Turbo: <https://huggingface.co/meituan-longcat/LongCat-Image-Edit-Turbo>
- DiffSynth-Studio: <https://github.com/modelscope/DiffSynth-Studio>
- FLUX.1-Kontext-dev: <https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev>
- diffusers Kontext trainer: <https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_flux_kontext.py>

## 8. Known Limitations

- Qwen-Edit training/inference uses `DiffSynth-Studio` as backend, not `diffusers`.
- DiffSynth LoRA checkpoints are not directly compatible with diffusers LoRA loading.
- `FLUX.1-Kontext-dev` and `Qwen-Image-Edit` are memory-intensive — at least 48GB GPU is recommended for high-resolution training.
