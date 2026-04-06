# Task3 Benchmark

## 1. Task

This folder contains the benchmark code for **Task3**, an outfit-level multimodal reasoning task on the FashionStylist dataset. Given an ordered set of item images from one outfit, the model predicts a structured JSON output with:

- `outfit_summary`
- `outfit_style`
- `season`
- `occasion`
- `need_to_modify`
- `mod_index`

The current release supports three evaluation settings:

- local open-weight multimodal large language model evaluation with vLLM
- Gemini evaluation through the API
- supervised fine-tuning with Unsloth LoRA

## 2. Structure

```text
Task3/
├── task3_dataset.py                dataset loader; builds segment-aware train/val/test splits
├── mllm_eval.py                    zero-shot evaluation for local open-weight MLLMs with vLLM
├── gemini_eval.py                  zero-shot Gemini evaluation on the full test split
├── sft_unsloth.py                  LoRA fine-tuning, validation checkpoint selection, and final test evaluation
├── download.py                     local model download utility
├── requirements.mllm_eval.txt      Python dependencies for mllm_eval.py
├── requirements.sft_unsloth.txt    Python dependencies for sft_unsloth.py
├── template/
│   ├── prompt.txt                  shared Task3 prompt template
│   └── api_clients_local.py        local Gemini API key configuration
└── SFT_results/                    saved adapters and evaluation outputs
```

The dataset root passed through `--root` should match the current FashionStylist layout:

```text
<dataset_root>/
├── Female/
│   ├── look(b1-500).csv
│   └── label(p1-2406).csv
├── Male/
│   ├── look(b1-300).csv
│   └── label(p1-1390).csv
└── Child/
    ├── look(b1-200).csv
    └── label(p1-841).csv
```

This benchmark uses the original dataset labels for `season` and `occasion`. When running evaluation or SFT, each group folder must also provide the corresponding item image directory used by the dataset loader. The split protocol is applied separately within `Female`, `Male`, and `Child`, with `train`/`val`/`test` ratios of `70/10/20`. The split seed is controlled by `--split-seed` and defaults to `42`.

## 3. Install

For local open-weight evaluation:

```bash
pip install -r requirements.mllm_eval.txt
```

For supervised fine-tuning:

```bash
pip install -r requirements.sft_unsloth.txt
```

`download.py` additionally requires `huggingface_hub`.

## 4. Prepare Resources

Download all supported local checkpoints:

```bash
python download.py
```

Download selected checkpoints:

```bash
python download.py --model qwen25vl-7b --model qwen3vl-8b
```

Supported aliases:

- `qwen25vl-7b`
- `qwen3vl-4b`
- `qwen3vl-4b-thinking`
- `qwen3vl-8b`
- `qwen3vl-8b-thinking`
- `unsloth-gemma3-4b`

Dataset check:

```bash
python task3_dataset.py --root /path/to/FashionStylist
```

## 5. Run

Local open-weight MLLMs:

```bash
python mllm_eval.py \
  --model qwen25vl-7b \
  --root /path/to/FashionStylist \
  --split test \
  --sample-mode both \
  --thinking-mode hidden
```

Common options:

- `--model`: selects the local model alias, local checkpoint path, or Hugging Face repository id.
- `--batch-size`: controls how many outfit samples are sent to one vLLM generation call.
- `--max-tokens`: sets the maximum number of output tokens per sample.
- `--temperature`: controls sampling randomness during generation.
- `--top-k`: limits token sampling to the top-k candidates at each decoding step.
- `--thinking-mode {hidden,visible}`: chooses whether reasoning stays hidden or is emitted before the final JSON.
- `--allow-hf-download`: allows automatic download from Hugging Face if the local model directory is missing.

Gemini:

Fill in `GEMINI_API_KEY` in `template/api_clients_local.py`, then run:

```bash
python gemini_eval.py \
  --api-model gemini-3.1-pro-preview \
  --root /path/to/FashionStylist \
  --sample-mode both \
  --thinking-mode hidden
```

Gemini evaluation runs on the full `test` split. The script also supports manifest-based replay.

Supervised fine-tuning:

```bash
python sft_unsloth.py \
  --model qwen25vl-7b \
  --root /path/to/FashionStylist \
  --train-split train \
  --selection-split val \
  --test-split test \
  --sample-mode both
```

The SFT pipeline follows a three-step protocol:

1. train on `train`
2. select the best checkpoint on `val`
3. evaluate the selected checkpoint on `test`

The default checkpoint selection metric is:

- `mod_index_accuracy_on_modified_only`

- `mllm_eval.py` writes `results/<model>_<setting>_<split>_results.json` and the corresponding `*_eval.json`
- `gemini_eval.py` writes prediction JSON, request JSONL, and optional sample manifests
- `sft_unsloth.py` writes adapters and evaluation outputs under `SFT_results/`

- `mllm_eval.py` expects structured outputs and does not recover labels from free-form natural language answers.
- `gemini_eval.py` is Gemini-only.
- `task3_dataset.py`, `mllm_eval.py`, `gemini_eval.py`, and `sft_unsloth.py` share the same prompt template in `template/prompt.txt`.
