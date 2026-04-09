#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.models import Inception_V3_Weights, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


EPS = 1e-8
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _log_progress(message: str) -> None:
    print(f"[eval] {message}", flush=True)


@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    subset: str
    bid: str
    pid: str
    category: str
    target_image: str
    generated_image: str
    output_subpath: str


@dataclass(frozen=True)
class GalleryItem:
    subset: str
    pid: str
    category: str
    target_image: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated garment images directly from outputs/ with retrieval metrics, "
            "paired reference metrics, and distribution-level FID/KID."
        )
    )
    parser.add_argument("--outputs-root", required=True, help="Root directory containing generated outputs.")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Raw dataset root containing subset folders such as Male_1-300 and their photos/label.csv.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["retrieval", "paired", "distribution"],
        default=["retrieval", "paired", "distribution"],
        help="Which metric groups to run.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for summary JSON/CSVs.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda if available, else cpu.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for future use.")
    parser.add_argument(
        "--retrieval-backbone",
        choices=["resnet50"],
        default="resnet50",
        help="Backbone for retrieval embeddings and paired feature cosine.",
    )
    parser.add_argument(
        "--crop-foreground-for-paired",
        action="store_true",
        help="Crop both generated/reference images to their non-white foreground bounding boxes before paired metrics.",
    )
    parser.add_argument(
        "--foreground-threshold",
        type=int,
        default=250,
        help="Pixel threshold for detecting non-white foreground in RGB images.",
    )
    parser.add_argument("--kid-subset-size", type=int, default=100, help="Subset size used by KID.")
    parser.add_argument("--kid-subsets", type=int, default=50, help="Number of random KID subsets.")
    parser.add_argument(
        "--distribution-min-samples",
        type=int,
        default=2,
        help="Minimum number of real/fake samples required to compute per-category FID/KID.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of generated output images to evaluate.")
    return parser.parse_args()


def _default_output_dir(outputs_root: Path) -> Path:
    return Path("output") / "eval" / outputs_root.name


def _resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _safe_mean(values: list[float]) -> float | None:
    filtered = [value for value in values if value is not None and not math.isnan(value)]
    if not filtered:
        return None
    return float(sum(filtered) / len(filtered))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            normalized: dict[str, str] = {}
            for key, value in (raw_row or {}).items():
                normalized[(key or "").strip()] = (value or "").strip()
            rows.append(normalized)
    return rows


def _resolve_image_path(photo_dir: Path, image_id: str) -> Path:
    candidate = photo_dir / image_id
    if candidate.exists():
        return candidate.resolve()
    for suffix in IMAGE_SUFFIXES:
        candidate = photo_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    normalized_image_id = image_id.strip().lower()
    for candidate in photo_dir.iterdir():
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if candidate.stem.lower() == normalized_image_id:
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find image '{image_id}' under {photo_dir}")


def _parse_output_dir_name(name: str) -> tuple[str, str]:
    match = re.match(r"^(?P<subset>.+)_(?P<bid>b[^/_]+)$", name)
    if not match:
        raise ValueError(
            f"Could not parse subset/bid from output directory '{name}'. "
            "Expected a name like 'Male_1-300_b255'."
        )
    return match.group("subset"), match.group("bid")


def _parse_output_filename(name: str) -> tuple[str, str]:
    match = re.match(r"^\d+_(?P<category>.+)_(?P<pid>p[^./]+)\.(?:png|jpg|jpeg|webp|bmp)$", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not parse category/pid from output file '{name}'. "
            "Expected a name like '01_outerwear_p1206.png'."
        )
    return match.group("category").lower(), match.group("pid")


def _load_eval_samples(
    outputs_root: str | Path,
    data_root: str | Path,
    limit: int | None = None,
) -> tuple[list[EvalSample], list[GalleryItem]]:
    samples: list[EvalSample] = []
    outputs_root_path = Path(outputs_root).resolve()
    data_root_path = Path(data_root).resolve()
    _log_progress(f"扫描输出目录: {outputs_root_path}")
    output_paths = sorted(
        path for path in outputs_root_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit is not None:
        output_paths = output_paths[:limit]
    _log_progress(f"找到 {len(output_paths)} 张生成图，开始解析文件名与匹配 GT")
    seen_subsets: set[str] = set()
    for idx, generated_path in enumerate(output_paths, start=1):
        if generated_path.parent == outputs_root_path:
            raise ValueError(
                f"Output file '{generated_path}' must live under a per-outfit directory like "
                f"'{outputs_root_path.name}/Male_1-300_b255/01_outerwear_p1206.png'."
            )
        subset, bid = _parse_output_dir_name(generated_path.parent.name)
        category, pid = _parse_output_filename(generated_path.name)
        target_image = _resolve_image_path(data_root_path / subset / "photos", pid)
        seen_subsets.add(subset)
        samples.append(
            EvalSample(
                sample_id=f"{subset}_{bid}_{pid}",
                subset=subset,
                bid=bid,
                pid=pid,
                category=category,
                target_image=str(target_image),
                generated_image=str(generated_path.resolve()),
                output_subpath=str(generated_path.relative_to(outputs_root_path)),
            )
        )
        if idx == len(output_paths) or idx % 100 == 0:
            _log_progress(f"已解析样本 {idx}/{len(output_paths)}")
    gallery = _build_gallery_items(data_root_path, sorted(seen_subsets))
    return samples, gallery


def _build_gallery_items(data_root: Path, subsets: list[str]) -> list[GalleryItem]:
    gallery: list[GalleryItem] = []
    _log_progress(f"开始构建检索库，涉及 {len(subsets)} 个 subset")
    for subset in subsets:
        subset_dir = data_root / subset
        label_path = subset_dir / "label.csv"
        photo_dir = subset_dir / "photos"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label.csv for subset '{subset}': {label_path}")
        if not photo_dir.exists():
            raise FileNotFoundError(f"Missing photos directory for subset '{subset}': {photo_dir}")
        subset_start = len(gallery)
        for row in _read_csv_rows(label_path):
            pid = row.get("photo", "").strip()
            category = row.get("category", "").strip().lower()
            if not pid or not category:
                continue
            gallery.append(
                GalleryItem(
                    subset=subset,
                    pid=pid,
                    category=category,
                    target_image=str(_resolve_image_path(photo_dir, pid)),
                )
            )
        _log_progress(f"subset {subset}: 新增 {len(gallery) - subset_start} 个 gallery 项")
    if not gallery:
        raise ValueError(f"No gallery items found under {data_root} for subsets: {', '.join(subsets)}")
    _log_progress(f"检索库构建完成，共 {len(gallery)} 个 gallery 项")
    return gallery


def load_rgb_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def crop_foreground(image: Image.Image, threshold: int = 250) -> Image.Image:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        return image
    mask = np.any(array < threshold, axis=2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0) + 1
    return image.crop((left, top, right, bottom))


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def resize_tensor_image(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return float("inf")
    return float(20.0 * math.log10(1.0 / math.sqrt(mse + EPS)))


def _gaussian_window(window_size: int, sigma: float, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    channel = pred.shape[0]
    device = pred.device
    dtype = pred.dtype
    window = _gaussian_window(window_size, sigma, channel, device, dtype)
    padding = window_size // 2
    pred_batch = pred.unsqueeze(0)
    target_batch = target.unsqueeze(0)
    mu_x = F.conv2d(pred_batch, window, padding=padding, groups=channel)
    mu_y = F.conv2d(target_batch, window, padding=padding, groups=channel)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(pred_batch * pred_batch, window, padding=padding, groups=channel) - mu_x_sq
    sigma_y_sq = F.conv2d(target_batch * target_batch, window, padding=padding, groups=channel) - mu_y_sq
    sigma_xy = F.conv2d(pred_batch * target_batch, window, padding=padding, groups=channel) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + EPS)
    return float(ssim_map.mean().item())


class RetrievalEncoder:
    def __init__(self, backbone: str, device: torch.device):
        if backbone != "resnet50":
            raise ValueError(f"Unsupported retrieval backbone: {backbone}")
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.transform = weights.transforms()
        self.device = device

    @torch.inference_mode()
    def encode_paths(self, paths: list[str], batch_size: int) -> np.ndarray:
        batches: list[np.ndarray] = []
        total_batches = math.ceil(len(paths) / batch_size) if paths else 0
        _log_progress(f"提取 ResNet50 特征: 共 {len(paths)} 张图，{total_batches} 个 batch")
        for batch_idx, start in enumerate(range(0, len(paths), batch_size), start=1):
            batch_paths = paths[start : start + batch_size]
            batch_images = [self.transform(load_rgb_image(path)) for path in batch_paths]
            batch = torch.stack(batch_images).to(self.device)
            features = self.model(batch)
            features = F.normalize(features, dim=1)
            batches.append(features.cpu().numpy().astype(np.float32))
            _log_progress(f"ResNet50 特征提取进度: batch {batch_idx}/{total_batches}")
        return np.concatenate(batches, axis=0) if batches else np.zeros((0, 2048), dtype=np.float32)


class InceptionEncoder:
    def __init__(self, device: torch.device):
        weights = Inception_V3_Weights.DEFAULT
        # torchvision 在加载预训练权重时要求 aux_logits=True；eval 模式下不会使用 aux 输出。
        model = models.inception_v3(weights=weights, aux_logits=True)
        model.eval()
        self.model = create_feature_extractor(model, return_nodes={"avgpool": "features"}).to(device)
        self.transform = weights.transforms()
        self.device = device

    @torch.inference_mode()
    def encode_paths(self, paths: list[str], batch_size: int) -> np.ndarray:
        batches: list[np.ndarray] = []
        total_batches = math.ceil(len(paths) / batch_size) if paths else 0
        _log_progress(f"提取 InceptionV3 特征: 共 {len(paths)} 张图，{total_batches} 个 batch")
        for batch_idx, start in enumerate(range(0, len(paths), batch_size), start=1):
            batch_paths = paths[start : start + batch_size]
            batch_images = [self.transform(load_rgb_image(path)) for path in batch_paths]
            batch = torch.stack(batch_images).to(self.device)
            features = self.model(batch)["features"].flatten(1)
            batches.append(features.cpu().numpy().astype(np.float64))
            _log_progress(f"InceptionV3 特征提取进度: batch {batch_idx}/{total_batches}")
        return np.concatenate(batches, axis=0) if batches else np.zeros((0, 2048), dtype=np.float64)


def _group_indices_by_category(items: list[Any]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        grouped[str(item.category)].append(idx)
    return grouped


def compute_retrieval_metrics(
    samples: list[EvalSample],
    gallery: list[GalleryItem],
    query_features: np.ndarray,
    gallery_features: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    category_to_indices = _group_indices_by_category(gallery)
    per_sample_rows: list[dict[str, Any]] = []
    ranks: list[float] = []
    recalls = {1: [], 5: [], 10: []}
    ndcgs = {5: [], 10: []}
    mrrs: list[float] = []
    _log_progress(f"开始计算 retrieval 指标，共 {len(samples)} 个 query")
    for idx, sample in enumerate(samples):
        gallery_idx = category_to_indices[sample.category]
        if not gallery_idx:
            continue
        scores = gallery_features[gallery_idx] @ query_features[idx]
        ranking = np.argsort(-scores)
        ranked_indices = [gallery_idx[position] for position in ranking]
        positive_positions = [
            rank
            for rank, candidate_idx in enumerate(ranked_indices, start=1)
            if gallery[candidate_idx].pid == sample.pid and gallery[candidate_idx].subset == sample.subset
        ]
        rank = float(positive_positions[0]) if positive_positions else float("inf")
        ranks.append(rank)
        per_sample = {
            "sample_id": sample.sample_id,
            "pid": sample.pid,
            "category": sample.category,
            "retrieval_rank": None if math.isinf(rank) else int(rank),
            "retrieval_gallery_size": len(gallery_idx),
        }
        for k in recalls:
            hit = 0.0 if math.isinf(rank) else float(rank <= k)
            recalls[k].append(hit)
            per_sample[f"recall_at_{k}"] = hit
        reciprocal_rank = 0.0 if math.isinf(rank) else 1.0 / rank
        mrrs.append(reciprocal_rank)
        per_sample["mrr"] = reciprocal_rank
        for k in ndcgs:
            if math.isinf(rank) or rank > k:
                ndcg = 0.0
            else:
                ndcg = 1.0 / math.log2(rank + 1.0)
            ndcgs[k].append(ndcg)
            per_sample[f"ndcg_at_{k}"] = ndcg
        per_sample_rows.append(per_sample)
        if (idx + 1) == len(samples) or (idx + 1) % 100 == 0:
            _log_progress(f"retrieval 进度: {idx + 1}/{len(samples)}")
    summary = {
        "num_queries": len(per_sample_rows),
        "mean_rank": _safe_mean([rank for rank in ranks if not math.isinf(rank)]),
        "mrr": _safe_mean(mrrs),
        "recall_at_1": _safe_mean(recalls[1]),
        "recall_at_5": _safe_mean(recalls[5]),
        "recall_at_10": _safe_mean(recalls[10]),
        "ndcg_at_5": _safe_mean(ndcgs[5]),
        "ndcg_at_10": _safe_mean(ndcgs[10]),
    }
    return summary, per_sample_rows


def compute_paired_metrics(
    samples: list[EvalSample],
    query_features: np.ndarray,
    reference_features: np.ndarray,
    crop_paired: bool,
    foreground_threshold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_sample_rows: list[dict[str, Any]] = []
    psnrs: list[float] = []
    ssims: list[float] = []
    cosines: list[float] = []
    _log_progress(f"开始计算 paired 指标，共 {len(samples)} 对图像")
    for idx, sample in enumerate(samples):
        generated_image = load_rgb_image(sample.generated_image)
        reference_image = load_rgb_image(sample.target_image)
        if crop_paired:
            generated_image = crop_foreground(generated_image, threshold=foreground_threshold)
            reference_image = crop_foreground(reference_image, threshold=foreground_threshold)
        target_size = (reference_image.height, reference_image.width)
        generated_tensor = image_to_tensor(generated_image)
        reference_tensor = image_to_tensor(reference_image)
        if generated_tensor.shape[1:] != target_size:
            generated_tensor = resize_tensor_image(generated_tensor, target_size)
        generated_tensor = generated_tensor.clamp(0.0, 1.0)
        reference_tensor = reference_tensor.clamp(0.0, 1.0)
        psnr_value = compute_psnr(generated_tensor, reference_tensor)
        ssim_value = compute_ssim(generated_tensor, reference_tensor)
        feature_cosine = float(np.dot(query_features[idx], reference_features[idx]))
        row = {
            "sample_id": sample.sample_id,
            "pid": sample.pid,
            "category": sample.category,
            "psnr": psnr_value,
            "ssim": ssim_value,
            "feature_cosine": feature_cosine,
        }
        psnrs.append(psnr_value)
        ssims.append(ssim_value)
        cosines.append(feature_cosine)
        per_sample_rows.append(row)
        if (idx + 1) == len(samples) or (idx + 1) % 100 == 0:
            _log_progress(f"paired 进度: {idx + 1}/{len(samples)}")
    summary = {
        "num_pairs": len(per_sample_rows),
        "psnr": _safe_mean(psnrs),
        "ssim": _safe_mean(ssims),
        "feature_cosine": _safe_mean(cosines),
    }
    return summary, per_sample_rows


def _covariance(features: np.ndarray) -> np.ndarray:
    if features.shape[0] <= 1:
        return np.eye(features.shape[1], dtype=np.float64) * EPS
    return np.cov(features, rowvar=False)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = _covariance(real_features)
    sigma_fake = _covariance(fake_features)
    diff = mu_real - mu_fake
    eigvals_real, eigvecs_real = np.linalg.eigh((sigma_real + sigma_real.T) / 2.0)
    eigvals_real = np.clip(eigvals_real, a_min=0.0, a_max=None)
    sqrt_sigma_real = (eigvecs_real * np.sqrt(eigvals_real)) @ eigvecs_real.T
    middle = sqrt_sigma_real @ sigma_fake @ sqrt_sigma_real
    middle = (middle + middle.T) / 2.0
    eigvals_middle = np.linalg.eigh(middle)[0]
    trace_sqrt = float(np.sum(np.sqrt(np.clip(eigvals_middle, a_min=0.0, a_max=None))))
    fid = float(diff @ diff + np.trace(sigma_real) + np.trace(sigma_fake) - 2.0 * trace_sqrt)
    return max(fid, 0.0)


def _polynomial_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dim = x.shape[1]
    return ((x @ y.T) / dim + 1.0) ** 3


def compute_kid(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    subset_size: int,
    subsets: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    real_count = real_features.shape[0]
    fake_count = fake_features.shape[0]
    actual_subset_size = min(subset_size, real_count, fake_count)
    if actual_subset_size < 2:
        return float("nan"), float("nan")
    estimates: list[float] = []
    for _ in range(subsets):
        real_idx = rng.choice(real_count, size=actual_subset_size, replace=False)
        fake_idx = rng.choice(fake_count, size=actual_subset_size, replace=False)
        x = real_features[real_idx]
        y = fake_features[fake_idx]
        k_xx = _polynomial_kernel(x, x)
        k_yy = _polynomial_kernel(y, y)
        k_xy = _polynomial_kernel(x, y)
        sum_xx = (np.sum(k_xx) - np.trace(k_xx)) / (actual_subset_size * (actual_subset_size - 1))
        sum_yy = (np.sum(k_yy) - np.trace(k_yy)) / (actual_subset_size * (actual_subset_size - 1))
        sum_xy = np.mean(k_xy)
        estimates.append(float(sum_xx + sum_yy - 2.0 * sum_xy))
    return float(np.mean(estimates)), float(np.std(estimates, ddof=1) if len(estimates) > 1 else 0.0)


def compute_distribution_metrics(
    samples: list[EvalSample],
    real_features: np.ndarray,
    fake_features: np.ndarray,
    kid_subset_size: int,
    kid_subsets: int,
    distribution_min_samples: int,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overall": None,
        "per_category": {},
    }
    if len(samples) < distribution_min_samples:
        return summary
    _log_progress("开始计算 distribution 指标 (FID/KID)")
    overall_kid_mean, overall_kid_std = compute_kid(real_features, fake_features, kid_subset_size, kid_subsets, seed)
    summary["overall"] = {
        "num_samples": len(samples),
        "fid": compute_fid(real_features, fake_features),
        "kid_mean": overall_kid_mean,
        "kid_std": overall_kid_std,
    }
    category_to_indices = _group_indices_by_category(samples)
    for category, indices in sorted(category_to_indices.items()):
        if len(indices) < distribution_min_samples:
            continue
        real_subset = real_features[indices]
        fake_subset = fake_features[indices]
        kid_mean, kid_std = compute_kid(real_subset, fake_subset, kid_subset_size, kid_subsets, seed)
        summary["per_category"][category] = {
            "num_samples": len(indices),
            "fid": compute_fid(real_subset, fake_subset),
            "kid_mean": kid_mean,
            "kid_std": kid_std,
        }
        _log_progress(f"distribution 类别进度: {category} ({len(indices)} 张)")
    return summary


def _merge_per_sample_rows(
    samples: list[EvalSample],
    retrieval_rows: list[dict[str, Any]] | None,
    paired_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {
        sample.sample_id: {
            "sample_id": sample.sample_id,
            "subset": sample.subset,
            "bid": sample.bid,
            "pid": sample.pid,
            "category": sample.category,
            "generated_image": sample.generated_image,
            "target_image": sample.target_image,
            "output_subpath": sample.output_subpath,
        }
        for sample in samples
    }
    for row in retrieval_rows or []:
        merged[row["sample_id"]].update(row)
    for row in paired_rows or []:
        merged[row["sample_id"]].update(row)
    return [merged[sample.sample_id] for sample in samples]


def _aggregate_per_category(per_sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample_rows:
        grouped[str(row["category"])].append(row)
    aggregated: list[dict[str, Any]] = []
    for category, rows in sorted(grouped.items()):
        aggregated.append(
            {
                "category": category,
                "num_samples": len(rows),
                "recall_at_1": _safe_mean([float(row["recall_at_1"]) for row in rows if "recall_at_1" in row]),
                "recall_at_5": _safe_mean([float(row["recall_at_5"]) for row in rows if "recall_at_5" in row]),
                "recall_at_10": _safe_mean([float(row["recall_at_10"]) for row in rows if "recall_at_10" in row]),
                "ndcg_at_5": _safe_mean([float(row["ndcg_at_5"]) for row in rows if "ndcg_at_5" in row]),
                "ndcg_at_10": _safe_mean([float(row["ndcg_at_10"]) for row in rows if "ndcg_at_10" in row]),
                "mrr": _safe_mean([float(row["mrr"]) for row in rows if "mrr" in row]),
                "psnr": _safe_mean([float(row["psnr"]) for row in rows if "psnr" in row]),
                "ssim": _safe_mean([float(row["ssim"]) for row in rows if "ssim" in row]),
                "feature_cosine": _safe_mean([float(row["feature_cosine"]) for row in rows if "feature_cosine" in row]),
            }
        )
    return aggregated


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    outputs_root = Path(args.outputs_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (_default_output_dir(outputs_root)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_progress(f"输出目录: {outputs_root}")
    _log_progress(f"数据目录: {Path(args.data_root).resolve()}")
    _log_progress(f"评估结果将写入: {output_dir}")

    samples, gallery_items = _load_eval_samples(outputs_root, args.data_root, limit=args.limit)
    if not samples:
        raise ValueError(
            f"No generated images found under {outputs_root}. "
            "Check --outputs-root or whether the outputs were fully generated."
        )
    _log_progress(f"样本加载完成: {len(samples)} 个 query, {len(gallery_items)} 个 gallery 项")

    retrieval_encoder = None
    query_features = None
    gallery_features = None
    reference_features = None
    if "retrieval" in args.metrics or "paired" in args.metrics:
        _log_progress(f"准备提取 retrieval/paired 特征，backbone={args.retrieval_backbone}")
        retrieval_encoder = RetrievalEncoder(args.retrieval_backbone, device)
        query_features = retrieval_encoder.encode_paths([sample.generated_image for sample in samples], args.batch_size)
        gallery_features = retrieval_encoder.encode_paths([item.target_image for item in gallery_items], args.batch_size)
        reference_features = retrieval_encoder.encode_paths([sample.target_image for sample in samples], args.batch_size)

    summary: dict[str, Any] = {
        "outputs_root": str(outputs_root),
        "data_root": str(Path(args.data_root).resolve()),
        "num_samples": len(samples),
        "num_gallery_items": len(gallery_items),
        "device": str(device),
        "metrics": {},
    }

    retrieval_rows = None
    paired_rows = None

    if "retrieval" in args.metrics:
        assert query_features is not None
        assert gallery_features is not None
        retrieval_summary, retrieval_rows = compute_retrieval_metrics(samples, gallery_items, query_features, gallery_features)
        summary["metrics"]["retrieval"] = retrieval_summary
        _log_progress("retrieval 指标计算完成")

    if "paired" in args.metrics:
        assert query_features is not None
        assert reference_features is not None
        paired_summary, paired_rows = compute_paired_metrics(
            samples=samples,
            query_features=query_features,
            reference_features=reference_features,
            crop_paired=args.crop_foreground_for_paired,
            foreground_threshold=args.foreground_threshold,
        )
        summary["metrics"]["paired"] = paired_summary
        _log_progress("paired 指标计算完成")

    if "distribution" in args.metrics:
        _log_progress("准备提取 distribution 特征")
        inception_encoder = InceptionEncoder(device)
        real_features = inception_encoder.encode_paths([sample.target_image for sample in samples], args.batch_size)
        fake_features = inception_encoder.encode_paths([sample.generated_image for sample in samples], args.batch_size)
        distribution_summary = compute_distribution_metrics(
            samples=samples,
            real_features=real_features,
            fake_features=fake_features,
            kid_subset_size=args.kid_subset_size,
            kid_subsets=args.kid_subsets,
            distribution_min_samples=args.distribution_min_samples,
            seed=args.seed,
        )
        summary["metrics"]["distribution"] = distribution_summary
        _log_progress("distribution 指标计算完成")

    per_sample_rows = _merge_per_sample_rows(samples, retrieval_rows, paired_rows)
    per_category_rows = _aggregate_per_category(per_sample_rows)
    distribution_per_category = summary["metrics"].get("distribution", {}).get("per_category", {})
    for row in per_category_rows:
        category_metrics = distribution_per_category.get(row["category"])
        if category_metrics:
            row["fid"] = category_metrics["fid"]
            row["kid_mean"] = category_metrics["kid_mean"]
            row["kid_std"] = category_metrics["kid_std"]
    _write_csv(output_dir / "per_sample.csv", per_sample_rows)
    _write_csv(output_dir / "per_category.csv", per_category_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")

    _log_progress("所有结果已写盘")
    print(f"samples={len(samples)}")
    print(f"gallery_items={len(gallery_items)}")
    print(f"summary_json={output_dir / 'summary.json'}")
    print(f"per_sample_csv={output_dir / 'per_sample.csv'}")
    print(f"per_category_csv={output_dir / 'per_category.csv'}")


if __name__ == "__main__":
    main()
