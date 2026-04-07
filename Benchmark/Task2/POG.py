"""
Task2 Fashion Outfit Model.

This script turns the provided split CSVs into a real multimodal outfit
training pipeline:
1. Align `fb/mb/kb` and `fp/mp/kp` ids from `data/*.csv` with `sourceData/*`.
2. Read English labels from `label_en.csv` and `look_en.csv`.
3. Build two text settings for each item:
   - `title`
   - `title_attrs` (curated natural-language description from high-value attributes)
4. Extract image/text features with FashionCLIP and cache them.
5. Train and evaluate the outfit model with FITB metrics (global retrieval).
"""

import argparse
import json
import random
import re
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SOURCE_SPECS = {
    "female1-500": {"dir_name": "Female_1-500", "prefix": "f"},
    "male1-300": {"dir_name": "Male_1-300", "prefix": "m"},
    "kid1-200": {"dir_name": "Child_1-200", "prefix": "k"},
}

TEXT_MODE_CHOICES = ("title", "title_attrs")
# High-value attribute fields for text construction (ordered for readability).
# Dropped low-signal fields: outline, detail, donningdoffing.
TEXT_ATTRIBUTE_FIELDS = (
    ("category", "category"),
    ("gender_en", "gender"),
    ("color_en", "color"),
    ("materials_en", "materials"),
    ("pattern_en", "pattern"),
    ("style_en", "style"),
)
SELECTION_METRIC = "Recall@10"


@dataclass
class FOMConfig:
    data_root: str = "data"
    source_data_root: str = "sourceData"
    feature_cache_dir: str = "cache/features"
    output_dir: str = "outputs"
    fashionclip_model: str = "patrickjohncyh/fashion-clip"
    text_mode: str = "title_attrs"

    clip_dim: int = 512
    fusion_mode: str = "concat"
    embed_dim: int = 256
    hidden_dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.15
    ff_dim: int = 256

    n_neg_samples: int = 6
    max_outfit_size: int = 10
    lr: float = 2e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    batch_size: int = 32
    feature_batch_size: int = 64
    text_max_length: int = 77
    eval_every: int = 1
    seed: int = 42
    limit_outfits: int = 0
    force_reextract: bool = False
    prepare_only: bool = False


@dataclass
class ItemRecord:
    item_id: str
    source_group: str
    local_photo_id: str
    image_path: str
    title_en: str
    attrs_en: Dict[str, str]


@dataclass
class BundleRecord:
    bundle_id: str
    source_group: str
    local_bundle_id: str
    image_path: str
    item_ids: List[str]
    look_en: str
    season_en: str
    occasion_en: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FOM on Task2 data.")
    parser.add_argument(
        "--text-modes",
        nargs="+",
        choices=TEXT_MODE_CHOICES,
        default=list(TEXT_MODE_CHOICES),
        help="Run one or both text settings.",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--source-data-root", default="sourceData")
    parser.add_argument("--feature-cache-dir", default="cache/features")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--fashionclip-model", default="patrickjohncyh/fashion-clip")
    parser.add_argument(
        "--fusion-mode",
        choices=("concat", "mean", "image_only", "text_only"),
        default="concat",
    )
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--n-neg-samples", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument(
        "--text-max-length",
        type=int,
        default=77,
        help="Maximum text token length for CLIP text encoder.",
    )
    parser.add_argument("--max-outfit-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--limit-outfits",
        type=int,
        default=0,
        help="Debug option. 0 means use the full split.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only align data and extract/cache multimodal item features.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_source_list(raw: str) -> List[str]:
    return [part.strip() for part in normalize_text(raw).split("|") if part.strip()]


def parse_photo_list(raw: str) -> List[str]:
    return [match.lower() for match in re.findall(r"[pP]\d+", normalize_text(raw))]


def to_dataset_style_id(source_group: str, local_id: str) -> str:
    if "::" in local_id:
        return local_id.strip()

    spec = SOURCE_SPECS[source_group]
    local_id = local_id.strip().lower()
    prefix = spec["prefix"]
    if local_id.startswith(prefix):
        return f"{source_group}::{local_id}"
    if local_id.startswith(("p", "b")):
        return f"{source_group}::{prefix}{local_id}"
    raise ValueError(f"Unsupported local id: {source_group=} {local_id=}")


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


class SourceDataRegistry:
    def __init__(self, source_root: str):
        self.source_root = Path(source_root)
        self.items: Dict[str, ItemRecord] = {}
        self.bundles: Dict[str, BundleRecord] = {}
        self._load()

    def _load(self) -> None:
        for source_group, spec in SOURCE_SPECS.items():
            group_root = self.source_root / spec["dir_name"]
            label_df = pd.read_csv(group_root / "label_en.csv", dtype=str).fillna("")
            look_df = pd.read_csv(group_root / "look_en.csv", dtype=str).fillna("")

            for row in label_df.to_dict("records"):
                local_photo_id = normalize_text(row["photo"]).lower()
                item_id = to_dataset_style_id(source_group, local_photo_id)
                attrs = {}
                for field_name, alias in TEXT_ATTRIBUTE_FIELDS:
                    value = normalize_text(row.get(field_name, ""))
                    if value:
                        attrs[alias] = value

                title_en = normalize_text(row.get("title_en", "")) or normalize_text(
                    row.get("title", "")
                )

                self.items[item_id] = ItemRecord(
                    item_id=item_id,
                    source_group=source_group,
                    local_photo_id=local_photo_id,
                    image_path=str(self._resolve_image_path(group_root, local_photo_id)),
                    title_en=title_en,
                    attrs_en=attrs,
                )

            for row in look_df.to_dict("records"):
                local_bundle_id = normalize_text(row["bandle"]).lower()
                bundle_id = to_dataset_style_id(source_group, local_bundle_id)
                local_photo_ids = parse_photo_list(row["photos"])
                item_ids = [
                    to_dataset_style_id(source_group, photo_id)
                    for photo_id in local_photo_ids
                ]

                self.bundles[bundle_id] = BundleRecord(
                    bundle_id=bundle_id,
                    source_group=source_group,
                    local_bundle_id=local_bundle_id,
                    image_path=str(self._resolve_image_path(group_root, local_bundle_id)),
                    item_ids=item_ids,
                    look_en=normalize_text(row.get("look_en", "")),
                    season_en=normalize_text(row.get("season_en", "")),
                    occasion_en=normalize_text(row.get("occasion_en", "")),
                )

    @staticmethod
    def _resolve_image_path(group_root: Path, local_id: str) -> Path:
        photo_dir = group_root / "photos"
        candidates = [
            photo_dir / f"{local_id}.png",
            photo_dir / f"{local_id[0].upper()}{local_id[1:]}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Cannot find image for {group_root.name}:{local_id}")

    def build_outfits_from_split(
        self,
        split_csv: str,
        limit_outfits: int = 0,
    ) -> Tuple[List[List[str]], Dict]:
        df = pd.read_csv(split_csv, dtype=str).fillna("")
        if limit_outfits > 0:
            df = df.iloc[:limit_outfits].copy()

        outfits: List[List[str]] = []
        order_mismatch_count = 0
        order_mismatch_examples = []

        for row in df.to_dict("records"):
            source_group = normalize_text(row["source_group"])
            bundle_source_id = normalize_text(row["bundle_source_id"])
            bundle_id = (
                bundle_source_id
                if bundle_source_id
                else to_dataset_style_id(source_group, row["outfit_id"])
            )

            if bundle_id not in self.bundles:
                raise KeyError(f"Missing bundle metadata for {bundle_id}")

            split_item_ids = parse_source_list(row["item_source_ids"])
            if not split_item_ids:
                split_item_ids = [
                    to_dataset_style_id(source_group, photo_id)
                    for photo_id in parse_source_list(row["item_photo_ids"])
                ]

            bundle_item_ids = self.bundles[bundle_id].item_ids
            if split_item_ids != bundle_item_ids:
                if sorted(split_item_ids) != sorted(bundle_item_ids):
                    raise ValueError(
                        f"Bundle alignment failed for {bundle_id}: "
                        f"{split_item_ids} vs {bundle_item_ids}"
                    )
                order_mismatch_count += 1
                if len(order_mismatch_examples) < 5:
                    order_mismatch_examples.append(
                        {
                            "bundle_id": bundle_id,
                            "split_item_ids": split_item_ids,
                            "look_en_item_ids": bundle_item_ids,
                        }
                    )

            for item_id in split_item_ids:
                if item_id not in self.items:
                    raise KeyError(f"Missing item metadata for {item_id}")

            outfits.append(split_item_ids)

        stats = {
            "split_csv": str(split_csv),
            "num_outfits": len(outfits),
            "unique_items": len({item_id for outfit in outfits for item_id in outfit}),
            "bundle_order_mismatches": order_mismatch_count,
            "bundle_order_mismatch_examples": order_mismatch_examples,
        }
        return outfits, stats

    def get_item_records(self, item_ids: List[str]) -> List[ItemRecord]:
        return [self.items[item_id] for item_id in item_ids]

    def build_alignment_report(self, split_stats: Dict[str, Dict]) -> Dict:
        return {
            "source_groups": {
                group: {
                    "num_items": sum(
                        1 for item in self.items.values() if item.source_group == group
                    ),
                    "num_bundles": sum(
                        1
                        for bundle in self.bundles.values()
                        if bundle.source_group == group
                    ),
                }
                for group in SOURCE_SPECS
            },
            "splits": split_stats,
        }


def build_item_text(item: ItemRecord, text_mode: str) -> str:
    if text_mode == "title":
        return item.title_en
    if text_mode == "title_attrs":
        # Put the title first because it is the cleanest caption-like signal.
        # Append a short attribute summary as supplementary context.
        title = item.title_en.strip()
        attrs = item.attrs_en
        parts = []
        for field in ("color", "materials", "pattern"):
            val = attrs.get(field, "")
            if val:
                parts.append(val)
        cat = attrs.get("category", "")
        if cat:
            parts.append(cat)
        phrase = " ".join(parts)
        qualifiers = []
        for field in ("style", "gender"):
            val = attrs.get(field, "")
            if val:
                qualifiers.append(val)
        if qualifiers:
            phrase = phrase + ", " + ", ".join(qualifiers)
        if title and phrase:
            return f"{title}. {phrase}"
        return title or phrase or "fashion item"
    raise ValueError(f"Unknown text mode: {text_mode}")


class FashionCLIPEncoder(nn.Module):
    def __init__(self, cfg: FOMConfig):
        super().__init__()
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "FashionCLIP feature extraction needs `transformers` installed."
            ) from exc

        self.model = CLIPModel.from_pretrained(cfg.fashionclip_model)
        self.processor = CLIPProcessor.from_pretrained(cfg.fashionclip_model)
        model_limit = getattr(
            getattr(self.model.config, "text_config", None),
            "max_position_embeddings",
            None,
        )
        tokenizer_limit = getattr(
            getattr(self.processor, "tokenizer", None),
            "model_max_length",
            None,
        )
        candidate_limits = [cfg.text_max_length]
        if isinstance(model_limit, int) and model_limit > 0:
            candidate_limits.append(model_limit)
        if (
            isinstance(tokenizer_limit, int)
            and tokenizer_limit > 0
            and tokenizer_limit < 100000
        ):
            candidate_limits.append(tokenizer_limit)
        self.max_text_length = min(candidate_limits) if candidate_limits else 77

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embs = self.model.get_image_features(**inputs)
        return F.normalize(embs, dim=-1)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        embs = self.model.get_text_features(**inputs)
        return F.normalize(embs, dim=-1)


class FeatureExtractor:
    def __init__(self, cfg: FOMConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.encoder = FashionCLIPEncoder(cfg).to(device)
        self.encoder.eval()

    def extract_items(
        self,
        items: List[ItemRecord],
        text_mode: str,
        batch_size: int,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, str]]:
        features = {}
        texts = {}

        for start in range(0, len(items), batch_size):
            batch_items = items[start : start + batch_size]
            images = []
            batch_texts = []
            for item in batch_items:
                with Image.open(item.image_path) as img:
                    images.append(img.convert("RGB"))
                text = build_item_text(item, text_mode)
                batch_texts.append(text)
                texts[item.item_id] = text

            img_embs = self.encoder.encode_images(images, self.device).cpu()
            txt_embs = self.encoder.encode_texts(batch_texts, self.device).cpu()

            for idx, item in enumerate(batch_items):
                features[item.item_id] = {
                    "img_emb": img_embs[idx],
                    "txt_emb": txt_embs[idx],
                }

        return features, texts


class MultiModalFusion(nn.Module):
    def __init__(self, cfg: FOMConfig):
        super().__init__()
        self.mode = cfg.fusion_mode

        if cfg.fusion_mode == "concat":
            in_dim = cfg.clip_dim * 2
        elif cfg.fusion_mode in ("mean", "image_only", "text_only"):
            in_dim = cfg.clip_dim
        else:
            raise ValueError(f"Unknown fusion mode: {cfg.fusion_mode}")

        self.proj = nn.Sequential(
            nn.Linear(in_dim, cfg.embed_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            x = torch.cat([img_emb, txt_emb], dim=-1)
        elif self.mode == "mean":
            x = (img_emb + txt_emb) / 2
        elif self.mode == "image_only":
            x = img_emb
        elif self.mode == "text_only":
            x = txt_emb
        else:
            raise ValueError(f"Unknown fusion mode: {self.mode}")
        return self.proj(x)


class TransitionLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FOM(nn.Module):
    def __init__(self, cfg: FOMConfig):
        super().__init__()
        self.cfg = cfg
        self.fusion = MultiModalFusion(cfg)
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim))
        self.transition = TransitionLayer(cfg.embed_dim, cfg.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
        )

    def fuse_embeddings(
        self,
        img_embs: torch.Tensor,
        txt_embs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_items, _ = img_embs.shape
        fused = self.fusion(
            img_embs.reshape(batch_size * num_items, -1),
            txt_embs.reshape(batch_size * num_items, -1),
        )
        return fused.reshape(batch_size, num_items, -1)

    def forward_encoder(
        self,
        item_embs: torch.Tensor,
        mask_pos: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = item_embs.size(0)
        masked_embs = item_embs.clone()
        gather_idx = mask_pos.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size,
            1,
            item_embs.size(-1),
        )
        masked_embs.scatter_(1, gather_idx, self.mask_token.expand(batch_size, -1, -1))

        hidden = self.transition(masked_embs)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)

        mask_hidden = hidden.gather(
            1,
            mask_pos.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, hidden.size(-1)),
        ).squeeze(1)
        return mask_hidden, hidden

    def _sample_negative(
        self,
        all_trans: torch.Tensor,
        batch_size: int,
        outfit_size: int,
        step_idx: int,
    ) -> torch.Tensor:
        if batch_size > 1:
            shift = random.randint(1, batch_size - 1)
            neg_idx = random.randint(0, outfit_size - 1)
            return all_trans.roll(shift, dims=0)[:, neg_idx, :]

        neg_idx = random.randint(0, outfit_size - 1)
        if outfit_size > 1 and neg_idx == step_idx:
            neg_idx = (neg_idx + 1) % outfit_size
        return all_trans[:, neg_idx, :]

    def compute_loss(
        self,
        img_embs: torch.Tensor,
        txt_embs: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, outfit_size, _ = img_embs.shape
        item_embs = self.fuse_embeddings(img_embs, txt_embs)
        all_trans = self.transition(item_embs)

        total_loss = 0.0
        valid_count = 0

        for step_idx in range(outfit_size):
            mask_pos = torch.full(
                (batch_size,),
                step_idx,
                dtype=torch.long,
                device=img_embs.device,
            )
            if padding_mask is not None:
                valid = ~padding_mask[:, step_idx]
                if valid.sum() == 0:
                    continue
            else:
                valid = torch.ones(batch_size, dtype=torch.bool, device=img_embs.device)

            mask_hidden, _ = self.forward_encoder(item_embs, mask_pos, padding_mask)
            pos_hidden = all_trans[:, step_idx, :]

            neg_hidden = torch.stack(
                [
                    self._sample_negative(all_trans, batch_size, outfit_size, step_idx)
                    for _ in range(self.cfg.n_neg_samples)
                ],
                dim=1,
            )

            pos_logits = (mask_hidden * pos_hidden).sum(-1, keepdim=True)
            neg_logits = torch.bmm(neg_hidden, mask_hidden.unsqueeze(-1)).squeeze(-1)
            logits = torch.cat([pos_logits, neg_logits], dim=-1)

            target = torch.zeros(batch_size, dtype=torch.long, device=img_embs.device)
            loss = F.cross_entropy(logits, target, reduction="none")
            total_loss += (loss * valid.float()).sum()
            valid_count += int(valid.sum().item())

        return total_loss / max(valid_count, 1)

    @torch.no_grad()
    def score_candidates(
        self,
        item_embs: torch.Tensor,
        mask_pos: torch.Tensor,
        cand_embs: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        mask_hidden, _ = self.forward_encoder(item_embs, mask_pos, padding_mask)
        cand_hidden = self.transition(cand_embs)
        return torch.bmm(cand_hidden, mask_hidden.unsqueeze(-1)).squeeze(-1)


class OutfitFeatureDataset(Dataset):
    def __init__(self, outfits: List[List[str]], item_features: Dict, max_size: int = 10):
        self.outfits = outfits
        self.item_features = item_features
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.outfits)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        outfit_ids = self.outfits[idx][: self.max_size]
        num_items = len(outfit_ids)

        img_embs = torch.stack([self.item_features[iid]["img_emb"] for iid in outfit_ids])
        txt_embs = torch.stack([self.item_features[iid]["txt_emb"] for iid in outfit_ids])

        pad_items = self.max_size - num_items
        if pad_items > 0:
            img_embs = F.pad(img_embs, (0, 0, 0, pad_items))
            txt_embs = F.pad(txt_embs, (0, 0, 0, pad_items))

        padding_mask = torch.zeros(self.max_size, dtype=torch.bool)
        padding_mask[num_items:] = True

        return {
            "img_embs": img_embs,
            "txt_embs": txt_embs,
            "padding_mask": padding_mask,
        }


class FITBFeatureDataset(Dataset):
    """FITB evaluation dataset — no candidate pool, scores against all items globally."""

    def __init__(
        self,
        outfits: List[List[str]],
        item_features: Dict,
        all_item_ids: List[str],
        max_size: int = 10,
    ):
        self.item_features = item_features
        self.max_size = max_size
        self.id_to_global_idx = {iid: i for i, iid in enumerate(all_item_ids)}
        self.samples = []

        for outfit_ids in outfits:
            outfit_ids = outfit_ids[:max_size]
            for mask_idx in range(len(outfit_ids)):
                self.samples.append(
                    {
                        "outfit_ids": outfit_ids,
                        "mask_pos": mask_idx,
                        "gt_global_idx": self.id_to_global_idx[outfit_ids[mask_idx]],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        outfit_ids = sample["outfit_ids"]
        num_items = len(outfit_ids)

        img_embs = torch.stack([self.item_features[iid]["img_emb"] for iid in outfit_ids])
        txt_embs = torch.stack([self.item_features[iid]["txt_emb"] for iid in outfit_ids])

        pad_items = self.max_size - num_items
        if pad_items > 0:
            img_embs = F.pad(img_embs, (0, 0, 0, pad_items))
            txt_embs = F.pad(txt_embs, (0, 0, 0, pad_items))

        padding_mask = torch.zeros(self.max_size, dtype=torch.bool)
        padding_mask[num_items:] = True

        return {
            "img_embs": img_embs,
            "txt_embs": txt_embs,
            "padding_mask": padding_mask,
            "mask_pos": torch.tensor(sample["mask_pos"], dtype=torch.long),
            "gt_global_idx": torch.tensor(sample["gt_global_idx"], dtype=torch.long),
        }



def train_one_epoch(
    model: FOM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        img = batch["img_embs"].to(device)
        txt = batch["txt_embs"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        loss = model.compute_loss(img, txt, padding_mask=padding_mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_fitb(
    model: FOM,
    dataloader: DataLoader,
    global_img_embs: torch.Tensor,
    global_txt_embs: torch.Tensor,
    device: torch.device,
    ks: Tuple[int, ...] = (1, 5, 10, 20),
) -> Dict[str, float]:
    """Evaluate FITB by scoring against the full global item gallery."""
    model.eval()

    # Pre-compute global fused+transitioned embeddings: [N_items, hidden_dim]
    n_items = global_img_embs.size(0)
    batch_sz = 512
    global_hidden_parts = []
    for start in range(0, n_items, batch_sz):
        gi = global_img_embs[start : start + batch_sz].unsqueeze(0).to(device)
        gt = global_txt_embs[start : start + batch_sz].unsqueeze(0).to(device)
        fused = model.fuse_embeddings(gi, gt).squeeze(0)  # [chunk, embed_dim]
        global_hidden_parts.append(model.transition(fused).cpu())
    global_hidden = torch.cat(global_hidden_parts, dim=0).to(device)  # [N_items, hidden_dim]

    all_ranks = []

    for batch in dataloader:
        img = batch["img_embs"].to(device)
        txt = batch["txt_embs"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        mask_pos = batch["mask_pos"].to(device)
        gt_idx = batch["gt_global_idx"]  # [B]

        item_embs = model.fuse_embeddings(img, txt)
        mask_hidden, _ = model.forward_encoder(item_embs, mask_pos, padding_mask)
        # mask_hidden: [B, hidden_dim]
        # Score against all global items: [B, N_items]
        scores = mask_hidden @ global_hidden.T
        all_ranks.append((scores.cpu(), gt_idx))

    # Compute metrics
    all_scores_list = []
    all_gt_list = []
    for scores, gt_idx in all_ranks:
        all_scores_list.append(scores)
        all_gt_list.append(gt_idx)

    all_scores_t = torch.cat(all_scores_list, dim=0)  # [total_samples, N_items]
    all_gt_t = torch.cat(all_gt_list, dim=0)  # [total_samples]

    metrics = {}
    for k in ks:
        k_actual = min(k, all_scores_t.size(-1))
        _, topk_indices = all_scores_t.topk(k_actual, dim=-1)
        hits = (topk_indices == all_gt_t.unsqueeze(-1)).any(dim=-1).float()
        metrics[f"Recall@{k}"] = hits.mean().item()

        # NDCG@k
        matches = (topk_indices == all_gt_t.unsqueeze(-1))
        positions = matches.float() * torch.arange(1, k_actual + 1).float()
        rank = positions.sum(dim=-1)
        ndcg = torch.where(rank > 0, 1.0 / torch.log2(rank + 1), torch.zeros_like(rank))
        metrics[f"NDCG@{k}"] = ndcg.mean().item()

    return metrics


def load_feature_cache(cache_path: Path) -> Tuple[Dict, Dict]:
    payload = torch.load(cache_path, map_location="cpu")
    if "features" not in payload:
        return payload, {}
    return payload["features"], payload.get("texts", {})


def save_feature_cache(
    cache_path: Path,
    cfg: FOMConfig,
    text_mode: str,
    features: Dict,
    texts: Dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "texts": texts,
            "text_mode": text_mode,
            "fashionclip_model": cfg.fashionclip_model,
            "config": asdict(cfg),
        },
        cache_path,
    )


def prepare_item_features(
    cfg: FOMConfig,
    registry: SourceDataRegistry,
    item_ids: List[str],
    device: torch.device,
) -> Tuple[Dict, Dict, Path]:
    cache_name = (
        f"item_features__{sanitize_name(cfg.fashionclip_model)}__{cfg.text_mode}.pt"
    )
    cache_path = Path(cfg.feature_cache_dir) / cache_name

    if cache_path.exists() and not cfg.force_reextract:
        features, texts = load_feature_cache(cache_path)
    else:
        features, texts = {}, {}

    missing_item_ids = [item_id for item_id in item_ids if item_id not in features]
    if missing_item_ids:
        extractor = FeatureExtractor(cfg, device)
        new_features, new_texts = extractor.extract_items(
            registry.get_item_records(missing_item_ids),
            cfg.text_mode,
            cfg.feature_batch_size,
        )
        features.update(new_features)
        texts.update(new_texts)
        save_feature_cache(cache_path, cfg, cfg.text_mode, features, texts)

    for item_id in item_ids:
        if item_id not in texts:
            texts[item_id] = build_item_text(registry.items[item_id], cfg.text_mode)
    if missing_item_ids:
        save_feature_cache(cache_path, cfg, cfg.text_mode, features, texts)

    return features, texts, cache_path


def build_dataloaders(
    cfg: FOMConfig,
    train_outfits: List[List[str]],
    val_outfits: List[List[str]],
    test_outfits: List[List[str]],
    item_features: Dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    all_item_ids = sorted({item_id for outfit in train_outfits + val_outfits + test_outfits for item_id in outfit})

    train_loader = DataLoader(
        OutfitFeatureDataset(train_outfits, item_features, cfg.max_outfit_size),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        FITBFeatureDataset(
            val_outfits,
            item_features,
            all_item_ids,
            max_size=cfg.max_outfit_size,
        ),
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(
        FITBFeatureDataset(
            test_outfits,
            item_features,
            all_item_ids,
            max_size=cfg.max_outfit_size,
        ),
        batch_size=64,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, all_item_ids


def _build_global_embs(
    item_features: Dict,
    all_item_ids: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack global image/text embeddings aligned with all_item_ids ordering."""
    global_img = torch.stack([item_features[iid]["img_emb"] for iid in all_item_ids])
    global_txt = torch.stack([item_features[iid]["txt_emb"] for iid in all_item_ids])
    return global_img, global_txt


def train_and_evaluate(
    cfg: FOMConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    item_features: Dict,
    all_item_ids: List[str],
    device: torch.device,
    run_dir: Path,
) -> Dict:
    model = FOM(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    global_img, global_txt = _build_global_embs(item_features, all_item_ids)

    history = []
    best_metric = float("-inf")
    best_path = run_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        epoch_record = {"epoch": epoch, "train_loss": train_loss}
        if epoch % cfg.eval_every == 0 or epoch == 1 or epoch == cfg.epochs:
            val_metrics = evaluate_fitb(
                model, val_loader, global_img, global_txt, device, ks=(1, 5, 10, 20),
            )
            epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})

            selected = val_metrics[SELECTION_METRIC]
            if selected > best_metric:
                best_metric = selected
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(cfg),
                        "val_metrics": val_metrics,
                        "epoch": epoch,
                    },
                    best_path,
                )

            metric_str = " | ".join(
                f"{key}: {value:.4f}" for key, value in val_metrics.items()
            )
            print(
                f"[{cfg.text_mode}] Epoch {epoch:03d} | "
                f"TrainLoss: {train_loss:.4f} | {metric_str}"
            )
        else:
            print(f"[{cfg.text_mode}] Epoch {epoch:03d} | TrainLoss: {train_loss:.4f}")

        history.append(epoch_record)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_fitb(
        model, test_loader, global_img, global_txt, device, ks=(1, 5, 10, 20, 50),
    )

    return {
        "best_epoch": checkpoint["epoch"],
        "best_val_metrics": checkpoint["val_metrics"],
        "test_metrics": test_metrics,
        "history": history,
        "best_checkpoint": str(best_path),
    }


def run_single_experiment(
    cfg: FOMConfig,
    registry: SourceDataRegistry,
    device: torch.device,
    split_cache: Dict[str, Tuple[List[List[str]], Dict]],
    output_root: Path,
) -> Dict:
    split_stats = {split_name: stats for split_name, (_, stats) in split_cache.items()}
    alignment_report = registry.build_alignment_report(split_stats)

    train_outfits = split_cache["train"][0]
    val_outfits = split_cache["val"][0]
    test_outfits = split_cache["test"][0]

    all_item_ids = sorted(
        {item_id for outfit in train_outfits + val_outfits + test_outfits for item_id in outfit}
    )
    item_features, texts, cache_path = prepare_item_features(cfg, registry, all_item_ids, device)

    run_dir = output_root / cfg.text_mode
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_item_ids = all_item_ids[:3]
    save_json(
        run_dir / "text_samples.json",
        {item_id: texts[item_id] for item_id in sample_item_ids},
    )
    save_json(run_dir / "alignment_report.json", alignment_report)

    result = {
        "text_mode": cfg.text_mode,
        "feature_cache": str(cache_path),
        "num_items": len(all_item_ids),
        "num_train_outfits": len(train_outfits),
        "num_val_outfits": len(val_outfits),
        "num_test_outfits": len(test_outfits),
    }

    if cfg.prepare_only:
        print(
            f"[{cfg.text_mode}] Prepared {len(all_item_ids)} items and cached features at "
            f"{cache_path}"
        )
        return result

    train_loader, val_loader, test_loader, all_item_ids_sorted = build_dataloaders(
        cfg,
        train_outfits,
        val_outfits,
        test_outfits,
        item_features,
    )
    result.update(train_and_evaluate(
        cfg, train_loader, val_loader, test_loader,
        item_features, all_item_ids_sorted, device, run_dir,
    ))
    save_json(run_dir / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base_cfg = FOMConfig(
        data_root=args.data_root,
        source_data_root=args.source_data_root,
        feature_cache_dir=args.feature_cache_dir,
        output_dir=args.output_dir,
        fashionclip_model=args.fashionclip_model,
        fusion_mode=args.fusion_mode,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        ff_dim=args.ff_dim,
        n_neg_samples=args.n_neg_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        feature_batch_size=args.feature_batch_size,
        text_max_length=args.text_max_length,
        max_outfit_size=args.max_outfit_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        eval_every=args.eval_every,
        limit_outfits=args.limit_outfits,
        force_reextract=args.force_reextract,
        prepare_only=args.prepare_only,
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    registry = SourceDataRegistry(base_cfg.source_data_root)
    data_root = Path(base_cfg.data_root)

    split_cache = {}
    for split_name in ("train", "val", "test"):
        split_csv = data_root / f"{split_name}.csv"
        split_cache[split_name] = registry.build_outfits_from_split(
            str(split_csv),
            limit_outfits=base_cfg.limit_outfits,
        )

    output_root = Path(base_cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for text_mode in args.text_modes:
        cfg = replace(base_cfg, text_mode=text_mode)
        print(f"\n========== Running text mode: {text_mode} ==========")
        results[text_mode] = run_single_experiment(
            cfg,
            registry,
            device,
            split_cache,
            output_root,
        )

    summary_path = output_root / "summary.json"
    save_json(summary_path, results)

    if len(results) > 1 and not base_cfg.prepare_only:
        comparison = {
            text_mode: payload["test_metrics"].get(SELECTION_METRIC, None)
            for text_mode, payload in results.items()
        }
        print("\nText-mode comparison on test set:")
        for text_mode, score in comparison.items():
            print(f"  {text_mode}: {SELECTION_METRIC}={score:.4f}")

    print(f"\nSaved experiment summary to {summary_path}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
    main()
