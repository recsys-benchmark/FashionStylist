#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from torchvision import transforms
except ImportError:  # pragma: no cover - torch is optional in this workspace
    torch = None
    DataLoader = None
    transforms = None

    class TorchDataset:  # type: ignore[override]
        pass


PHOTO_ID_RE = re.compile(r"(?:[mfkc])?p\d+", re.IGNORECASE)
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DATASET_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATASET_ROOT = os.environ.get(
    "FASHION_STYLIST_DATA_ROOT",
    os.environ.get("TASK3_DATASET_ROOT", str(SHARED_DATASET_ROOT)),
)
EXPECTED_SEGMENTS = ("female", "male", "kid")
SPLIT_CHOICES = ("all", "train", "val", "test")

BAG_KEYWORDS = (
    "bag",
    "tote",
    "kelly",
    "messenger",
    "crossbody",
    "shoulder bag",
    "backpack",
    "handbag",
    "purse",
    "wallet",
)
SHOE_KEYWORDS = (
    "shoe",
    "boot",
    "derby",
    "loafer",
    "mule",
    "birkenstock",
    "mary jane",
    "flip-flop",
    "sandal",
    "slipper",
    "sneaker",
)
ACCESSORY_KEYWORDS = (
    "hat",
    "scarf",
    "tie",
    "necklace",
    "earring",
    "glove",
    "sock",
    "watch",
    "earmuff",
    "headband",
    "wristband",
    "headphones",
    "sunglasses",
    "glasses",
    "drawstring",
    "accessory",
    "basketball",
    "soccer",
    "volleyball",
    "belt",
)
ONEPIECE_KEYWORDS = (
    "dress",
    "pinafore",
    "overall dress",
    "jumpsuit",
    "matching set",
)
BOTTOM_KEYWORDS = (
    "pants",
    "trousers",
    "skirt",
    "mini skirt",
    "long skirt",
    "midi skirt",
    "shorts",
    "jeans",
)
OUTERWEAR_KEYWORDS = (
    "coat",
    "jacket",
    "trench",
    "padded jacket",
    "down jacket",
    "leather jacket",
    "blazer",
    "baseball jacket",
    "cardigan",
)
MID_LAYER_TOP_KEYWORDS = (
    "sweatshirt",
    "sweater",
    "knit",
    "knitted top",
    "pullover",
    "vest",
)
INNER_TOP_KEYWORDS = (
    "t-shirt",
    "tee",
    "shirt",
    "tank",
    "polo",
    "top",
    "turtleneck",
    "base layer",
    "camisole",
    "blouse",
)
ACCESSORY_OVERRIDE_BLOCKERS = (
    "coat",
    "jacket",
    "trench",
    "padded jacket",
    "down jacket",
    "leather jacket",
    "blazer",
    "baseball jacket",
    "cardigan",
    "sweatshirt",
    "sweater",
    "knitted top",
    "pullover",
    "vest",
    "t-shirt",
    "tee",
    "shirt",
    "tank",
    "polo",
    "top",
    "turtleneck",
    "base layer",
    "camisole",
    "blouse",
)


@dataclass(frozen=True)
class ItemRecord:
    source_group: str
    photo_id: str
    title: str
    gender: str
    item_style: str
    major_category: str
    image_path: Path


@dataclass(frozen=True)
class OutfitRecord:
    source_group: str
    outfit_id: str
    outfit_style: str
    outfit_summary: str
    look_text: str
    season: str
    occasion: str
    item_records: tuple[ItemRecord, ...]


def parse_outfit_style(look_text: str) -> str:
    parts = re.split(r"[：:]", look_text, maxsplit=1)
    return parts[0].strip() if parts else look_text.strip()


def ensure_sentence_punctuation(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return normalized
    if normalized[-1] in ".!?;…":
        return normalized
    return f"{normalized}."


def parse_outfit_summary(look_text: str) -> str:
    parts = re.split(r"[：:]", look_text, maxsplit=1)
    if len(parts) == 2:
        return ensure_sentence_punctuation(parts[1])
    return ensure_sentence_punctuation(look_text)


def split_photo_ids(photo_text: str) -> list[str]:
    photo_ids = [photo_id.lower() for photo_id in PHOTO_ID_RE.findall(photo_text)]
    if photo_ids:
        return photo_ids
    return [part.strip().lower() for part in re.split(r",\s*", photo_text) if part.strip()]


def infer_audience_segment(source_group: str) -> str:
    lower_name = source_group.lower()
    if "female" in lower_name:
        return "female"
    if "male" in lower_name:
        return "male"
    if "kid" in lower_name or "child" in lower_name or "children" in lower_name:
        return "kid"
    return source_group


def get_first_present_value(row: dict[str, str], candidates: tuple[str, ...], row_context: str) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    candidate_text = ", ".join(candidates)
    raise KeyError(f"Missing required column among [{candidate_text}] in {row_context}")


def normalize_outfit_season(season: str) -> str:
    return season.strip()


def normalize_outfit_occasion(occasion: str) -> str:
    return occasion.strip()


def infer_major_category(title: str, item_style: str) -> str:
    text = f"{title} {item_style}".lower()
    if any(keyword in text for keyword in BAG_KEYWORDS):
        return "bag"
    if any(keyword in text for keyword in SHOE_KEYWORDS):
        return "shoes"
    if any(keyword in text for keyword in ACCESSORY_KEYWORDS) and not any(
        keyword in text for keyword in ACCESSORY_OVERRIDE_BLOCKERS
    ):
        return "accessory"
    if any(keyword in text for keyword in ONEPIECE_KEYWORDS):
        return "onepiece"
    if any(keyword in text for keyword in BOTTOM_KEYWORDS):
        return "bottom"
    if any(keyword in text for keyword in OUTERWEAR_KEYWORDS):
        return "outerwear"
    if any(keyword in text for keyword in MID_LAYER_TOP_KEYWORDS):
        return "mid_layer_top"
    if any(keyword in text for keyword in INNER_TOP_KEYWORDS):
        return "inner_top"
    if any(keyword in text for keyword in ACCESSORY_KEYWORDS):
        return "accessory"
    return "accessory"


def default_image_loader(image_path: Path, image_mode: str) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert(image_mode)


def build_default_transform() -> Callable[[Image.Image], Any] | None:
    if transforms is None:
        return None
    return transforms.ToTensor()


def summarize_images(images: list[Any]) -> list[Any]:
    image_summaries: list[Any] = []
    for image in images:
        if torch is not None and isinstance(image, torch.Tensor):
            image_summaries.append(tuple(image.shape))
        elif hasattr(image, "size"):
            image_summaries.append(getattr(image, "size"))
        else:
            image_summaries.append(type(image).__name__)
    return image_summaries


def format_mod_index(value: Any) -> Any:
    return "NONE" if value is None else value


class OutfitNegativeSampleDataset(TorchDataset):
    def __init__(
        self,
        root: str | Path = DEFAULT_DATASET_ROOT,
        transform: Callable[[Image.Image], Any] | None = None,
        image_loader: Callable[[Path, str], Any] | None = None,
        image_mode: str = "RGB",
        negative_scope: str = "same_group",
        seed: int = 42,
        deterministic: bool = False,
        sample_mode: str = "both",
        split: str = "all",
        split_seed: int = 42,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        if negative_scope not in {"same_group", "same_segment", "same_gender", "global"}:
            raise ValueError("negative_scope must be one of: same_group, same_segment, same_gender, global")
        if sample_mode not in {"original", "modified", "both"}:
            raise ValueError("sample_mode must be one of: original, modified, both")
        if split not in SPLIT_CHOICES:
            raise ValueError(f"split must be one of: {', '.join(SPLIT_CHOICES)}")

        self.transform = transform
        self.image_loader = image_loader or default_image_loader
        self.image_mode = image_mode
        self.negative_scope = negative_scope
        self.seed = seed
        self.deterministic = deterministic
        self.sample_mode = sample_mode
        self.split = split
        self.split_seed = split_seed
        self.epoch = 0

        self.items_by_group: dict[str, dict[str, ItemRecord]] = {}
        self.all_items: list[ItemRecord] = []
        self.all_outfits: list[OutfitRecord] = []
        self.outfits: list[OutfitRecord] = []
        self.outfit_counts_by_segment: dict[str, int] = {}
        self.split_outfit_counts_by_segment: dict[str, int] = {}

        self._load_index()
        self.outfit_style_candidates = sorted({outfit.outfit_style for outfit in self.all_outfits})
        self.outfit_season_candidates = sorted({outfit.season for outfit in self.all_outfits if outfit.season})
        self.outfit_occasion_candidates = sorted({outfit.occasion for outfit in self.all_outfits if outfit.occasion})

    def _load_index(self) -> None:
        outfits_by_segment: dict[str, list[OutfitRecord]] = {}
        for group_dir in self._discover_group_dirs():
            look_csv = self._find_single_file(group_dir, "look*.csv")
            label_csv = self._find_single_file(group_dir, "label*.csv")
            photo_dir = self._find_single_dir(group_dir, "photos*", required=False)

            items = self._load_items_for_group(group_dir.name, label_csv, photo_dir)
            outfits = self._load_outfits_for_group(group_dir.name, look_csv, items)
            audience_segment = infer_audience_segment(group_dir.name)

            self.items_by_group[group_dir.name] = items
            self.all_items.extend(items.values())
            self.all_outfits.extend(outfits)
            outfits_by_segment.setdefault(audience_segment, []).extend(outfits)

        if not self.all_outfits:
            raise ValueError(f"No outfits found under: {self.root}")
        self.outfit_counts_by_segment = {
            segment: len(segment_outfits)
            for segment, segment_outfits in sorted(outfits_by_segment.items())
        }
        self.outfits = self._select_outfits_for_split(outfits_by_segment)

    def _select_outfits_for_split(
        self,
        outfits_by_segment: dict[str, list[OutfitRecord]],
    ) -> list[OutfitRecord]:
        selected_outfits: list[OutfitRecord] = []
        split_counts: dict[str, int] = {}
        for segment in sorted(outfits_by_segment):
            split_buckets = self._build_segment_split_buckets(segment, outfits_by_segment[segment])
            if self.split == "all":
                segment_outfits = split_buckets["train"] + split_buckets["val"] + split_buckets["test"]
            else:
                segment_outfits = split_buckets[self.split]
            split_counts[segment] = len(segment_outfits)
            selected_outfits.extend(segment_outfits)
        self.split_outfit_counts_by_segment = split_counts
        return selected_outfits

    def _build_segment_split_buckets(
        self,
        segment: str,
        segment_outfits: list[OutfitRecord],
    ) -> dict[str, list[OutfitRecord]]:
        ordered_outfits = sorted(segment_outfits, key=lambda outfit: outfit.outfit_id)
        shuffled_outfits = list(ordered_outfits)
        random.Random(f"{self.split_seed}:{segment}").shuffle(shuffled_outfits)

        train_count = len(shuffled_outfits) * 7 // 10
        val_count = len(shuffled_outfits) // 10
        train_end = train_count
        val_end = train_end + val_count

        return {
            "train": shuffled_outfits[:train_end],
            "val": shuffled_outfits[train_end:val_end],
            "test": shuffled_outfits[val_end:],
        }

    def _discover_group_dirs(self) -> list[Path]:
        direct_children = sorted(path for path in self.root.iterdir() if path.is_dir())
        preferred_group_dirs = [
            path for path in direct_children if self._looks_like_group_dir(path) and self._is_expected_segment_dir(path)
        ]
        if preferred_group_dirs:
            return sorted(preferred_group_dirs, key=self._group_sort_key)
        group_dirs = [path for path in direct_children if self._looks_like_group_dir(path)]
        if group_dirs:
            return group_dirs
        if self._looks_like_group_dir(self.root):
            return [self.root]
        raise FileNotFoundError(
            f"No dataset group folders were found under {self.root}. "
            "Expected directories like Female, Male, and Child that each contain one look*.csv, one label*.csv, "
            "and one photos* directory."
        )

    @staticmethod
    def _is_expected_segment_dir(path: Path) -> bool:
        return infer_audience_segment(path.name) in EXPECTED_SEGMENTS

    @staticmethod
    def _group_sort_key(path: Path) -> tuple[int, str]:
        segment = infer_audience_segment(path.name)
        try:
            rank = EXPECTED_SEGMENTS.index(segment)
        except ValueError:
            rank = len(EXPECTED_SEGMENTS)
        return rank, path.name.lower()

    @staticmethod
    def _looks_like_group_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        has_look_csv = any(candidate.is_file() for candidate in path.glob("look*.csv"))
        has_label_csv = any(candidate.is_file() for candidate in path.glob("label*.csv"))
        return has_look_csv and has_label_csv

    @staticmethod
    def _find_single_file(group_dir: Path, pattern: str) -> Path:
        matches = sorted(path for path in group_dir.glob(pattern) if path.is_file())
        if not matches:
            raise FileNotFoundError(f"Missing file matching {pattern!r} in {group_dir}")
        return matches[0]

    @staticmethod
    def _find_single_dir(group_dir: Path, pattern: str, required: bool = True) -> Path | None:
        matches = sorted(path for path in group_dir.glob(pattern) if path.is_dir())
        if not matches:
            if required:
                raise FileNotFoundError(f"Missing directory matching {pattern!r} in {group_dir}")
            return None
        return matches[0]

    @staticmethod
    def _resolve_item_image_path(source_group: str, photo_id: str, photo_dir: Path | None) -> Path:
        if photo_dir is None:
            raise FileNotFoundError(
                f"Task3 requires item images, but no directory matching 'photos*' was found under the "
                f"dataset folder {source_group!r}. Add the image directory before running evaluation or SFT."
            )
        for suffix in (".png", ".jpg", ".jpeg", ".webp"):
            candidate = photo_dir / f"{photo_id}{suffix}"
            if candidate.is_file():
                return candidate
        return photo_dir / f"{photo_id}.png"

    def _load_items_for_group(
        self,
        source_group: str,
        label_csv: Path,
        photo_dir: Path | None,
    ) -> dict[str, ItemRecord]:
        items: dict[str, ItemRecord] = {}
        with label_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle), start=2):
                row_context = f"{label_csv}:{row_index}"
                photo_id = get_first_present_value(row, ("photo", "itemID", "item_id"), row_context)
                title = get_first_present_value(row, ("title",), row_context)
                gender = get_first_present_value(row, ("gender",), row_context)
                item_style = get_first_present_value(row, ("style",), row_context)
                items[photo_id] = ItemRecord(
                    source_group=source_group,
                    photo_id=photo_id,
                    title=title,
                    gender=gender,
                    item_style=item_style,
                    major_category=infer_major_category(title, item_style),
                    image_path=self._resolve_item_image_path(source_group, photo_id, photo_dir),
                )
        return items

    def _load_outfits_for_group(
        self,
        source_group: str,
        look_csv: Path,
        items: dict[str, ItemRecord],
    ) -> list[OutfitRecord]:
        outfits: list[OutfitRecord] = []
        with look_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle), start=2):
                row_context = f"{look_csv}:{row_index}"
                outfit_id = get_first_present_value(row, ("outfit", "outfitID", "outfit_id"), row_context)
                photo_ids_text = get_first_present_value(row, ("photos", "items", "itemIDs"), row_context)
                look_text = get_first_present_value(row, ("look",), row_context)
                item_records = []
                for photo_id in split_photo_ids(photo_ids_text):
                    if photo_id not in items:
                        raise KeyError(f"Missing item {photo_id!r} referenced by outfit {outfit_id!r}")
                    item_records.append(items[photo_id])

                outfits.append(
                    OutfitRecord(
                        source_group=source_group,
                        outfit_id=outfit_id,
                        outfit_style=parse_outfit_style(look_text),
                        outfit_summary=parse_outfit_summary(look_text),
                        look_text=look_text,
                        season=normalize_outfit_season(row.get("season", "")),
                        occasion=normalize_outfit_occasion(row.get("occasion", "")),
                        item_records=tuple(item_records),
                    )
                )
        return outfits

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.sample_mode == "both":
            return len(self.outfits) * 2
        return len(self.outfits)

    def __getitem__(self, index: int) -> dict[str, Any]:
        outfit, need_to_modify = self._resolve_sample(index)
        original_items = list(outfit.item_records)

        if need_to_modify:
            rng = self._build_rng(index)
            negative_index, _, negative_item, _ = self._select_negative_replacement(
                source_outfit=outfit,
                original_items=original_items,
                rng=rng,
            )
            input_items = list(original_items)
            input_items[negative_index] = negative_item
            mod_index = negative_index
        else:
            input_items = original_items
            mod_index = None

        images = [self._load_image(item.image_path) for item in input_items]

        return {
            "input": {
                "images": images,
                "outfit_id": outfit.outfit_id,
                "group": outfit.source_group,
                "candidate_style": self.outfit_style_candidates,
                "candidate_season": self.outfit_season_candidates,
                "candidate_occasion": self.outfit_occasion_candidates,
            },
            "label": {
                "outfit_summary": outfit.outfit_summary,
                "outfit_style": outfit.outfit_style,
                "season": outfit.season,
                "occasion": outfit.occasion,
                "need_to_modify": int(need_to_modify),
                "mod_index": mod_index,
            },
        }

    def _resolve_sample(self, index: int) -> tuple[OutfitRecord, bool]:
        if not 0 <= index < len(self):
            raise IndexError(f"Sample index out of range: {index}")

        if self.sample_mode == "original":
            return self.outfits[index], False
        if self.sample_mode == "modified":
            return self.outfits[index], True

        outfit_index = index // 2
        need_to_modify = bool(index % 2)
        return self.outfits[outfit_index], need_to_modify

    def _build_rng(self, index: int) -> random.Random:
        if not self.deterministic:
            return random.Random()
        return random.Random(f"{self.seed}:{self.epoch}:{index}")

    def _load_image(self, image_path: Path) -> Any:
        image = self.image_loader(image_path, self.image_mode)
        return self.transform(image) if self.transform is not None else image

    def _scope_priority(self) -> tuple[str, ...]:
        if self.negative_scope == "same_group":
            return ("same_group", "same_segment", "global")
        if self.negative_scope in {"same_segment", "same_gender"}:
            return ("same_segment", "global")
        return ("global",)

    def _select_negative_replacement(
        self,
        source_outfit: OutfitRecord,
        original_items: list[ItemRecord],
        rng: random.Random,
    ) -> tuple[int, ItemRecord, ItemRecord, str]:
        replacement_options = self._collect_replacement_options(source_outfit, original_items)
        if not replacement_options:
            raise RuntimeError(f"No valid negative item found for outfit {source_outfit.outfit_id}")

        category_to_options: dict[str, list[tuple[int, ItemRecord, list[ItemRecord], str]]] = {}
        for option in replacement_options:
            category_to_options.setdefault(option[1].major_category, []).append(option)

        chosen_category = rng.choice(list(category_to_options))
        negative_index, replaced_item, candidates, scope_used = rng.choice(category_to_options[chosen_category])
        negative_item = rng.choice(candidates)
        return negative_index, replaced_item, negative_item, scope_used

    def _collect_replacement_options(
        self,
        source_outfit: OutfitRecord,
        original_items: list[ItemRecord],
    ) -> list[tuple[int, ItemRecord, list[ItemRecord], str]]:
        replacement_options: list[tuple[int, ItemRecord, list[ItemRecord], str]] = []
        for negative_index, replaced_item in enumerate(original_items):
            sampled = self._find_negative_candidates(
                source_outfit=source_outfit,
                replaced_item=replaced_item,
                original_items=original_items,
            )
            if sampled is None:
                continue
            candidates, scope_used = sampled
            replacement_options.append((negative_index, replaced_item, candidates, scope_used))
        return replacement_options

    def _find_negative_candidates(
        self,
        source_outfit: OutfitRecord,
        replaced_item: ItemRecord,
        original_items: list[ItemRecord],
    ) -> tuple[list[ItemRecord], str] | None:
        blocked_keys = {(item.source_group, item.photo_id) for item in original_items}
        for scope in self._scope_priority():
            candidates = [
                candidate
                for candidate in self.all_items
                if candidate.major_category == replaced_item.major_category
                and candidate.gender == replaced_item.gender
                and candidate.item_style != replaced_item.item_style
                and (candidate.source_group, candidate.photo_id) not in blocked_keys
                and self._matches_scope(scope, source_outfit, replaced_item, candidate)
            ]
            if candidates:
                return candidates, scope

        return None

    @staticmethod
    def _matches_scope(
        scope: str,
        source_outfit: OutfitRecord,
        replaced_item: ItemRecord,
        candidate: ItemRecord,
    ) -> bool:
        if scope == "same_group":
            return candidate.source_group == source_outfit.source_group
        if scope == "same_segment":
            return infer_audience_segment(candidate.source_group) == infer_audience_segment(source_outfit.source_group)
        if scope == "global":
            return True
        return False


def collate_outfit_negative_samples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        return {}

    input_batch = {
        "images": [sample["input"]["images"] for sample in batch],
        "outfit_id": [sample["input"]["outfit_id"] for sample in batch],
        "group": [sample["input"]["group"] for sample in batch],
        "candidate_style": batch[0]["input"]["candidate_style"],
        "candidate_season": batch[0]["input"]["candidate_season"],
        "candidate_occasion": batch[0]["input"]["candidate_occasion"],
        "outfit_sizes": [len(sample["input"]["images"]) for sample in batch],
    }
    label_batch = {
        "outfit_summary": [sample["label"]["outfit_summary"] for sample in batch],
        "outfit_style": [sample["label"]["outfit_style"] for sample in batch],
        "season": [sample["label"]["season"] for sample in batch],
        "occasion": [sample["label"]["occasion"] for sample in batch],
        "need_to_modify": [sample["label"]["need_to_modify"] for sample in batch],
        "mod_index": [sample["label"]["mod_index"] for sample in batch],
    }
    collated = {"input": input_batch, "label": label_batch}

    if torch is None:
        return collated

    image_batches = input_batch["images"]
    if not all(images and all(isinstance(image, torch.Tensor) for image in images) for images in image_batches):
        return collated

    image_shapes = {tuple(image.shape) for images in image_batches for image in images}
    if len(image_shapes) != 1:
        return collated

    sample_tensor = image_batches[0][0]
    max_items = max(len(images) for images in image_batches)
    padded_images = torch.zeros(
        (len(image_batches), max_items, *sample_tensor.shape),
        dtype=sample_tensor.dtype,
    )
    item_mask = torch.zeros((len(image_batches), max_items), dtype=torch.bool)
    mod_mask = torch.zeros((len(image_batches), max_items), dtype=torch.bool)

    for batch_index, images in enumerate(image_batches):
        for item_index, image in enumerate(images):
            padded_images[batch_index, item_index] = image
            item_mask[batch_index, item_index] = True
        mod_index = label_batch["mod_index"][batch_index]
        if mod_index is not None:
            mod_mask[batch_index, mod_index] = True

    input_batch["images"] = padded_images
    input_batch["item_mask"] = item_mask
    input_batch["mod_mask"] = mod_mask
    input_batch["outfit_sizes"] = torch.tensor(input_batch["outfit_sizes"], dtype=torch.long)
    label_batch["need_to_modify"] = torch.tensor(label_batch["need_to_modify"], dtype=torch.long)
    if all(mod_index is not None for mod_index in label_batch["mod_index"]):
        label_batch["mod_index"] = torch.tensor(label_batch["mod_index"], dtype=torch.long)
    return collated


def create_dataloader(
    dataset: OutfitNegativeSampleDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Any:
    if DataLoader is None:
        raise RuntimeError("torch is not available. Please use the configured flowtimo interpreter.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_outfit_negative_samples,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview a batch loaded by torch DataLoader.")
    parser.add_argument(
        "--root",
        default=DEFAULT_DATASET_ROOT,
        help=(
            "Dataset root directory. Pass the parent directory that contains the three group folders, "
            "for example Female, Male, and Child, and ensure each group folder contains look*.csv, "
            f"label*.csv, and photos*. Default: {DEFAULT_DATASET_ROOT}"
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="How many rows to print from the first batch. Default: 3",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="DataLoader batch size. Default: 4",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader num_workers. Default: 0",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("original", "modified", "both"),
        default="both",
        help="Which samples to include: original, modified, or both. Default: both",
    )
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default="all",
        help="Dataset split to preview: all, train, val, or test. Default: all",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used for the per-segment train/val/test split. Default: 42",
    )
    args = parser.parse_args()

    transform = build_default_transform()
    if transform is None:
        raise RuntimeError("torchvision is not available. Please use the configured flowtimo interpreter.")

    dataset = OutfitNegativeSampleDataset(
        root=args.root,
        transform=transform,
        sample_mode=args.sample_mode,
        split=args.split,
        split_seed=args.split_seed,
    )
    sample = dataset[0]
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(dataloader))
    preview_count = max(0, min(args.num_samples, len(batch["input"]["outfit_id"])))

    sample_input_preview = {
        "outfit_id": sample["input"]["outfit_id"],
        "group": sample["input"]["group"],
        "candidate_style_count": len(sample["input"]["candidate_style"]),
        "candidate_style_preview": sample["input"]["candidate_style"][:10],
        "candidate_season_count": len(sample["input"]["candidate_season"]),
        "candidate_season_preview": sample["input"]["candidate_season"][:10],
        "candidate_occasion_count": len(sample["input"]["candidate_occasion"]),
        "candidate_occasion_preview": sample["input"]["candidate_occasion"][:10],
        "num_images": len(sample["input"]["images"]),
        "image_shapes": summarize_images(sample["input"]["images"]),
    }
    sample_label_preview = sample["label"]
    sample_label_preview = {
        **sample_label_preview,
        "mod_index": format_mod_index(sample_label_preview["mod_index"]),
    }
    batch_input_preview = {
        "outfit_id": batch["input"]["outfit_id"],
        "group": batch["input"]["group"],
        "candidate_style_count": len(batch["input"]["candidate_style"]),
        "candidate_style_preview": batch["input"]["candidate_style"][:10],
        "candidate_season_count": len(batch["input"]["candidate_season"]),
        "candidate_season_preview": batch["input"]["candidate_season"][:10],
        "candidate_occasion_count": len(batch["input"]["candidate_occasion"]),
        "candidate_occasion_preview": batch["input"]["candidate_occasion"][:10],
        "outfit_sizes": (
            batch["input"]["outfit_sizes"].tolist()
            if torch is not None and isinstance(batch["input"]["outfit_sizes"], torch.Tensor)
            else batch["input"]["outfit_sizes"]
        ),
        "images_shape": (
            tuple(batch["input"]["images"].shape)
            if torch is not None and isinstance(batch["input"]["images"], torch.Tensor)
            else [summarize_images(images) for images in batch["input"]["images"]]
        ),
        "item_mask_shape": (
            tuple(batch["input"]["item_mask"].shape)
            if "item_mask" in batch["input"]
            and torch is not None
            and isinstance(batch["input"]["item_mask"], torch.Tensor)
            else None
        ),
        "mod_mask_shape": (
            tuple(batch["input"]["mod_mask"].shape)
            if "mod_mask" in batch["input"]
            and torch is not None
            and isinstance(batch["input"]["mod_mask"], torch.Tensor)
            else None
        ),
    }
    batch_label_preview = {
        "outfit_style": batch["label"]["outfit_style"],
        "outfit_summary": batch["label"]["outfit_summary"],
        "season": batch["label"]["season"],
        "occasion": batch["label"]["occasion"],
        "need_to_modify": (
            batch["label"]["need_to_modify"].tolist()
            if torch is not None and isinstance(batch["label"]["need_to_modify"], torch.Tensor)
            else batch["label"]["need_to_modify"]
        ),
        "mod_index": (
            batch["label"]["mod_index"].tolist()
            if torch is not None and isinstance(batch["label"]["mod_index"], torch.Tensor)
            else [format_mod_index(mod_index) for mod_index in batch["label"]["mod_index"]]
        ),
    }

    print(f"dataset_size={len(dataset)}")
    print(f"split={dataset.split}")
    print(f"split_seed={dataset.split_seed}")
    print(f"outfit_count={len(dataset.outfits)}")
    print(f"outfit_counts_by_segment={dataset.outfit_counts_by_segment}")
    print(f"split_outfit_counts_by_segment={dataset.split_outfit_counts_by_segment}")
    print(f"sample_mode={args.sample_mode}")
    print(f"batch_size={len(batch['input']['outfit_id'])}")
    if torch is not None and isinstance(batch["input"]["outfit_sizes"], torch.Tensor):
        print(f"outfit_sizes={batch['input']['outfit_sizes'].tolist()}")
    else:
        print(f"outfit_sizes={batch['input']['outfit_sizes']}")
    print(f"candidate_style_count={len(batch['input']['candidate_style'])}")
    print(f"candidate_style_preview={batch['input']['candidate_style'][:10]}")
    print(f"candidate_season_count={len(batch['input']['candidate_season'])}")
    print(f"candidate_season_preview={batch['input']['candidate_season'][:10]}")
    print(f"candidate_occasion_count={len(batch['input']['candidate_occasion'])}")
    print(f"candidate_occasion_preview={batch['input']['candidate_occasion'][:10]}")
    if torch is not None and isinstance(batch["input"]["images"], torch.Tensor):
        print(f"images_shape={tuple(batch['input']['images'].shape)}")
    if "item_mask" in batch["input"]:
        print(f"item_mask_shape={tuple(batch['input']['item_mask'].shape)}")
    print(f"preview_count={preview_count}")
    print(f"input_keys={list(batch['input'])}")
    print(f"label_keys={list(batch['label'])}")
    print("\ndataset[0]['input']:")
    print(pformat(sample_input_preview, sort_dicts=False))
    print("\ndataset[0]['label']:")
    print(pformat(sample_label_preview, sort_dicts=False))
    print("\nbatch['input']:")
    print(pformat(batch_input_preview, sort_dicts=False))
    print("\nbatch['label']:")
    print(pformat(batch_label_preview, sort_dicts=False))

    for index in range(preview_count):
        if torch is not None and isinstance(batch["label"]["need_to_modify"], torch.Tensor):
            need_to_modify = int(batch["label"]["need_to_modify"][index])
        else:
            need_to_modify = batch["label"]["need_to_modify"][index]

        mod_index = batch["label"]["mod_index"][index]
        if torch is not None and isinstance(batch["label"]["mod_index"], torch.Tensor):
            mod_index = int(mod_index)
        else:
            mod_index = format_mod_index(mod_index)

        outfit_size = batch["input"]["outfit_sizes"][index]
        if torch is not None and isinstance(batch["input"]["outfit_sizes"], torch.Tensor):
            outfit_size = int(outfit_size)

        print(f"\n[{index}] outfit_id={batch['input']['outfit_id'][index]} group={batch['input']['group'][index]}")
        print(f"outfit_style={batch['label']['outfit_style'][index]}")
        print(f"season={batch['label']['season'][index]}")
        print(f"occasion={batch['label']['occasion'][index]}")
        print(f"outfit_summary={batch['label']['outfit_summary'][index]}")
        print(f"need_to_modify={need_to_modify} mod_index={mod_index}")
        print(f"outfit_size={outfit_size}")
    # print(batch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
