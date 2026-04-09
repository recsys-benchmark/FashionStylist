from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MODEL_ALIASES = {
    "qwen": "qwen_edit",
    "qwen-edit": "qwen_edit",
    "qwen_edit": "qwen_edit",
    "longcat": "longcat_edit_turbo",
    "longcat-edit": "longcat_edit_turbo",
    "longcat_edit": "longcat_edit_turbo",
    "longcat-edit-turbo": "longcat_edit_turbo",
    "longcat_edit_turbo": "longcat_edit_turbo",
    "longcat-image-edit": "longcat_edit_turbo",
    "longcat-image-edit-turbo": "longcat_edit_turbo",
    "flux": "flux_kontext",
    "flux-kontext": "flux_kontext",
    "flux.kontext": "flux_kontext",
    "flux_kontext": "flux_kontext",
}

CATEGORY_ALIASES = {
    "inner top": "inner top",
    "mid layer top": "mid layer top",
    "onepiece": "one piece",
}

DEFAULT_SINGLE_ITEM_NEGATIVE_PROMPT = (
    "(human feature, skin, face, hand, arm, leg, model body:1.5), original background, "
    "environmental shadows, other garments, overlapping clothes, noise, artifacts, "
    "duplicate items, multiple instances, hallucinated patterns, missing parts."
)

SINGLE_ITEM_CATEGORY_CONSTRAINTS = {
    "outerwear": {
        "shape_constraint": "Reconstruct the full shape of this outerwear, including the collar and sleeves.",
        "quantity_constraint": "Strictly present exactly ONE single un-duplicated item.",
    },
    "inner top": {
        "shape_constraint": (
            "Reconstruct the full torso and neckline of this inner top as if laid out new, "
            "without the outerwear obscuring it."
        ),
        "quantity_constraint": "Strictly present exactly ONE single un-duplicated item.",
    },
    "mid layer top": {
        "shape_constraint": (
            "Reconstruct the full torso, neckline, and sleeves of this mid layer top without any occlusion."
        ),
        "quantity_constraint": "Strictly present exactly ONE single un-duplicated item.",
    },
    "bottom": {
        "shape_constraint": "Reconstruct the full shape from waistband to bottom hems.",
        "quantity_constraint": "Strictly present exactly ONE single pair of bottoms. Zero duplicate pants or skirts.",
    },
    "one piece": {
        "shape_constraint": (
            "Extract the entire continuous garment from the neckline down to the hem. "
            "Do not crop or cut the item in half."
        ),
        "quantity_constraint": "Strictly present exactly ONE complete continuous garment.",
    },
    "shoes": {
        "shape_constraint": "Keep the visible shoe product complete and natural.",
        "quantity_constraint": "Do not invent extra shoes or force a pair if the source only supports a single shoe product.",
    },
    "bag": {
        "shape_constraint": "Retain the full structural volume and straps/handles of the bag.",
        "quantity_constraint": "Strictly present exactly ONE single un-duplicated bag.",
    },
    "accessory": {
        "shape_constraint": "Maintain highly detailed macro textures of the material (e.g., metal, leather, knit).",
        "quantity_constraint": "Present the accessory clearly. If it is a paired item (like earrings or gloves), keep them together.",
    },
}

FLUX_SHORT_CATEGORY_HINTS = {
    "outerwear": "Show full collar and sleeves.",
    "inner top": "Show full torso and neckline.",
    "mid layer top": "Show full torso, neckline, and sleeves.",
    "bottom": "Show full waistband to hems.",
    "one piece": "Show full garment neckline to hem.",
    "shoes": "Show both shoes as one matching pair.",
    "bag": "Keep full bag and straps.",
    "accessory": "Keep the full accessory; keep pairs together.",
}


@dataclass(frozen=True)
class PromptBundle:
    qwen_edit_infer: str
    qwen_edit_negative: str
    longcat_edit_infer: str
    longcat_edit_negative: str
    flux_kontext_infer: str
    qwen_sft: str
    longcat_sft: str
    flux_sft: str


def canonical_model_name(name: str) -> str:
    key = name.strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model: {name}")
    return MODEL_ALIASES[key]


def _normalize_items(items: Iterable[str] | None) -> list[str]:
    if items is None:
        return []
    normalized = []
    for item in items:
        value = str(item).strip()
        if value:
            normalized.append(value)
    return normalized


def _single_item_category(items: list[str]) -> str | None:
    if not items:
        return None
    return items[0]


def _normalize_category_name(category: str | None) -> str:
    if not category:
        return ""
    normalized = category.strip().lower().replace("-", " ").replace("_", " ")
    return CATEGORY_ALIASES.get(normalized, normalized)


def generate_extraction_prompt(
    category: str | None,
    outfit_summary: str | None = None,
    extra_constraints: str | None = None,
) -> tuple[str, str]:
    normalized_category = _normalize_category_name(category)
    display_category = normalized_category or "garment"

    base_prompt = (
        f"A professional e-commerce product flat-lay photography of the isolated {display_category} "
        "worn by the model in the source image. Top-down view, completely cut out and perfectly "
        "positioned on a pure solid white (#FFFFFF) background. "
        "{shape_constraint} "
        "{quantity_constraint} "
        "Maintain the exact original fabric, texture, color, and structural details from the source. "
        "High-end clean studio lighting, pristine and highly detailed, sharp edges without any bleeding."
    )

    constraints = SINGLE_ITEM_CATEGORY_CONSTRAINTS.get(
        normalized_category,
        {
            "shape_constraint": "Reconstruct the full recognizable shape of the item.",
            "quantity_constraint": "Strictly present exactly ONE single instance.",
        },
    )

    positive_prompt = base_prompt.format(
        shape_constraint=constraints["shape_constraint"],
        quantity_constraint=constraints["quantity_constraint"],
    )

    context_parts = []
    if outfit_summary:
        context_parts.append(f"Reference outfit summary: {outfit_summary.strip()}.")
    if extra_constraints:
        context_parts.append(f"Additional constraints: {extra_constraints.strip()}.")
    if context_parts:
        positive_prompt = positive_prompt + " " + " ".join(context_parts)

    return positive_prompt, DEFAULT_SINGLE_ITEM_NEGATIVE_PROMPT


def generate_flux_extraction_prompt(category: str | None) -> str:
    normalized_category = _normalize_category_name(category)
    display_category = normalized_category or "garment"
    category_hint = FLUX_SHORT_CATEGORY_HINTS.get(normalized_category, "Show the full item.")
    return (
        f"Extract only the {display_category} as a flat lay on pure white. "
        f"{category_hint} Keep exact color, fabric, shape, and details. One item only."
    )


def build_prompt_bundle(
    items: Iterable[str] | None = None,
    outfit_summary: str | None = None,
    extra_constraints: str | None = None,
) -> PromptBundle:
    item_list = _normalize_items(items)
    single_item_category = _single_item_category(item_list)
    summary = outfit_summary.strip() if outfit_summary else ""
    constraints = extra_constraints.strip() if extra_constraints else ""

    qwen_edit_infer, qwen_edit_negative = generate_extraction_prompt(
        single_item_category,
        outfit_summary=summary,
        extra_constraints=constraints,
    )

    longcat_edit_infer, longcat_edit_negative = generate_extraction_prompt(
        single_item_category,
        outfit_summary=summary,
        extra_constraints=constraints,
    )

    flux_kontext_infer = generate_flux_extraction_prompt(single_item_category)

    qwen_sft, _ = generate_extraction_prompt(
        single_item_category,
        outfit_summary=summary,
        extra_constraints=constraints,
    )

    longcat_sft, _ = generate_extraction_prompt(
        single_item_category,
        outfit_summary=summary,
        extra_constraints=constraints,
    )

    flux_sft = generate_flux_extraction_prompt(single_item_category)

    return PromptBundle(
        qwen_edit_infer=qwen_edit_infer,
        qwen_edit_negative=qwen_edit_negative,
        longcat_edit_infer=longcat_edit_infer,
        longcat_edit_negative=longcat_edit_negative,
        flux_kontext_infer=flux_kontext_infer,
        qwen_sft=qwen_sft,
        longcat_sft=longcat_sft,
        flux_sft=flux_sft,
    )


def select_inference_prompt(
    model_name: str,
    items: Iterable[str] | None = None,
    outfit_summary: str | None = None,
    extra_constraints: str | None = None,
) -> tuple[str, str | None]:
    bundle = build_prompt_bundle(items=items, outfit_summary=outfit_summary, extra_constraints=extra_constraints)
    canonical = canonical_model_name(model_name)
    if canonical == "qwen_edit":
        return bundle.qwen_edit_infer, bundle.qwen_edit_negative
    if canonical == "longcat_edit_turbo":
        return bundle.longcat_edit_infer, bundle.longcat_edit_negative
    if canonical == "flux_kontext":
        return bundle.flux_kontext_infer, None
    raise ValueError(f"Unsupported model: {model_name}")


def build_training_prompt(
    model_name: str,
    items: Iterable[str] | None = None,
    outfit_summary: str | None = None,
    extra_constraints: str | None = None,
) -> str:
    bundle = build_prompt_bundle(items=items, outfit_summary=outfit_summary, extra_constraints=extra_constraints)
    canonical = canonical_model_name(model_name)
    if canonical == "qwen_edit":
        return bundle.qwen_sft
    if canonical == "longcat_edit_turbo":
        return bundle.longcat_sft
    if canonical == "flux_kontext":
        return bundle.flux_sft
    raise ValueError(f"Unsupported model: {model_name}")
