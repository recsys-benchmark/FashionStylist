# FashionStylist: An Expert Knowledge-enhanced Multimodal Dataset for Fashion Understanding

## News

- [x] **April 1, 2026**: Released **FashionStylist V1**, including **1,000 outfit-level entries** and **4,637 item-level annotations** across **Female**, **Male**, and **Child** subsets.
<!-- - ☐ **TBA**: Release benchmark and baseline code.
- ☐ **TBA**: Release **FashionStylist V2**, a larger-scale version of the dataset. -->

## Dataset Overview

FashionStylist is an expert knowledge-enhanced multimodal fashion dataset for outfit-level and item-level fashion understanding. The current release organizes the data into three subsets:

- **Female**: 500 outfits, 2,406 items
- **Male**: 300 outfits, 1,390 items
- **Child**: 200 outfits, 841 items

In V1, each outfit is linked to all of its items through an outfit identifier (`outfitID`) and a list of item identifiers (`items`). The dataset supports research on outfit understanding, item attribute recognition, fashion description grounding, cross-modal retrieval, and multimodal reasoning.

<!-- All textual annotations in the current release are in **Chinese**. -->

### File Organization

```text
FashionStylist/
├── Female/
│   ├── look(b1-500).csv
│   └── label(p1-2406).csv
├── Male/
│   ├── look(b1-300).csv
│   └── label(p1-1390).csv
├── Child/
│   ├── look(b1-200).csv
│   └── label(p1-841).csv
└── README.md
```

### Annotation Schema

**Outfit-level annotations** (`look*.csv`)

- `outfitID`: outfit identifier
- `items`: comma-separated item identifiers belonging to the outfit
- `look`: free-form outfit description
- `season`: normalized season label, with 6 classes: `春`, `夏`, `秋`, `冬`, `春夏`, `秋冬`
- `occasion`: normalized occasion label, with 7 base classes (`运动`, `出行`, `日常`, `校园`, `社交`, `商务`, `居家`) and their slash-separated combinations (e.g., `日常/出行`, `运动/出行`)
- `URL link`: source product or style reference URL

**Item-level annotations** (`label*.csv`)

- `itemID`: item identifier
- `title`: item title
- `gender`: target gender group
- `style`: style annotation
- `outline`: silhouette / outline
- `materials`: material annotation
- `color`: color annotation
- `pattern`: pattern annotation
- `detail`: design detail annotation
- `donning/doffing`: wearing or removal mode
- `URL link`: source product URL

Note: the last field is semantically the same across all subsets, but its raw header name varies slightly in the released CSV files (`donning/doffing`, `donning、doffing`, or `donningdoffing`).

## Todo List

- [x] Release **FashionStylist V1**
- [ ] Release an **English version** of FashionStylist
- [ ] Release benchmark and baseline code for the dataset
- [ ] Release **FashionStylist V2**, a larger-scale version of the dataset

## Acknowledgement

Our benchmark baselines are built on several excellent open-source projects. We thank the authors and maintainers of these repositories:

- **CIRP**: https://github.com/HappyPointer/CIRP
- **DiFashion**: https://github.com/YiyanXu/DiFashion
- **CLHE**: https://github.com/Xiaohao-Liu/CLHE
