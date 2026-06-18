"""
build_confusion_dataset.py

1. Loads kavinh07/ocr_dataset_shamadhan_synth_30k_p2 (all splits) from HuggingFace Hub.
2. Generates confusion-pair training images locally (same pipeline as generate_combined.py).
3. Merges: existing_train + confusion → new train  |  existing_val → val unchanged.
4. Saves sharded Parquet to --output_dir.
5. Optionally pushes to Hub with --push_to_hub.

Usage (inside Docker):
  python3 build_confusion_dataset.py \
    --source_dataset kavinh07/ocr_dataset_shamadhan_synth_30k_p2 \
    --confusion_count 300 \
    --font_size 11 \
    --blur 0.3 \
    --work_dir /app/confusion_work \
    --output_dir /app/hf_extended \
    [--push_to_hub your-org/dataset-name] \
    [--hf_token your_token] \
    [--reset]
"""

import argparse
import math
import os
import random
import sys

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

# Import confusion constants and image helpers from generate_combined
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_combined import (
    _BN_CONFUSION_CHARS,
    _apply_augraphy,
    _load_json,
    _rotate_image,
    _save_json,
)
from trdg.data_generator import FakeTextDataGenerator
from trdg.utils import load_fonts


# ── HF dataset schema ────────────────────────────────────────────────────────

def _hf_features():
    from datasets import Features, Value
    from datasets import Image as HFImage
    return Features({
        'image':      HFImage(),
        'text':       Value('string'),
        'class_name': Value('string'),
        'source':     Value('string'),
    })


# ── Confusion image generation ───────────────────────────────────────────────

def generate_confusion_images(work_dir, confusion_count, font_size, blur, reset):
    """Generate confusion-pair images to work_dir/images/ + work_dir/labels/."""
    print(f"\n── Generating confusion images ({confusion_count}/char × "
          f"{len(_BN_CONFUSION_CHARS)} chars) ──")

    image_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'trdg', 'images'
    )
    fonts = load_fonts('bn')
    plan_path = os.path.join(work_dir, '.plan_confusion.json')

    if not reset and os.path.exists(plan_path):
        print(f"  Resuming from {plan_path}")
        plan = _load_json(plan_path)
    else:
        os.makedirs(os.path.join(work_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'labels'), exist_ok=True)
        plan = []
        idx = 0
        for char, words in _BN_CONFUSION_CHARS.items():
            for _ in range(confusion_count):
                word = random.choice(words)
                plan.append({
                    'idx':        idx,
                    'text':       word,
                    'char':       char,
                    'image_path': os.path.join(work_dir, 'images', f'bn_conf_{idx}.png'),
                    'label_path': os.path.join(work_dir, 'labels', f'bn_conf_{idx}.txt'),
                })
                idx += 1
        _save_json(plan, plan_path)

    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    print(f"  {len(plan) - len(todo)} done, {len(todo)} remaining")

    for item in tqdm(todo, desc='Confusion images'):
        font = random.choice(fonts)
        img = None
        attempts = 0
        while img is None and attempts < 5:
            img = FakeTextDataGenerator.generate(
                index=item['idx'],
                text=item['text'],
                font=font,
                out_dir=None,
                size=font_size,
                extension=None,
                skewing_angle=0,
                random_skew=False,
                blur=blur,
                random_blur=True,
                background_type=3,
                distorsion_type=0,
                distorsion_orientation=0,
                is_handwritten=False,
                name_format=0,
                width=-1,
                alignment=1,
                text_color='#000000',
                orientation=0,
                space_width=1.0,
                character_spacing=0,
                margins=(5, 5, 5, 5),
                fit=True,
                output_mask=False,
                word_split=True,
                image_dir=image_dir,
                stroke_width=0,
                stroke_fill='#282828',
                image_mode='RGB',
            )
            attempts += 1

        if img is None:
            continue

        img = _rotate_image(img)
        img = _apply_augraphy(img)
        img.save(item['image_path'])
        with open(item['label_path'], 'w', encoding='utf-8') as f:
            f.write(item['text'])

    return plan


# ── Build HF Dataset from confusion images ───────────────────────────────────

def build_confusion_dataset(plan):
    """Build a HuggingFace Dataset from generated confusion images."""
    from datasets import Dataset

    features = _hf_features()

    def gen():
        for item in plan:
            if not os.path.exists(item['image_path']):
                continue
            try:
                img = PILImage.open(item['image_path']).convert('RGB')
                yield {
                    'image':      img,
                    'text':       item['text'],
                    'class_name': 'confusion_training',
                    'source':     'synthetic_confusion',
                }
            except Exception as e:
                print(f"  skip {item['image_path']}: {e}")

    print(f"\nBuilding confusion Dataset from {len(plan)} images...")
    return Dataset.from_generator(gen, features=features)


# ── Shard writer ─────────────────────────────────────────────────────────────

def write_sharded_parquet(dataset, split_dir, shard_size, split_name):
    """Write dataset as sharded Parquet files."""
    total = len(dataset)
    n_shards = math.ceil(total / shard_size)
    os.makedirs(split_dir, exist_ok=True)
    print(f"\nWriting {split_name}: {total:,} samples → {n_shards} shards")

    for i in tqdm(range(n_shards), desc=f'Shards {split_name}'):
        shard_path = os.path.join(
            split_dir, f'shard-{i:04d}-of-{n_shards:04d}.parquet'
        )
        if os.path.exists(shard_path):
            continue
        chunk = dataset.select(range(i * shard_size, min((i + 1) * shard_size, total)))
        tmp = shard_path + '.tmp'
        chunk.to_parquet(tmp)
        os.replace(tmp, shard_path)


# ── Dataset card ─────────────────────────────────────────────────────────────

def write_readme(output_dir, source_dataset, train_n, val_n, confusion_n):
    card = os.path.join(output_dir, 'README.md')
    with open(card, 'w', encoding='utf-8') as f:
        f.write(f"""\
---
task_categories:
- image-to-text
language:
- bn
- en
tags:
- ocr
- nid
- bangla
---

# NID OCR Extended Dataset

Built from [{source_dataset}](https://huggingface.co/datasets/{source_dataset})
with {confusion_n:,} additional confusion-pair training images.

| Split      | Samples |
|:-----------|--------:|
| train      | {train_n:,} |
| validation | {val_n:,} |

## Sources
- **{source_dataset}** — original synthetic + shamadhan real data
- **synthetic_confusion** — {confusion_n:,} confusion-pair images (train only)

## Columns
| Column | Type | Description |
|:---|:---|:---|
| `image` | Image | Cropped NID field (RGB) |
| `text` | string | OCR ground-truth label |
| `class_name` | string | NID field type or `confusion_training` |
| `source` | string | `synthetic`, `shamadhan`, or `synthetic_confusion` |

## Load
```python
from datasets import load_dataset
ds = load_dataset('parquet', data_files={{
    'train':      '{output_dir}/train/*.parquet',
    'validation': '{output_dir}/validation/*.parquet',
}})
```
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dataset', default='kavinh07/ocr_dataset_shamadhan_synth_30k_p2')
    parser.add_argument('--confusion_count', type=int, default=300,
                        help='Images per confusable char pair')
    parser.add_argument('--font_size',       type=int,   default=11)
    parser.add_argument('--blur',            type=float, default=0.3)
    parser.add_argument('--work_dir',        default='/app/confusion_work',
                        help='Directory for generated confusion images')
    parser.add_argument('--output_dir',      default='/app/hf_extended',
                        help='Output directory for final Parquet dataset')
    parser.add_argument('--shard_size',      type=int,   default=500)
    parser.add_argument('--push_to_hub',     default=None,
                        help='HuggingFace repo id to push to (e.g. your-org/dataset-name)')
    parser.add_argument('--hf_token',        default=None,
                        help='HuggingFace write token (or set HF_TOKEN env var)')
    parser.add_argument('--reset',           action='store_true',
                        help='Regenerate confusion images from scratch')
    args = parser.parse_args()

    os.makedirs(args.work_dir,   exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    token = args.hf_token or os.environ.get('HF_TOKEN')

    # ── Step 1: Load source dataset ──────────────────────────────────────────
    from datasets import load_dataset, concatenate_datasets
    print(f"\nLoading {args.source_dataset} ...")
    source = load_dataset(args.source_dataset, token=token)
    print(f"  train : {len(source['train']):,}")
    print(f"  val   : {len(source['validation']):,}")

    # ── Step 2: Generate confusion images ────────────────────────────────────
    plan = generate_confusion_images(
        work_dir=args.work_dir,
        confusion_count=args.confusion_count,
        font_size=args.font_size,
        blur=args.blur,
        reset=args.reset,
    )
    completed_plan = [p for p in plan if os.path.exists(p['image_path'])]
    print(f"  {len(completed_plan):,} confusion images ready")

    # ── Step 3: Build confusion Dataset ──────────────────────────────────────
    confusion_ds = build_confusion_dataset(completed_plan)
    print(f"  confusion dataset: {len(confusion_ds):,} samples")

    # ── Step 4: Merge — confusion goes to train only ─────────────────────────
    print("\nMerging datasets...")
    features = _hf_features()

    # Cast to ensure consistent features before concat
    source_train = source['train'].cast(features)
    source_val   = source['validation'].cast(features)
    confusion_ds = confusion_ds.cast(features)

    train_combined = concatenate_datasets([source_train, confusion_ds])
    train_combined = train_combined.shuffle(seed=42)

    print(f"  final train : {len(train_combined):,}  "
          f"({len(source_train):,} original + {len(confusion_ds):,} confusion)")
    print(f"  final val   : {len(source_val):,}  (unchanged)")

    # ── Step 5: Write sharded Parquet ────────────────────────────────────────
    write_sharded_parquet(
        train_combined,
        os.path.join(args.output_dir, 'train'),
        args.shard_size,
        'train',
    )
    write_sharded_parquet(
        source_val,
        os.path.join(args.output_dir, 'validation'),
        args.shard_size,
        'validation',
    )
    write_readme(
        args.output_dir,
        args.source_dataset,
        len(train_combined),
        len(source_val),
        len(confusion_ds),
    )

    print(f"\nDataset ready in {args.output_dir}/")

    # ── Step 6: Push to Hub (optional) ───────────────────────────────────────
    if args.push_to_hub:
        from datasets import load_dataset as ld
        print(f"\nPushing to {args.push_to_hub} ...")
        final_ds = ld('parquet', data_files={
            'train':      os.path.join(args.output_dir, 'train', '*.parquet'),
            'validation': os.path.join(args.output_dir, 'validation', '*.parquet'),
        })
        final_ds.push_to_hub(args.push_to_hub, token=token)
        print(f"  Pushed to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == '__main__':
    main()
