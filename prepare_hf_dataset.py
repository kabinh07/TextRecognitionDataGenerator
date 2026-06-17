"""
Resumable pipeline: synthetic output/ + shamadhan real data → sharded HF Parquet dataset.

State is persisted after every shard. If Docker stops mid-run, just re-run the same
command — already-written shards are skipped automatically.

Layout written:
  hf_dataset/
    .progress.json          ← tracks completed shards
    .samples_train.json     ← train sample metadata (paths + labels, no images)
    .samples_val.json       ← val sample metadata
    train/
      shard-0000.parquet
      shard-0001.parquet
      ...
    validation/
      shard-0000.parquet
      ...
    README.md               ← HF dataset card

Load after completion:
  from datasets import load_dataset
  ds = load_dataset('parquet', data_files={
      'train':      'hf_dataset/train/*.parquet',
      'validation': 'hf_dataset/validation/*.parquet',
  })

Upload:
  ds.push_to_hub('your-org/nid-ocr-dataset')

Usage:
  python prepare_hf_dataset.py
  python prepare_hf_dataset.py --shard_size 500 --dataset_dir hf_dataset
  python prepare_hf_dataset.py --reset   # discard state and start over
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

# ── column schema ────────────────────────────────────────────────────────────

try:
    from datasets import Dataset, Features
    from datasets import Image as HFImage
    from datasets import Value
    HF_FEATURES = Features({
        'image':      HFImage(),
        'text':       Value('string'),
        'class_name': Value('string'),
        'source':     Value('string'),
    })
except ImportError:
    HF_FEATURES = None  # checked at runtime


# ── state helpers (atomic write) ─────────────────────────────────────────────

def _save_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def state_path(dataset_dir):
    return os.path.join(dataset_dir, '.progress.json')


def samples_path(dataset_dir, split):
    return os.path.join(dataset_dir, f'.samples_{split}.json')


def load_state(dataset_dir):
    p = state_path(dataset_dir)
    if os.path.exists(p):
        return _load_json(p)
    return None


def save_state(state, dataset_dir):
    _save_json(state, state_path(dataset_dir))


# ── collectors ───────────────────────────────────────────────────────────────

def collect_synthetic(output_dir: str):
    root = Path(output_dir)
    samples = []
    for img_path in sorted(root.rglob('*.png')):
        if 'mask' in img_path.name:
            continue
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        label = label_path.read_text(encoding='utf-8').strip()
        if not label:
            continue
        parts = img_path.parts
        class_name = parts[-3] if len(parts) >= 3 and parts[-2] == 'images' else 'synthetic'
        samples.append({
            'image_path': str(img_path.resolve()),
            'text':       label,
            'class_name': class_name,
            'source':     'synthetic',
        })
    return samples


def collect_shamadhan(shamadhan_dir: str):
    csv_file = os.path.join(shamadhan_dir, 'dataset.csv')
    img_dir  = os.path.join(shamadhan_dir, 'cropped_images')
    samples  = []
    with open(csv_file, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('disabled', '').strip():
                continue
            label = row.get('corrected_labels', '').strip() or row.get('labels', '').strip()
            if not label:
                continue
            img_path = os.path.abspath(os.path.join(img_dir, row['image_name'] + '.png'))
            if not os.path.exists(img_path):
                continue
            parts = row['image_name'].split('_', 1)
            class_name = parts[1] if len(parts) == 2 else 'unknown'
            samples.append({
                'image_path': img_path,
                'text':       label,
                'class_name': class_name,
                'source':     'shamadhan',
            })
    return samples


# ── shard writer (atomic) ────────────────────────────────────────────────────

def write_shard(sample_metas: list, shard_path: str, desc: str):
    """Load images from disk, build a Dataset, write Parquet. Atomic via temp file."""
    from PIL import Image as PILImage

    tmp = shard_path + '.tmp'
    skipped = 0

    def gen():
        nonlocal skipped
        for s in sample_metas:
            try:
                img = PILImage.open(s['image_path']).convert('RGB')
                yield {
                    'image':      img,
                    'text':       s['text'],
                    'class_name': s['class_name'],
                    'source':     s['source'],
                }
            except Exception as e:
                skipped += 1
                print(f'\n  skip {s["image_path"]}: {e}')

    ds = Dataset.from_generator(gen, features=HF_FEATURES)
    ds.to_parquet(tmp)
    os.replace(tmp, shard_path)

    if skipped:
        print(f'  {desc}: {skipped} images skipped (corrupt/missing)')


# ── dataset card ─────────────────────────────────────────────────────────────

def write_readme(dataset_dir, train_n, val_n):
    card = os.path.join(dataset_dir, 'README.md')
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

# NID OCR Dataset

| Split      | Samples |
|:-----------|--------:|
| train      | {train_n:,} |
| validation | {val_n:,} |

## Sources
- **Synthetic** — TextRecognitionDataGenerator (token-balanced)
- **Shamadhan** — real cropped NID field images (reviewed labels)

## Columns
| Column | Type | Description |
|:---|:---|:---|
| `image` | Image | Cropped NID field (RGB) |
| `text` | string | OCR ground-truth label |
| `class_name` | string | NID field type (e.g. `en_name`, `address_line_00`) |
| `source` | string | `synthetic` or `shamadhan` |

## Load
```python
from datasets import load_dataset
ds = load_dataset('parquet', data_files={{
    'train':      'hf_dataset/train/*.parquet',
    'validation': 'hf_dataset/validation/*.parquet',
}})
```

## Upload
```python
ds.push_to_hub('your-org/nid-ocr-dataset')
```
""")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',    default='output',
                        help='generate.py output directory with synthetic images')
    parser.add_argument('--shamadhan_dir',
                        default='/app/ocr_dataset_shamadhan',
                        help='Shamadhan dataset root (contains dataset.csv + cropped_images/)')
    parser.add_argument('--dataset_dir',   default='hf_dataset',
                        help='Destination directory for the sharded dataset')
    parser.add_argument('--val_split',     type=float, default=0.2,
                        help='Fraction of synthetic data held out for validation (default 0.2)')
    parser.add_argument('--shard_size',    type=int, default=500,
                        help='Number of images per Parquet shard (default 500)')
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument('--reset',         action='store_true',
                        help='Ignore existing state and start over')
    args = parser.parse_args()

    if HF_FEATURES is None:
        raise ImportError('pip install datasets')

    os.makedirs(args.dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dataset_dir, 'train'),      exist_ok=True)
    os.makedirs(os.path.join(args.dataset_dir, 'validation'), exist_ok=True)

    random.seed(args.seed)

    # ── Phase 1: Resolve sample lists (resumable) ────────────────────────────
    state = None if args.reset else load_state(args.dataset_dir)

    if state is None or not state.get('samples_saved'):
        print('Phase 1 — collecting and splitting samples...')

        synthetic = collect_synthetic(args.output_dir)
        print(f'  synthetic : {len(synthetic):,}')
        shamadhan = collect_shamadhan(args.shamadhan_dir)
        print(f'  shamadhan : {len(shamadhan):,}')

        # Confusion-pair images must go to train only (they're targeted hard examples,
        # not representative of the distribution — putting them in val skews metrics).
        confusion = [
            s for s in synthetic
            if Path(s['image_path']).name.startswith('bn_conf_')
        ]
        regular = [
            s for s in synthetic
            if not Path(s['image_path']).name.startswith('bn_conf_')
        ]
        print(f'  confusion (train-only) : {len(confusion):,}')

        random.shuffle(regular)
        val_n      = max(1, int(len(regular) * args.val_split))
        val_samp   = regular[:val_n]
        train_samp = regular[val_n:] + shamadhan + confusion  # real + confusion → train only
        random.shuffle(train_samp)

        _save_json(train_samp, samples_path(args.dataset_dir, 'train'))
        _save_json(val_samp,   samples_path(args.dataset_dir, 'validation'))

        import math
        state = {
            'samples_saved':  True,
            'shard_size':     args.shard_size,
            'train': {
                'total':        len(train_samp),
                'total_shards': math.ceil(len(train_samp) / args.shard_size),
                'done_shards':  [],
            },
            'validation': {
                'total':        len(val_samp),
                'total_shards': math.ceil(len(val_samp) / args.shard_size),
                'done_shards':  [],
            },
        }
        save_state(state, args.dataset_dir)
        print(f'\n  train : {len(train_samp):,}  '
              f'({len(regular)-val_n:,} synth + {len(shamadhan):,} shamadhan '
              f'+ {len(confusion):,} confusion)')
        print(f'  val   : {len(val_samp):,}  (synthetic only)')
    else:
        print('Phase 1 — resuming from saved sample lists.')
        train_samp = _load_json(samples_path(args.dataset_dir, 'train'))
        val_samp   = _load_json(samples_path(args.dataset_dir, 'validation'))
        print(f'  train : {len(train_samp):,}   val : {len(val_samp):,}')

    # ── Phase 2: Write shards (resumable per shard) ──────────────────────────
    for split, samples in [('train', train_samp), ('validation', val_samp)]:
        split_state = state[split]
        done        = set(split_state['done_shards'])
        total_sh    = split_state['total_shards']
        sz          = state['shard_size']
        split_dir   = os.path.join(args.dataset_dir, split)

        remaining = [i for i in range(total_sh) if i not in done]
        if not remaining:
            print(f'\n{split}: all {total_sh} shards already done, skipping.')
            continue

        print(f'\nPhase 2 — {split}: {len(remaining)}/{total_sh} shards to write')

        for shard_idx in remaining:
            start = shard_idx * sz
            chunk = samples[start: start + sz]
            shard_file = os.path.join(
                split_dir, f'shard-{shard_idx:04d}-of-{total_sh:04d}.parquet'
            )
            desc = f'{split} shard {shard_idx+1}/{total_sh}'
            print(f'  Writing {desc}  ({len(chunk)} images) ...')
            write_shard(chunk, shard_file, desc)

            # Mark done and persist immediately
            split_state['done_shards'].append(shard_idx)
            save_state(state, args.dataset_dir)

        print(f'  {split} complete.')

    # ── Phase 3: Dataset card ────────────────────────────────────────────────
    write_readme(args.dataset_dir, state['train']['total'], state['validation']['total'])

    print(f'\nDataset ready in {args.dataset_dir}/')
    print('To load:')
    print("  from datasets import load_dataset")
    print(f"  ds = load_dataset('parquet', data_files={{")
    print(f"      'train':      '{args.dataset_dir}/train/*.parquet',")
    print(f"      'validation': '{args.dataset_dir}/validation/*.parquet',")
    print(f"  }})")


if __name__ == '__main__':
    main()
