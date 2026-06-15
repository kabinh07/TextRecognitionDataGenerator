"""
generate_combined.py

Single container replacement for text_generator + address_generator.
Sources:
  - Shamadhan NID JSON (data/nid_data_token_balanced.json or nid_data_texts.json)
    → Bangla + English: names, dates, NID numbers, misc fields
  - postal_codes.csv + AddressGenerator
    → Bangla address fields only

All images use background_type=3 (trdg/images/ paper/document textures).
No list_data used.
"""

import os
import csv
import json
import random
import math
import re
import collections
import argparse
import sys
from tqdm import tqdm

from trdg.generators import GeneratorFromStrings
from trdg.data_generator import FakeTextDataGenerator
from trdg.utils import load_fonts

from address_generator import AddressGenerator


# ── Shared helpers (from generate.py) ────────────────────────────────────────

def _save_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def detect_text_language(text):
    if not text or not isinstance(text, str):
        return 'en'
    bn = sum(1 for c in text if 'ঀ' <= c <= '৿')
    en = sum(1 for c in text if c.isascii() and c.isalpha())
    return 'bn' if bn > en else 'en'


def balance_tokens(texts, target_count):
    if not texts:
        return []
    texts = list(set(t for t in texts if isinstance(t, str) and t.strip()))
    if len(texts) <= target_count:
        return texts

    token_counts = collections.Counter()
    for text in tqdm(texts, desc="Token frequencies"):
        tokens = re.findall(r'[ঀ-৿]+|[a-zA-Z0-9]+', text)
        token_counts.update(tokens)

    weights = []
    for text in tqdm(texts, desc="Sampling weights"):
        tokens = re.findall(r'[ঀ-৿]+|[a-zA-Z0-9]+', text)
        score = sum(math.sqrt(token_counts[t]) for t in tokens) if tokens else 0
        weights.append(1.0 / score if score > 0 else 1.0)

    print(f"Sampling {target_count} from {len(texts)}...")
    scored = [
        (math.log(random.random()) / w, t)
        for w, t in zip(weights, texts) if w > 0
    ]
    scored.sort(reverse=True)
    return [t for _, t in scored[:target_count]]


# ── Address quality profiles ──────────────────────────────────────────────────
# (name, weight, font_size_min, font_size_max, blur_max, skew_angle)
# Text always black; background_type=3 (trdg/images/).

_ADDR_PROFILES = [
    ('normal',  25, 32, 40, 0, 2),
    ('medium',  40, 22, 31, 1, 2),
    ('low_res', 35, 14, 21, 1, 3),
]
_ADDR_WEIGHTS = [p[1] for p in _ADDR_PROFILES]


def _pick_addr_profile():
    p = random.choices(_ADDR_PROFILES, weights=_ADDR_WEIGHTS, k=1)[0]
    _, _, fs_min, fs_max, blur_max, skew = p
    return random.randint(fs_min, fs_max), blur_max, skew


# ── Shamadhan generation ──────────────────────────────────────────────────────

def _generate_shamadhan(lang, class_dict, text_count, font_size, blur,
                        output_dir, image_dir, reset):
    print(f"\n── Shamadhan {lang.upper()} ({text_count} images) ─────────────────")
    fonts = load_fonts(lang)
    plan_path = os.path.join(output_dir, f'.plan_{lang}.json')

    # Class-balanced pool with token balancing (single-line only)
    text_to_class = {}
    for cn, texts in class_dict.items():
        for t in texts:
            if t and t not in text_to_class:
                text_to_class[t] = cn
    selected = balance_tokens(list(text_to_class.keys()), text_count)
    balanced = [(t, text_to_class[t]) for t in selected]

    if not reset and os.path.exists(plan_path):
        print(f"  Resuming from {plan_path}")
        plan = _load_json(plan_path)
        # Backfill: strip any newlines cached in older plan files
        for p in plan:
            p['text'] = ' '.join(p['text'].split())
    else:
        plan = [
            {
                'idx': idx,
                'text': text,
                'class_name': class_name,
                'image_path': os.path.join(output_dir, 'images', f'{lang}_img_{idx}.png'),
                'label_path': os.path.join(output_dir, 'labels', f'{lang}_img_{idx}.txt'),
            }
            for idx, (text, class_name) in enumerate(balanced)
        ]
        _save_json(plan, plan_path)

    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    print(f"  {len(plan) - len(todo)} done, {len(todo)} remaining")

    if not todo:
        return

    generator = GeneratorFromStrings(
        strings=[p['text'] for p in todo],
        count=len(todo),
        fonts=fonts,
        size=font_size,
        language=lang,
        skewing_angle=3,
        random_skew=True,
        distorsion_type=0,
        blur=blur,
        random_blur=True,
        fit=True,
        word_split=True,
        background_type=3,
        image_dir=image_dir,
        orientation=0,
    )

    for (img, lbl), item in tqdm(zip(generator, todo), total=len(todo),
                                  desc=f'Shamadhan {lang.upper()}'):
        if img is None:
            continue
        os.makedirs(os.path.dirname(item['image_path']), exist_ok=True)
        os.makedirs(os.path.dirname(item['label_path']), exist_ok=True)
        img.save(item['image_path'])
        with open(item['label_path'], 'w', encoding='utf-8') as f:
            f.write(lbl)


# ── Address generation ────────────────────────────────────────────────────────

def _generate_addresses(address_count, csv_path, output_dir, image_dir, reset,
                        extra_texts=None, font_size=None, blur=None):
    print(f"\n── Bangla Addresses ({address_count} images) ──────────────────────")
    addr_gen = AddressGenerator(csv_path)
    print(f"  Village pool: {len(addr_gen._bn_villages):,}")
    if extra_texts:
        print(f"  Real address pool: {len(extra_texts):,} texts from shamadhan CSV")
    fonts = load_fonts('bn')
    plan_path = os.path.join(output_dir, '.plan_addr.json')

    if not reset and os.path.exists(plan_path):
        print(f"  Resuming from {plan_path}")
        plan = _load_json(plan_path)
    else:
        real_pool = extra_texts or []
        plan = []
        for idx in range(address_count):
            # 30% real shamadhan addresses (if available), 70% synthetic
            if real_pool and random.random() < 0.3:
                text = random.choice(real_pool)
            else:
                text = addr_gen.generate_bn()
            plan.append({
                'idx': idx,
                'text': text,
                'image_path': os.path.join(output_dir, 'images', f'bn_addr_{idx}.png'),
                'label_path': os.path.join(output_dir, 'labels', f'bn_addr_{idx}.txt'),
            })
        _save_json(plan, plan_path)

    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    print(f"  {len(plan) - len(todo)} done, {len(todo)} remaining")

    for item in tqdm(todo, desc='Bangla addresses'):
        profile_size, profile_blur, skew = _pick_addr_profile()
        item_font_size = font_size if font_size is not None else profile_size
        blur_max = blur if blur is not None else profile_blur
        font = random.choice(fonts)
        img = None
        attempts = 0

        while img is None and attempts < 5:
            img = FakeTextDataGenerator.generate(
                index=item['idx'],
                text=item['text'],
                font=font,
                out_dir=None,
                size=item_font_size,
                extension=None,
                skewing_angle=skew,
                random_skew=True,
                blur=blur_max,
                random_blur=True,
                background_type=3,
                distorsion_type=0,
                distorsion_orientation=0,
                is_handwritten=False,
                name_format=0,
                width=-1,
                alignment=0,
                text_color='#282828',
                orientation=0,
                space_width=1.0,
                character_spacing=0,
                margins=(5, 5, 5, 5),
                fit=True,
                output_mask=False,
                word_split=True,
                image_dir=image_dir,
            )
            if img is None:
                item_font_size = max(item_font_size + 4, 32)
                blur_max = 0
            attempts += 1

        if img is None:
            print(f"  WARNING: skipped addr idx={item['idx']}")
            continue

        os.makedirs(os.path.dirname(item['image_path']), exist_ok=True)
        os.makedirs(os.path.dirname(item['label_path']), exist_ok=True)
        img.save(item['image_path'])
        with open(item['label_path'], 'w', encoding='utf-8') as f:
            f.write(item['text'])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate NID OCR images: shamadhan data + Bangla addresses'
    )
    parser.add_argument('--text_count', type=int, default=50,
                        help='Shamadhan images per language (bn + en each, default 50)')
    parser.add_argument('--address_count', type=int, default=50,
                        help='Bangla address images (default 50)')
    parser.add_argument('--font_size', type=int, default=32,
                        help='Font size for shamadhan images')
    parser.add_argument('--blur', type=float, default=0.8,
                        help='Max blur radius for shamadhan images (random 0..blur)')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--csv_path', type=str, default='data/postal_codes.csv')
    parser.add_argument('--reset', action='store_true',
                        help='Ignore saved plans and restart from scratch')
    parser.add_argument('--prepare_hf', action='store_true',
                        help='Run prepare_hf_dataset.py after generation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'trdg', 'images'
    )

    # ── Load shamadhan data ────────────────────────────────────────────────────
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_dir, 'data')

    # Build label_id → class_name map from label_mappings.csv if present
    label_map = {}
    mappings_path = os.path.join(data_path, 'label_mappings.csv')
    if os.path.exists(mappings_path):
        with open(mappings_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                label_map[row['label_id'].strip()] = row['label_name'].strip()
        print(f"Loaded {len(label_map)} label mappings from {mappings_path}")

    # Try CSV sources first (merged_reviewed_latest.csv → dataset.csv)
    texts_by_lang = {'bn': {}, 'en': {}}
    csv_candidates = [
        os.path.join(data_path, 'merged_reviewed_latest.csv'),
        os.path.join(data_path, 'dataset.csv'),
    ]
    # Address classes from shamadhan CSV go to address pipeline, not shamadhan
    _ADDRESS_CLASSES = {'address_full', 'address_line_xx'}

    loaded_csv = False
    shamadhan_addr_texts = []  # real address texts from CSV for address pipeline
    for csv_src in csv_candidates:
        if not os.path.exists(csv_src):
            continue
        print(f"Loading shamadhan data: {csv_src}")
        row_count = 0
        with open(csv_src, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('disabled', '').strip():
                    continue
                text = (row.get('corrected_labels') or '').strip()
                if not text:
                    continue
                label_id = row.get('labels', '').strip()
                class_name = label_map.get(label_id, f'label_{label_id}')
                # Address classes go to address pipeline (multi-line OK)
                if class_name in _ADDRESS_CLASSES:
                    shamadhan_addr_texts.append(text)
                    row_count += 1
                    continue
                # All other classes: enforce single-line
                text = ' '.join(text.split())
                lang = detect_text_language(text)
                if class_name == 'en_name':
                    texts_by_lang['en'].setdefault(class_name, []).extend(
                        [text, text.upper()]
                    )
                else:
                    texts_by_lang[lang].setdefault(class_name, []).append(text)
                row_count += 1
        if row_count:
            print(f"  {row_count:,} usable rows loaded "
                  f"({len(shamadhan_addr_texts):,} address texts routed to address pipeline)")
            loaded_csv = True
            break

    # Fallback: try legacy JSON sources
    if not loaded_csv:
        json_candidates = [
            os.path.join(data_path, 'nid_data_token_balanced.json'),
            os.path.join(data_path, 'nid_data_texts.json'),
        ]
        json_data = None
        for path in json_candidates:
            if os.path.exists(path):
                print(f"Loading shamadhan data: {path}")
                json_data = _load_json(path)
                break
        if json_data is None:
            print("ERROR: No shamadhan data found. Files in data/:")
            if os.path.isdir(data_path):
                for f in sorted(os.listdir(data_path)):
                    fpath = os.path.join(data_path, f)
                    size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
                    print(f"  {f}  ({size:,} bytes)")
            else:
                print(f"  data/ directory not found at {data_path}")
            sys.exit(1)
        for class_name, texts in json_data.items():
            if class_name == 'name_en':
                texts = list(texts) + [t.upper() for t in texts if isinstance(t, str)]
            for t in texts:
                if isinstance(t, str) and t.strip():
                    t = ' '.join(t.split())  # enforce single-line
                    lang = detect_text_language(t)
                    texts_by_lang[lang].setdefault(class_name, []).append(t)

    print("\nShamadhan class distribution:")
    for lang in ['bn', 'en']:
        if texts_by_lang[lang]:
            total = sum(len(v) for v in texts_by_lang[lang].values())
            print(f"  {lang.upper()}: {len(texts_by_lang[lang])} classes, {total:,} texts")

    # ── Generate shamadhan images ──────────────────────────────────────────────
    for lang in ['bn', 'en']:
        if texts_by_lang[lang]:
            _generate_shamadhan(
                lang=lang,
                class_dict=texts_by_lang[lang],
                text_count=args.text_count,
                font_size=args.font_size,
                blur=args.blur,
                output_dir=args.output_dir,
                image_dir=image_dir,
                reset=args.reset,
            )

    # ── Generate address images ────────────────────────────────────────────────
    _generate_addresses(
        address_count=args.address_count,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        image_dir=image_dir,
        reset=args.reset,
        extra_texts=shamadhan_addr_texts if loaded_csv else None,
        font_size=args.font_size,
        blur=args.blur,
    )

    # ── Analytics ─────────────────────────────────────────────────────────────
    try:
        from dataset_analytics import run_analytics
        run_analytics(
            output_dir=args.output_dir,
            report_path=os.path.join(args.output_dir, 'dataset_analytics.md'),
        )
    except Exception as e:
        print(f"Analytics skipped: {e}")

    if args.prepare_hf:
        import subprocess
        subprocess.run([sys.executable, 'prepare_hf_dataset.py'], check=True)

    img_dir = os.path.join(args.output_dir, 'images')
    total = len([f for f in os.listdir(img_dir) if f.endswith('.png')]) if os.path.isdir(img_dir) else 0
    print(f"\nDone. {total:,} total images in {args.output_dir}/")


if __name__ == '__main__':
    main()
