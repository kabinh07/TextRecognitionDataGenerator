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
                        double_line_prob, output_dir, image_dir, reset):
    print(f"\n── Shamadhan {lang.upper()} ({text_count} images) ─────────────────")
    fonts = load_fonts(lang)
    plan_path = os.path.join(output_dir, f'.plan_{lang}.json')

    pool_size = text_count * 2 if (lang == 'bn' and double_line_prob > 0) else text_count

    # Class-balanced pool with token balancing
    text_to_class = {}
    for cn, texts in class_dict.items():
        for t in texts:
            if t and t not in text_to_class:
                text_to_class[t] = cn
    selected = balance_tokens(list(text_to_class.keys()), pool_size)
    balanced = [(t, text_to_class[t]) for t in selected]

    # Bangla double-line pairing
    if lang == 'bn' and double_line_prob > 0:
        paired = []
        i = 0
        while len(paired) < text_count and i < len(balanced):
            if i + 1 < len(balanced) and random.random() < double_line_prob:
                t1, c1 = balanced[i]
                t2, _ = balanced[i + 1]
                paired.append((f"{t1}\n{t2}", c1))
                i += 2
            else:
                paired.append(balanced[i])
                i += 1
        balanced = paired[:text_count]

    if not reset and os.path.exists(plan_path):
        print(f"  Resuming from {plan_path}")
        plan = _load_json(plan_path)
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

def _generate_addresses(address_count, csv_path, output_dir, image_dir, reset):
    print(f"\n── Bangla Addresses ({address_count} images) ──────────────────────")
    addr_gen = AddressGenerator(csv_path)
    print(f"  Village pool: {len(addr_gen._bn_villages):,}")
    fonts = load_fonts('bn')
    plan_path = os.path.join(output_dir, '.plan_addr.json')

    if not reset and os.path.exists(plan_path):
        print(f"  Resuming from {plan_path}")
        plan = _load_json(plan_path)
    else:
        plan = [
            {
                'idx': idx,
                'text': addr_gen.generate_bn(),
                'image_path': os.path.join(output_dir, 'images', f'bn_addr_{idx}.png'),
                'label_path': os.path.join(output_dir, 'labels', f'bn_addr_{idx}.txt'),
            }
            for idx in range(address_count)
        ]
        _save_json(plan, plan_path)

    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    print(f"  {len(plan) - len(todo)} done, {len(todo)} remaining")

    for item in tqdm(todo, desc='Bangla addresses'):
        font_size, blur_max, skew = _pick_addr_profile()
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
                font_size = min(font_size + 4, 32)
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
    parser.add_argument('--double_line_prob', type=float, default=0.3,
                        help='Probability of pairing two Bangla shamadhan texts on one image')
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
    candidates = [
        os.path.join(data_dir, 'data', 'nid_data_token_balanced.json'),
        os.path.join(data_dir, 'data', 'nid_data_texts.json'),
    ]
    json_data = None
    for path in candidates:
        if os.path.exists(path):
            print(f"Loading shamadhan data: {path}")
            json_data = _load_json(path)
            break
    if json_data is None:
        print("ERROR: No shamadhan JSON found (nid_data_token_balanced.json / nid_data_texts.json)")
        sys.exit(1)

    # Group texts by detected language
    texts_by_lang = {'bn': {}, 'en': {}}
    for class_name, texts in json_data.items():
        if class_name == 'name_en':
            texts = list(texts) + [t.upper() for t in texts if isinstance(t, str)]
        for t in texts:
            if isinstance(t, str) and t.strip():
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
                double_line_prob=args.double_line_prob,
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
