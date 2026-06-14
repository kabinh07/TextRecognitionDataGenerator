"""
Generates 10k synthetic Bangla NID address images from postal_codes.csv.
English OCR data (names, DOB, NID numbers) comes from the existing
generate.py pipeline using shamadhan + list data.

Degradation strategy: lower pixel resolution via smaller font sizes,
not blur. Text is always black. Three resolution tiers:
  25% normal   — font 32-40px  (clean scan)
  40% medium   — font 22-31px  (average scan)
  35% low_res  — font 14-21px  (poor scan / low DPI)
"""

import os
import json
import random
import argparse
from tqdm import tqdm

from trdg.data_generator import FakeTextDataGenerator
from trdg.utils import load_fonts

from address_generator import AddressGenerator


# ─── Quality profiles ─────────────────────────────────────────────────────────
# (name, weight, font_size_min, font_size_max, blur_max, bg_type, skew_angle)
# Text color is always black (#282828).
# blur_max kept at 1 for minimal anti-aliasing smoothing only.
# bg_type: 0=gaussian noise

_PROFILES = [
    ('normal',  25, 32, 40, 0, 0, 2),
    ('medium',  40, 22, 31, 1, 0, 2),
    ('low_res', 35, 14, 21, 1, 0, 3),
]
_PROFILE_WEIGHTS = [p[1] for p in _PROFILES]


def _pick_profile():
    profile = random.choices(_PROFILES, weights=_PROFILE_WEIGHTS, k=1)[0]
    name, _, fs_min, fs_max, blur_max, bg_type, skew = profile
    font_size = random.randint(fs_min, fs_max)
    return name, font_size, blur_max, bg_type, skew


def _save_json(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _generate_images(plan, fonts, image_dir):
    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    done = len(plan) - len(todo)
    if done:
        print(f"  {done}/{len(plan)} already done, resuming...")
    if not todo:
        return

    for item in tqdm(todo, desc="Generating Bangla address images"):
        _, font_size, blur_max, bg_type, skew = _pick_profile()

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
                background_type=bg_type,
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
                # Contrast check failed — use larger font as fallback
                font_size = min(font_size + 4, 32)
                blur_max = 0
            attempts += 1

        if img is None:
            print(f"  WARNING: skipped idx={item['idx']} after {attempts} retries")
            continue

        os.makedirs(os.path.dirname(item['image_path']), exist_ok=True)
        os.makedirs(os.path.dirname(item['label_path']), exist_ok=True)
        img.save(item['image_path'])
        with open(item['label_path'], 'w', encoding='utf-8') as f:
            f.write(item['text'])


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic Bangla NID address images from postal_codes.csv'
    )
    parser.add_argument(
        '--text_count', type=int, default=10000,
        help='Number of Bangla address images to generate (default: 10000)'
    )
    parser.add_argument('--output_dir', type=str, default='output_address')
    parser.add_argument('--csv_path', type=str, default='data/postal_codes.csv')
    parser.add_argument('--reset', action='store_true', help='Restart from scratch')
    parser.add_argument('--show_samples', action='store_true',
                        help='Print sample addresses and exit without generating images')
    args = parser.parse_args()

    print(f"Loading postal data from {args.csv_path} ...")
    addr_gen = AddressGenerator(args.csv_path)
    print(f"  Village pool: {len(addr_gen._bn_villages):,} synthetic Bangla names")

    if args.show_samples:
        addr_gen.show_samples(n=10)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'trdg', 'images'
    )

    print(f"\n── BN ({args.text_count:,} images) ───────────────────────────")
    fonts = load_fonts('bn')
    plan_path = os.path.join(args.output_dir, '.plan_bn.json')

    if not args.reset and os.path.exists(plan_path):
        print(f"  Resuming from {plan_path}")
        plan = _load_json(plan_path)
    else:
        print(f"  Building plan for {args.text_count:,} images...")
        plan = [
            {
                'idx': idx,
                'text': addr_gen.generate_bn(),
                'image_path': os.path.join(args.output_dir, 'images', f'bn_addr_{idx}.png'),
                'label_path': os.path.join(args.output_dir, 'labels', f'bn_addr_{idx}.txt'),
            }
            for idx in range(args.text_count)
        ]
        _save_json(plan, plan_path)
        print(f"  Plan saved → {plan_path}")

    _generate_images(plan, fonts, image_dir)

    total = len([
        f for f in os.listdir(os.path.join(args.output_dir, 'images'))
        if f.endswith('.png')
    ]) if os.path.isdir(os.path.join(args.output_dir, 'images')) else 0
    print(f"\nDone. {total:,} images in {args.output_dir}/images/")


if __name__ == '__main__':
    main()
