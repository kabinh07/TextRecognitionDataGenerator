"""
generate_english_lowinktrain.py

10 000-sample English OCR training set with near-faded (low-ink) augmentation.
Two text categories:
  - names:  "FirstName (LastName)"  — 60 % of samples
  - nid:    English-digit NID numbers (10 / 13 digits, with/without spaces) — 40 %

Every image gets the "almost faded" treatment:
  • text colour: very light gray (#888–#cc range)
  • PIL noise + blur + JPEG compression
  • augraphy LowInkRandomLines + BadPhotoCopy + Faxify (all at high probability)

Output:
  {output_dir}/images/   — PNG images
  {output_dir}/labels/   — matching .txt labels
  {output_dir}/hf/       — sharded HF Parquet dataset (train split only)

Usage:
  python generate_english_lowinktrain.py --output_dir /app/en_lowinktrain_output --total 10000
  python generate_english_lowinktrain.py --output_dir /app/en_lowinktrain_output --total 10000 --reset
  # push to hub:
  python generate_english_lowinktrain.py ... --push_to_hub kavinh07/en-lowinktrain-ocr
"""

import argparse
import io
import math
import os
import random
import string
import sys

import numpy as np
from PIL import Image as PILImage, ImageFilter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_combined import _apply_augraphy, _load_json, _maybe_downscale, _save_json
from trdg.data_generator import FakeTextDataGenerator
from trdg.utils import load_fonts


# ── Corpora ───────────────────────────────────────────────────────────────────

_FIRST_NAMES = [
    # Male
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan",
    "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry",
    "Justin", "Scott", "Brandon", "Benjamin", "Samuel", "Raymond", "Gregory",
    "Frank", "Alexander", "Patrick", "Jack", "Dennis", "Jerry",
    "Tyler", "Aaron", "Henry", "Jose", "Adam", "Douglas", "Nathan",
    "Peter", "Zachary", "Kyle", "Walter", "Harold", "Jeremy", "Ethan",
    "Carl", "Keith", "Roger", "Gerald", "Christian", "Terry", "Sean",
    "Arthur", "Austin", "Noah", "Lawrence", "Jesse", "Joe", "Bryan",
    "Billy", "Jordan", "Albert", "Dylan", "Bruce", "Willie", "Gabriel",
    "Alan", "Juan", "Logan", "Wayne", "Ralph", "Roy", "Eugene",
    "Randy", "Vincent", "Russell", "Louis", "Philip", "Bobby", "Johnny",
    "Bradley", "Harry", "Fred", "Joe",
    # Female
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Susan", "Jessica",
    "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Dorothy", "Kimberly", "Emily", "Donna", "Michelle",
    "Carol", "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca",
    "Sharon", "Laura", "Cynthia", "Kathleen", "Amy", "Angela", "Shirley",
    "Anna", "Brenda", "Pamela", "Emma", "Nicole", "Helen", "Samantha",
    "Katherine", "Christine", "Debra", "Rachel", "Carolyn", "Janet",
    "Catherine", "Maria", "Heather", "Diane", "Julie", "Joyce", "Victoria",
    "Kelly", "Christina", "Lauren", "Joan", "Evelyn", "Olivia", "Judith",
    "Megan", "Cheryl", "Martha", "Andrea", "Frances", "Hannah", "Jacqueline",
    "Ann", "Gloria", "Jean", "Kathryn", "Alice", "Teresa", "Sara",
    "Janice", "Doris", "Madison", "Julia", "Grace", "Judy", "Abigail",
    "Marie", "Denise", "Beverly", "Amber", "Theresa", "Marilyn", "Danielle",
    "Diana", "Brittany", "Natalie", "Sophia", "Rose", "Isabella", "Alexis",
    "Kayla", "Charlotte", "Avery", "Zoey", "Leah", "Peyton", "Audrey",
]

_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Turner", "Phillips", "Evans", "Edwards", "Collins",
    "Stewart", "Morris", "Morales", "Murphy", "Cook", "Rogers", "Gutierrez",
    "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey", "Reed", "Kelly",
    "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson", "Watson",
    "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
    "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel",
    "Myers", "Long", "Ross", "Foster", "Jimenez", "Powell", "Jenkins",
    "Perry", "Russell", "Sullivan", "Bell", "Coleman", "Butler", "Henderson",
    "Barnes", "Gonzales", "Fisher", "Vasquez", "Simmons", "Romero", "Jordan",
    "Patterson", "Alexander", "Hamilton", "Graham", "Reynolds", "Griffin",
    "Wallace", "Moreno", "West", "Cole", "Hayes", "Bryant", "Herrera",
    "Gibson", "Ellis", "Tran", "Medina", "Aguilar", "Stevens", "Murray",
    "Ford", "Castro", "Marshall", "Owens", "Harrison", "Fernandez", "Mcdonald",
    "Woods", "Washington", "Kennedy", "Wells", "Vargas", "Henry", "Chen",
    "Freeman", "Webb", "Tucker", "Guzman", "Burns", "Crawford", "Olson",
    "Simpson", "Porter", "Hunter", "Gordon", "Mendez", "Silva", "Shaw",
    "Snyder", "Mason", "Dixon", "Munoz", "Rose", "Black", "Holmes",
    "Knight", "Stone", "Spencer", "Bowman", "Lane", "Hawkins", "Perkins",
    "Obrien", "George", "Weaver", "Lambert", "Hicks", "Frazier", "Reyes",
    "Cross", "Bowers", "Higgins", "Warren", "Pierce", "Flowers", "Welch",
]

# NID format templates: (total_digits, group_sizes or None for solid)
_NID_FORMATS = [
    (10, None),           # 1234567890
    (13, None),           # 1234567890123
    (10, (3, 3, 4)),      # 123 456 7890
    (13, (4, 5, 4)),      # 1234 56789 0123
    (17, None),           # 17-digit
    (17, (5, 6, 6)),      # 12345 678901 234567
]


def _gen_nid_text():
    fmt = random.choice(_NID_FORMATS)
    total, groups = fmt
    digits = [str(random.randint(0, 9)) for _ in range(total)]
    # First digit non-zero for realism
    digits[0] = str(random.randint(1, 9))
    if groups is None:
        return ''.join(digits)
    out, i = [], 0
    for size in groups:
        out.append(''.join(digits[i:i + size]))
        i += size
    return ' '.join(out)


_MIDDLE_INITIAL_LETTERS = ['A', 'I'] + list(string.ascii_uppercase)
_MIDDLE_INITIAL_WEIGHTS = [8, 8] + [1] * 26  # boost A / I — underrepresented


def _gen_name_text():
    first = random.choice(_FIRST_NAMES)
    last  = random.choice(_LAST_NAMES)

    # Middle token: single initial (weighted toward A/I) / full name / none
    r = random.random()
    if r < 0.35:
        letter = random.choices(_MIDDLE_INITIAL_LETTERS, weights=_MIDDLE_INITIAL_WEIGHTS)[0]
        middle = f"{letter}."
    elif r < 0.55:
        middle = random.choice(_FIRST_NAMES)
    else:
        middle = None

    # Parenthesis around last name — only sometimes, not every sample
    paren = random.random() < 0.5
    last_part = f"({last})" if paren else last

    name = f"{first} {middle} {last_part}" if middle else f"{first} {last_part}"

    # Maximize CAPITALIZED variants
    if random.random() < 0.75:
        name = name.upper()

    return name


# ── Augmentation ──────────────────────────────────────────────────────────────

def _very_faded_color():
    """Return a very light gray — almost invisible text for low-ink simulation."""
    v = random.randint(0x88, 0xcc)   # 136–204 out of 255
    return f'#{v:02x}{v:02x}{v:02x}'


def _apply_heavy_low_ink(img):
    """PIL-level: strong noise + blur + very low JPEG quality."""
    arr = np.array(img.convert('RGB')).astype(np.float32)
    sigma = random.uniform(10, 25)
    arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    img = PILImage.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=random.randint(40, 65))
    buf.seek(0)
    return PILImage.open(buf).convert('RGB')


def _augraphy_result(result, fallback):
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (tuple, list)) and len(result) > 0 and isinstance(result[0], np.ndarray):
        return result[0]
    if isinstance(result, dict):
        return result.get('output', fallback)
    return fallback


def _apply_augraphy_heavy_low_ink(img):
    """Augraphy pipeline at max ink-starvation settings."""
    try:
        from augraphy import BadPhotoCopy, Faxify, LowInkRandomLines
    except ImportError:
        return img

    arr = np.array(img.convert('RGB'))
    augmentors = [
        (BadPhotoCopy, 0.75),
        (Faxify,       0.65),
    ]

    # LowInkRandomLines: count_range must stay small relative to the crop
    # height (single text line, ~40-70px) or the lines fully overwrite the
    # glyphs and the text collapses into an illegible horizontal stripe.
    try:
        low_ink_inst = LowInkRandomLines(count_range=(3, 8), noise_probability=0.4)
    except Exception as e:
        print(f"  LowInkRandomLines heavy params failed ({e}), using defaults")
        try:
            low_ink_inst = LowInkRandomLines()
        except Exception:
            low_ink_inst = None

    if low_ink_inst is not None:
        augmentors.insert(0, (lambda inst=low_ink_inst: inst, 0.6))

    for aug_cls, prob in augmentors:
        if random.random() < prob:
            try:
                result = aug_cls()(arr)
                out = _augraphy_result(result, arr)
                if isinstance(out, np.ndarray) and out.shape[:2] == arr.shape[:2]:
                    arr = out
            except Exception as e:
                print(f"  augraphy warn ({aug_cls}): {e}")

    return PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── Plan builder ──────────────────────────────────────────────────────────────

def _build_plan(total, out_img_dir, out_lbl_dir):
    name_count = int(total * 0.60)
    nid_count  = total - name_count
    items = []

    for i in range(name_count):
        text  = _gen_name_text()
        fname = f'en_name_{i:06d}'
        items.append({
            'text':       text,
            'kind':       'name',
            'image_path': os.path.join(out_img_dir, f'{fname}.png'),
            'label_path': os.path.join(out_lbl_dir, f'{fname}.txt'),
        })

    for i in range(nid_count):
        text  = _gen_nid_text()
        fname = f'en_nid_{i:06d}'
        items.append({
            'text':       text,
            'kind':       'nid',
            'image_path': os.path.join(out_img_dir, f'{fname}.png'),
            'label_path': os.path.join(out_lbl_dir, f'{fname}.txt'),
        })

    random.shuffle(items)
    return items


# ── Generation loop ───────────────────────────────────────────────────────────

def _generate(plan, en_fonts, image_dir, font_size):
    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    print(f"  {len(plan) - len(todo)} done, {len(todo)} remaining")

    for idx, item in enumerate(tqdm(todo, desc='  generating')):
        font       = random.choice(en_fonts)
        text_color = _very_faded_color()
        # Random bold 50 % of the time — bold samples are always uppercase
        # and get stretched horizontally (matches thick embossed NID print look)
        stroke = 1 if random.random() < 0.5 else 0
        text   = item['text'].upper() if stroke else item['text']

        img = None
        for _ in range(5):
            try:
                img = FakeTextDataGenerator.generate(
                    index=idx,
                    text=text,
                    font=font,
                    out_dir=None,
                    size=font_size,
                    extension=None,
                    skewing_angle=0,
                    random_skew=False,
                    blur=0,
                    random_blur=False,
                    background_type=3,   # real paper/doc textures — matches NID look
                    distorsion_type=0,
                    distorsion_orientation=0,
                    is_handwritten=False,
                    name_format=0,
                    width=-1,
                    alignment=1,
                    text_color=text_color,
                    orientation=0,
                    space_width=1.0,
                    character_spacing=0,
                    margins=(5, 5, 5, 5),
                    fit=True,
                    output_mask=False,
                    word_split=False,  # English — no Bangla diacritic split needed
                    image_dir=image_dir,
                    stroke_width=stroke,
                    stroke_fill=text_color,
                    image_mode='RGB',
                )
                if img is not None:
                    break
            except Exception:
                img = None

        if img is None:
            continue

        # Bold → stretch horizontally (thick embossed NID-print look)
        if stroke:
            w, h = img.size
            stretch = random.uniform(1.15, 1.35)
            img = img.resize((max(1, int(w * stretch)), h), PILImage.BILINEAR)

        # Heavy low-ink augmentation on every image
        img = _apply_augraphy_heavy_low_ink(img)
        img = _apply_heavy_low_ink(img)
        img = _maybe_downscale(img)

        os.makedirs(os.path.dirname(item['image_path']), exist_ok=True)
        os.makedirs(os.path.dirname(item['label_path']), exist_ok=True)
        img.save(item['image_path'])
        with open(item['label_path'], 'w', encoding='utf-8') as f:
            f.write(text)


# ── HF dataset builder ────────────────────────────────────────────────────────

def build_hf_dataset(plan, hf_dir, shard_size=500):
    try:
        from datasets import Dataset, Features, Value
        from datasets import Image as HFImage
    except ImportError:
        print("datasets not installed — skipping HF build")
        return

    features = Features({
        'image':      HFImage(),
        'text':       Value('string'),
        'kind':       Value('string'),   # 'name' or 'nid'
        'class_name': Value('string'),
        'source':     Value('string'),
    })

    done = [p for p in plan if os.path.exists(p['image_path'])]
    print(f"\nBuilding HF dataset from {len(done):,} images...")

    os.makedirs(os.path.join(hf_dir, 'train'), exist_ok=True)
    n_shards = math.ceil(len(done) / shard_size)

    for shard_i in tqdm(range(n_shards), desc='Shards'):
        shard_path = os.path.join(
            hf_dir, 'train',
            f'shard-{shard_i:04d}-of-{n_shards:04d}.parquet'
        )
        if os.path.exists(shard_path):
            continue
        chunk = done[shard_i * shard_size: (shard_i + 1) * shard_size]

        def gen(chunk=chunk):
            for item in chunk:
                try:
                    img = PILImage.open(item['image_path']).convert('RGB')
                    yield {
                        'image':      img,
                        'text':       item['text'],
                        'kind':       item['kind'],
                        'class_name': 'en_lowinktrain',
                        'source':     'synthetic_en_lowinktrain',
                    }
                except Exception as e:
                    print(f"  skip {item['image_path']}: {e}")

        ds = Dataset.from_generator(gen, features=features)
        tmp = shard_path + '.tmp'
        ds.to_parquet(tmp)
        os.replace(tmp, shard_path)

    _write_readme(hf_dir, done)
    print(f"Dataset ready in {hf_dir}/train/")


def _write_readme(hf_dir, done):
    names = sum(1 for p in done if p['kind'] == 'name')
    nids  = len(done) - names
    with open(os.path.join(hf_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(f"""\
---
task_categories:
- image-to-text
language:
- en
tags:
- ocr
- english
- low-ink
- faded
- synthetic
---

# English Low-Ink OCR Dataset

{len(done):,} synthetic English OCR images with near-faded (low-ink) augmentation.
Two categories: names in `FirstName (LastName)` format, and NID-style numeric strings.

| Split | Samples |
|:------|--------:|
| train | {len(done):,} |

## Breakdown
| Kind | Count |
|:-----|------:|
| name (FirstName (LastName)) | {names:,} |
| nid (numeric string) | {nids:,} |

## Columns
| Column | Type | Description |
|:---|:---|:---|
| `image` | Image | Cropped word/phrase image (RGB) |
| `text` | string | Ground-truth text label |
| `kind` | string | `name` or `nid` |
| `class_name` | string | `en_lowinktrain` |
| `source` | string | `synthetic_en_lowinktrain` |

## Augmentation
Every image is rendered with very light gray text (#888–#cc) then passed through:
- `LowInkRandomLines` (count 30–60, prob 0.7, applied 95 % of the time)
- `BadPhotoCopy` (75 %)
- `Faxify` (65 %)
- Gaussian noise σ=10–25 + blur 0.5–1.2 + JPEG quality 40–65

## Load
```python
from datasets import load_dataset
ds = load_dataset('parquet', data_files={{'train': 'hf/train/*.parquet'}})
```
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',  default='en_lowinktrain_output')
    parser.add_argument('--total',       type=int,   default=10000)
    parser.add_argument('--font_size',   type=int,   default=32)
    parser.add_argument('--blur',        type=float, default=0.3)
    parser.add_argument('--shard_size',  type=int,   default=500)
    parser.add_argument('--push_to_hub', default=None,
                        help='HF repo id to push to, e.g. kavinh07/en-lowinktrain-ocr')
    parser.add_argument('--hf_token',   default=None)
    parser.add_argument('--reset',       action='store_true')
    args = parser.parse_args()

    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trdg', 'images')
    en_fonts  = load_fonts('en')
    if not en_fonts:
        print("ERROR: no English fonts found in trdg/fonts/en/", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(en_fonts)} English font(s).")

    out_img  = os.path.join(args.output_dir, 'images')
    out_lbl  = os.path.join(args.output_dir, 'labels')
    hf_dir   = os.path.join(args.output_dir, 'hf')
    plan_path = os.path.join(args.output_dir, '.plan_en_lowinktrain.json')

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    os.makedirs(hf_dir,  exist_ok=True)

    if not args.reset and os.path.exists(plan_path):
        print(f"Resuming from {plan_path}")
        plan = _load_json(plan_path)
    else:
        print(f"Building plan: {args.total} samples (60% names, 40% NID)...")
        plan = _build_plan(args.total, out_img, out_lbl)
        _save_json(plan, plan_path)
        print(f"Plan saved: {len(plan)} items")

    _generate(plan, en_fonts, image_dir, args.font_size)

    done  = [p for p in plan if os.path.exists(p['image_path'])]
    names = sum(1 for p in done if p['kind'] == 'name')
    nids  = len(done) - names
    print(f"\nDone. {len(done):,}/{len(plan):,} images in {out_img}")
    print(f"  Names (FirstName (LastName)): {names:,}")
    print(f"  NID numbers:                  {nids:,}")

    build_hf_dataset(plan, hf_dir, shard_size=args.shard_size)

    token = args.hf_token or os.environ.get('HF_TOKEN')
    if args.push_to_hub and token:
        from datasets import load_dataset as ld
        print(f"\nPushing to {args.push_to_hub} ...")
        ds = ld('parquet', data_files={'train': os.path.join(hf_dir, 'train', '*.parquet')})
        ds.push_to_hub(args.push_to_hub, token=token)
        print(f"  https://huggingface.co/datasets/{args.push_to_hub}")
    elif args.push_to_hub:
        print("--push_to_hub set but no HF_TOKEN found — skipping push")


if __name__ == '__main__':
    main()
