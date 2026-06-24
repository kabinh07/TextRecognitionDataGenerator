"""
generate_hard_negatives.py

Targeted hard-negative OCR training images for 4 model failure modes:
  repeated_digits  — 3-5 consecutive identical digits in NID/phone contexts
  ga_pa            — গ/প glyph confusion (closed-top shapes)
  da_da            — ড/দ retroflex vs dental confusion
  low_ink          — faded/low-ink variants of all three above

Output: {output_dir}/images/ + labels/  (same flat dir as generate_combined output)
Picked up automatically by collect_synthetic() in prepare_hf_dataset.py.

Usage:
  python generate_hard_negatives.py --output_dir output --count 500
  python generate_hard_negatives.py --output_dir output --count 500 --reset
"""

import argparse
import io
import os
import random
import sys

import numpy as np
from PIL import Image as PILImage, ImageFilter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_combined import _apply_augraphy, _load_json, _maybe_downscale, _save_json
from trdg.data_generator import FakeTextDataGenerator
from trdg.utils import load_fonts


# ── Corpora ───────────────────────────────────────────────────────────────────

_REPEATED_DIGIT_TEXTS = [
    # 3-repeated in NID/phone contexts
    "19944824509000020", "01700000123", "199448245090000",
    "0190000456789", "19881234500003", "AC: 10000020",
    "01911000034", "1234000056", "19990000124567",
    "01800001234", "0000123456", "9900001234",
    "19551234000089", "0170000012345", "9999123456",
    "8888001234", "11110000234", "33330000567",
    "19440000123456", "01955000123",
    # 4-repeated runs
    "00001234", "11112345", "99990123",
    "12340000", "56780000", "00009999",
    "01700001111", "01900009999", "01600008888",
    "19990001111", "20001111234",
    # 5-repeated runs
    "000001234", "111110000", "999990001",
    "1234000005", "567890000",
    # mixed / realistic
    "011100001234", "1900000001122",
    "19440011100023", "01700044400123",
    "199000012345", "01811111234",
    "0000", "1111", "9999", "8888",
    "000", "111", "999", "888",
    "1234000000", "0000000001",
    "55550000", "66660123", "77770099",
    "0001234567890", "1111000056789",
    "01300000555", "01400011100",
    "19671111000023", "20001999111100",
]

_GA_PA_TEXTS = [
    "পলাশ গ্রাম", "গোপালগঞ্জ পল্লী", "পাগলা গেট",
    "গ্রাম: পাঁচগাছি", "প্রতাপগঞ্জ গ্রাম", "গোপাল পাড়া",
    "পারগাও গ্রাম", "গাজীপুর পৌরসভা", "পলাশবাড়ী গ্রাম",
    "গোপালপুর পাড়া", "পাথরঘাটা গ্রাম", "গাংনী পৌরসভা",
    "প্রতাপ গ্রাম", "পদ্মাগ্রাম", "গোপালনগর পল্লী",
    "পাইকগাছা গ্রাম", "গাজনার পাড়া", "পলাশপুর গ্রাম",
    "গৌরপুর পল্লী", "পাটগ্রাম গ্রাম", "গাইবান্ধা পৌরসভা",
    "পাঁচগাছি গ্রাম", "গাজীপুর পল্লী", "পদ্মা গ্রাম",
    "গোপালগঞ্জ পৌরসভা", "পাথরিয়া গ্রাম",
    "পলাশ: গ্রাম পাড়া", "গাজী পল্লী পাথর",
    "প্রাগপুর গ্রাম", "পাগলা গ্রাম", "গ্রামীণ পল্লী",
    "পল্লী গ্রাম: গাজনা", "প্রতাপপুর গ্রাম",
    "গোপালচন্দ্র পাড়া", "পাকিস্তান গ্রাম",
]

_DA_DA_TEXTS = [
    "গেন্ডারিয়া", "মান্ডা", "বান্ডেল রোড",
    "ইন্দিরা রোড ভান্ডার", "ডাকঘর: গেন্ডারিয়া",
    "রাস্তা: গেন্ডারিয়া মেইন রোড", "মান্ডা, খিলগাঁও",
    "গ্রাম: চান্দগাঁও, ডাকঘর: চান্দগাঁও",
    "বান্ডেল ঘাট রোড, কেরানীগঞ্জ", "মুন্ডা পাড়া",
    "গান্ডারিয়া ডাকঘর, ঢাকা", "মান্ডা থানা, ঢাকা",
    "ঠিকানা: গেন্ডারিয়া, ডাকঘর: গেন্ডারিয়া",
    "বাসা: ৩৪, গেন্ডারিয়া মেইন রোড",
    "গ্রাম: মান্ডা, ডাকঘর: খিলগাঁও",
    "চান্দপুর ডাকঘর", "বান্ডেল ঘাট",
    "মান্ডা বাজার, ঢাকা", "ডান্ডিয়া মোড়",
    "ইন্দিরা রোড ভান্ডার, ঢাকা",
    "গেন্ডারিয়া, ঢাকা-১২০৪",
    "মান্ডা ডাকঘর, খিলগাঁও, ঢাকা",
    "ন্ডা বাজার", "চান্দগাঁও থানা",
    "মান্ডা রোড, খিলগাঁও", "গেন্ডা পাড়া",
    "বান্ডেল মোড়, কেরানীগঞ্জ", "ডাকঘর: মান্ডা",
    "চান্দগাঁও পৌরসভা, চট্টগ্রাম",
    "গেন্ডারিয়া মেইন রোড, ঢাকা",
]

# ── Comprehensive Bengali conjunct (যুক্তবর্ণ) word list ─────────────────────
# Covers all common consonant clusters + all vowel matras + rare characters.
# Each line targets a distinct conjunct family so no cluster is left uncovered.

_CONJUNCT_TEXTS = [
    # ক্ক  ক্ট  ক্ত  ক্ন  ক্ব  ক্ম  ক্য  ক্র  ক্ল  ক্ষ  ক্স
    "মক্কা", "ডক্টর", "শক্তি", "বাক্নি", "বক্বক", "বাক্ময়",
    "বাক্য", "চক্র", "শুক্ল", "লক্ষ্য", "পক্স",
    # ক্ষ  ক্ষ্ণ  ক্ষ্ম  ক্ষ্য
    "ক্ষমা", "তীক্ষ্ণ", "লক্ষ্মী", "ক্ষ্যাপা",
    # গ্ধ  গ্ন  গ্ব  গ্ম  গ্য  গ্র  গ্ল
    "দুগ্ধ", "মুগ্ধ", "যুগ্ম", "ভাগ্য", "গ্রাম", "গ্লাস", "জগ্নাথ",
    # ঘ্ন  ঘ্র
    "ঘ্নী", "ঘ্রাণ",
    # ঙ্ক  ঙ্খ  ঙ্গ  ঙ্ঘ
    "শঙ্কা", "শঙ্খ", "বাংলা", "সঙ্ঘ",
    # চ্চ  চ্ছ  চ্ন  চ্য
    "বাচ্চা", "ইচ্ছা", "চক্চক্", "বাচ্য",
    # ছ্র
    "ছ্রিলঙ্কা",
    # জ্জ  জ্ঞ  জ্ব  জ্য  জ্র
    "উজ্জ্বল", "জ্ঞান", "জ্বর", "রাজ্য", "ব্রজ",
    # জ্ঞ
    "অজ্ঞান", "বিজ্ঞান", "জ্ঞাপন",
    # ঞ্চ  ঞ্জ
    "পঞ্চম", "রঞ্জন", "গঞ্জ",
    # ট্ট  ট্ব  ট্য
    "ঘণ্টাটি", "ভট্টাচার্য", "নাট্য",
    # ড্ড  ড্ব
    "আড্ডা", "ড্বিপ",
    # ণ্ট  ণ্ড  ণ্ণ  ণ্য
    "ঘণ্টা", "মণ্ডল", "পণ্য",
    # ত্ত  ত্থ  ত্ন  ত্ব  ত্ম  ত্য  ত্র  ত্ল
    "উত্তর", "তত্ত্ব", "যত্ন", "স্বত্ব", "আত্মা", "সত্য", "মাত্র", "তাত্ল",
    # থ্য  থ্র
    "স্বাস্থ্য", "থ্রি",
    # দ্দ  দ্ধ  দ্ব  দ্ভ  দ্ম  দ্য  দ্র
    "উদ্দেশ্য", "বুদ্ধ", "বিদ্বান", "অদ্ভুত", "পদ্ম", "বিদ্যা", "আদ্র",
    # ধ্ব  ধ্য  ধ্র
    "ধ্বনি", "সাধ্য", "ধ্রুব",
    # ন্ট  ন্ড  ন্ত  ন্থ  ন্দ  ন্ধ  ন্ন  ন্ব  ন্ম  ন্য  ন্র  ন্স
    "প্রিন্ট", "মান্ডা", "শান্ত", "গ্রন্থ", "আনন্দ", "বন্ধু",
    "নিন্দা", "জন্ম", "নন্দন", "মন্ত্র", "নন্স",
    # প্ট  প্ত  প্ন  প্প  প্য  প্র  প্ল  প্স
    "অ্যাপ্ট", "তিপ্তা", "স্বপ্ন", "প্রাপ্ত", "দাপ্য", "প্রেম", "প্লেট", "গিপ্স",
    # ফ্র  ফ্ল
    "ফ্রান্স", "ফ্লোর",
    # ব্জ  ব্দ  ব্ধ  ব্ব  ব্য  ব্র  ব্ল
    "ব্যবসা", "শব্দ", "গব্ধ", "ডাব্বা", "ব্যক্তি", "ব্রিজ", "ব্লক",
    # ভ্র  ভ্য  ভ্ব
    "ভ্রাতা", "সভ্য", "ভ্রমণ",
    # ম্ন  ম্প  ম্ফ  ম্ব  ম্ভ  ম্ম  ম্য  ম্র  ম্ল
    "নিম্ন", "কম্পন", "ম্ফল", "লম্বা", "সম্ভব", "সম্মান",
    "ম্যাচ", "ম্রিয়মাণ", "অম্ল",
    # য্য
    "মহাশয্য",
    # র্ক  র্গ  র্ট  র্ড  র্ত  র্থ  র্দ  র্ধ  র্ন  র্প  র্ব  র্ম  র্য  র্ল  র্শ  র্ষ  র্স  র্হ
    "কর্ক", "মর্গ", "কার্ট", "কার্ড", "কর্তা", "অর্থ", "গর্দান",
    "বর্ধন", "পর্ন", "কর্প", "গর্ব", "ধর্ম", "কর্ম", "কার্য",
    "বর্ল", "বর্শা", "বর্ষ", "বর্ষণ", "বর্হি",
    # ল্ক  ল্গ  ল্ট  ল্ড  ল্প  ল্ফ  ল্ব  ল্ম  ল্য  ল্ল
    "শল্ক", "ল্গু", "বল্টু", "ল্ড্", "বিল্প", "গল্ফ", "বল্ব",
    "চলচ্চিত্র", "ল্য", "উল্লাস",
    # শ্চ  শ্ন  শ্ব  শ্ম  শ্য  শ্র  শ্ল
    "নিশ্চিত", "শ্মশান", "বিশ্ব", "শ্মশান", "শ্যামল", "শ্রম", "শ্লোক",
    # ষ্ক  ষ্ট  ষ্ণ  ষ্ত  ষ্ন  ষ্প  ষ্ব  ষ্ম  ষ্য
    "ভিষ্ক", "কষ্ট", "কৃষ্ণ", "নিষ্ঠা", "বৃষ্টি", "পুষ্প",
    "বৈষ্ণব", "উষ্মা", "ষষ্ঠী",
    # স্ক  স্ট  স্ত  স্থ  স্ন  স্প  স্ফ  স্ব  স্ম  স্য  স্র  স্ল
    "স্কুল", "স্টেশন", "স্তর", "স্থান", "স্নান", "স্পষ্ট",
    "স্ফীত", "স্বাধীন", "স্মরণ", "সাস্য", "ইসরাইল", "স্লোগান",
    # হ্ণ  হ্ন  হ্ব  হ্ম  হ্য  হ্র  হ্ল
    "ব্রহ্মণ", "চিহ্ন", "বিহ্বল", "ব্রহ্ম", "সহ্য", "হ্রাস", "হ্লাদ",
    # ড়  ঢ়  য়  (special conjuncts)
    "গড়", "বাড়ি", "পড়া", "ঢাকঢোল", "যায়", "হয়",
    # রেফ (র + ্ + consonant)
    "র্কট", "র্গল", "র্চনা", "র্ছেদ", "র্জন", "র্ণ", "র্তন",
    "র্দন", "র্ধন", "র্পণ", "র্বণ", "র্মণ", "র্যন্ত", "র্শন",
    # ◌্র (consonant + ্ + র)
    "ক্রম", "গ্রন্থ", "ত্রাণ", "দ্রুত", "প্রাণ", "ব্রাহ্মণ",
    "ম্রিয়মাণ", "ল্রস", "শ্রেণী", "স্রোত",
    # ◌্য (consonant + ্ + য)
    "কার্য", "জাতীয়", "ভাষ্য", "রাজ্য", "সত্য", "স্বাস্থ্য",
    # vowel matras on conjuncts — ি  ী  ু  ূ  ৃ  ে  ৈ  ো  ৌ  ং  ঃ  ঁ
    "বিশ্বাস", "প্রীতি", "পুষ্টি", "ভূমিকা", "কৃষ্টি",
    "শ্রেষ্ঠ", "সৈন্য", "স্বাধীনতা", "স্বাস্থ্যকর",
    "সংস্কৃতি", "দুঃখ", "চাঁদ",
    # চন্দ্রবিন্দু
    "চাঁদ", "গাঁয়", "মাঁ", "কাঁচ", "হাঁটু", "শাঁখ",
    # anusvara ং
    "বাংলাদেশ", "সংসদ", "সংখ্যা", "রংপুর", "ঢাকাং",
    # visarga ঃ
    "দুঃখ", "প্রঃ", "সঃ", "অতঃপর",
    # Bengali numerals + conjuncts in context
    "১ম শ্রেণী", "২য় বর্ষ", "৩য় শ্রেণী",
    "বিদ্যালয়: ৫ম শ্রেণী",
    # compound words mixing multiple conjuncts
    "স্বাস্থ্যসম্মত", "সাংবিধানিক", "প্রতিষ্ঠাতা",
    "বিশ্ববিদ্যালয়", "পরিবহন", "সংস্কৃতি",
    "রাজনৈতিক", "প্রশাসনিক", "বিজ্ঞানসম্মত",
    "গণতান্ত্রিক", "ভৌগোলিক", "স্থাপত্যকলা",
    "সাহিত্যিক", "দার্শনিক", "অর্থনৈতিক",
    "ভাষাতাত্ত্বিক", "মনস্তাত্ত্বিক", "ঐতিহাসিক",
    "সামাজিক", "পারিবারিক", "প্রযুক্তিগত",
    # NID field realistic compound words
    "ইব্রাহিম", "আব্দুল্লাহ", "মোস্তফা", "মোস্তাফিজুর",
    "রহমতুল্লাহ", "নাজমুস্সাকিব", "আব্দুর রহমান",
    "ফেরদৌস", "মাহবুবুল আলম", "নূরুল ইসলাম",
    "শহীদুল্লাহ", "গোলাম মোস্তফা",
    "হাবিবুল্লাহ", "আশরাফুল ইসলাম",
    # address specific
    "উত্তরখান", "শ্যামলী", "মোহাম্মদপুর",
    "আদাবর", "কল্যাণপুর", "মিরপুর",
    "শ্রীনগর", "মুন্সিগঞ্জ", "নারায়ণগঞ্জ",
    "চট্টগ্রাম", "ময়মনসিংহ", "রাজশাহী",
    "খুলনা", "সিলেট", "বরিশাল",
    "সাভার", "টঙ্গী", "গাজীপুর",
    "ফরিদপুর", "যশোর", "কুষ্টিয়া",
    "ব্রাহ্মণবাড়িয়া", "কিশোরগঞ্জ", "নেত্রকোণা",
    "সুনামগঞ্জ", "হবিগঞ্জ", "মৌলভীবাজার",
    "শেরপুর", "জামালপুর", "টাঙ্গাইল",
    "মানিকগঞ্জ", "নরসিংদী", "মাদারীপুর",
]

# ── Low-ink colour helpers ────────────────────────────────────────────────────

def _faded_black():
    v = random.randint(0x1a, 0x55)
    return f'#{v:02x}{v:02x}{v:02x}'


def _faded_red():
    r = random.randint(180, 225)
    gb = random.randint(80, 160)
    return f'#{r:02x}{gb:02x}{gb:02x}'


# ── Augmentation ──────────────────────────────────────────────────────────────

def _apply_low_ink(img):
    """Noise + slight blur + JPEG compression to simulate faded prints."""
    arr = np.array(img.convert('RGB')).astype(np.float32)
    sigma = random.uniform(5, 15)
    arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    img = PILImage.fromarray(arr)
    if random.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=random.randint(60, 80))
    buf.seek(0)
    return PILImage.open(buf).convert('RGB')


def _augraphy_result(result, fallback):
    """Extract ndarray from augraphy result (handles ndarray, tuple, or dict)."""
    if isinstance(result, np.ndarray):
        return result
    if isinstance(result, (tuple, list)) and len(result) > 0 and isinstance(result[0], np.ndarray):
        return result[0]
    if isinstance(result, dict):
        return result.get('output', fallback)
    return fallback


def _apply_augraphy_low_ink(img):
    """Augraphy pipeline biased toward ink-starvation effects."""
    try:
        from augraphy import BadPhotoCopy, Faxify, LowInkRandomLines
    except ImportError:
        return img
    arr = np.array(img.convert('RGB'))
    augmentors = [
        (BadPhotoCopy, 0.6),
        (Faxify,       0.5),
    ]
    # LowInkRandomLines: try custom params, fall back to defaults
    try:
        low_ink_inst = LowInkRandomLines(count_range=(20, 45), noise_probability=0.6)
    except Exception as e:
        print(f"  LowInkRandomLines custom params failed ({e}), using defaults")
        try:
            low_ink_inst = LowInkRandomLines()
        except Exception:
            low_ink_inst = None
    if low_ink_inst is not None:
        augmentors.insert(0, (lambda inst=low_ink_inst: inst, 0.9))

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

def _build_plan(texts, lang, count, out_img_dir, out_lbl_dir, prefix, low_ink):
    items = []
    for i in range(count):
        text = random.choice(texts)
        fname = f'{prefix}_{i:06d}'
        items.append({
            'text':       text,
            'lang':       lang,
            'low_ink':    low_ink,
            'image_path': os.path.join(out_img_dir, f'{fname}.png'),
            'label_path': os.path.join(out_lbl_dir, f'{fname}.txt'),
        })
    return items


def _build_plan_exhaustive(texts, lang, count, out_img_dir, out_lbl_dir, prefix, low_ink):
    """Every text appears at least once; random fills up to max(count, len(texts))."""
    pool = list(texts)
    random.shuffle(pool)
    target = max(count, len(pool))
    seq = pool + [random.choice(pool) for _ in range(target - len(pool))]
    items = []
    for i, text in enumerate(seq):
        fname = f'{prefix}_{i:06d}'
        items.append({
            'text':       text,
            'lang':       lang,
            'low_ink':    low_ink,
            'image_path': os.path.join(out_img_dir, f'{fname}.png'),
            'label_path': os.path.join(out_lbl_dir, f'{fname}.txt'),
        })
    return items


# ── Generation loop ───────────────────────────────────────────────────────────

def _generate_plan(plan, bn_fonts, en_fonts, image_dir, font_size, blur):
    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    print(f"  {len(plan) - len(todo)} done, {len(todo)} remaining")

    for idx, item in enumerate(tqdm(todo, desc='  generating')):
        lang     = item['lang']
        low_ink  = item['low_ink']
        fonts    = bn_fonts if lang == 'bn' else en_fonts
        font     = random.choice(fonts)

        text_color = _faded_black() if low_ink else '#000000'

        img = None
        for _ in range(5):
            try:
                img = FakeTextDataGenerator.generate(
                    index=idx,
                    text=item['text'],
                    font=font,
                    out_dir=None,
                    size=font_size,
                    extension=None,
                    skewing_angle=0,
                    random_skew=False,
                    blur=blur if not low_ink else random.uniform(0.3, 0.8),
                    random_blur=not low_ink,
                    background_type=1,
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
                    word_split=(lang == 'bn'),
                    image_dir=image_dir,
                    stroke_width=0,
                    stroke_fill=text_color,
                    image_mode='RGB',
                )
                if img is not None:
                    break
            except Exception:
                img = None

        if img is None:
            continue

        if low_ink:
            img = _apply_augraphy_low_ink(img)
            img = _apply_low_ink(img)
        else:
            img = _apply_augraphy(img)

        img = _maybe_downscale(img)

        os.makedirs(os.path.dirname(item['image_path']), exist_ok=True)
        os.makedirs(os.path.dirname(item['label_path']), exist_ok=True)
        img.save(item['image_path'])
        with open(item['label_path'], 'w', encoding='utf-8') as f:
            f.write(item['text'])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output',
                        help='Root output dir (same as generate_combined --output_dir)')
    parser.add_argument('--font_size',  type=int,   default=32)
    parser.add_argument('--blur',       type=float, default=0.3)
    parser.add_argument('--count',      type=int,   default=500,
                        help='Images per category; low-ink doubles total')
    parser.add_argument('--reset',      action='store_true')
    args = parser.parse_args()

    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trdg', 'images')
    bn_fonts  = load_fonts('bn')
    en_fonts  = load_fonts('en')

    out_img = os.path.join(args.output_dir, 'images')
    out_lbl = os.path.join(args.output_dir, 'labels')
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    plan_path = os.path.join(args.output_dir, '.plan_hardneg.json')

    if not args.reset and os.path.exists(plan_path):
        print(f"Resuming from {plan_path}")
        plan = _load_json(plan_path)
    else:
        print("Building plan...")
        plan = []
        plan += _build_plan(_REPEATED_DIGIT_TEXTS, 'en', args.count,
                            out_img, out_lbl, 'repdig',      low_ink=False)
        plan += _build_plan(_REPEATED_DIGIT_TEXTS, 'en', args.count,
                            out_img, out_lbl, 'repdig_fade', low_ink=True)
        plan += _build_plan(_GA_PA_TEXTS, 'bn', args.count,
                            out_img, out_lbl, 'gapa',        low_ink=False)
        plan += _build_plan(_GA_PA_TEXTS, 'bn', args.count,
                            out_img, out_lbl, 'gapa_fade',   low_ink=True)
        plan += _build_plan(_DA_DA_TEXTS, 'bn', args.count,
                            out_img, out_lbl, 'dada',        low_ink=False)
        plan += _build_plan(_DA_DA_TEXTS, 'bn', args.count,
                            out_img, out_lbl, 'dada_fade',   low_ink=True)
        # Conjuncts: exhaustive pass ensures every cluster appears, then random fills
        plan += _build_plan_exhaustive(_CONJUNCT_TEXTS, 'bn', args.count,
                            out_img, out_lbl, 'conjunct',      low_ink=False)
        plan += _build_plan_exhaustive(_CONJUNCT_TEXTS, 'bn', args.count,
                            out_img, out_lbl, 'conjunct_fade', low_ink=True)
        _save_json(plan, plan_path)
        print(f"Plan: {len(plan)} items ({args.count}×2 per 4 categories)")

    _generate_plan(plan, bn_fonts, en_fonts, image_dir, args.font_size, args.blur)

    done = sum(1 for p in plan if os.path.exists(p['image_path']))
    print(f"\nDone. {done:,}/{len(plan):,} images in {out_img}")


if __name__ == '__main__':
    main()
