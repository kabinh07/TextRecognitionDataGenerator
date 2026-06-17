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
import numpy as np
from PIL import Image as PILImage
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
# (name, weight, font_size_min, font_size_max, blur_max)
# Text always black; background_type=3 (trdg/images/).
# Rotation is applied post-generation via _rotate_image(), not by trdg.

_ADDR_PROFILES = [
    ('normal',  25, 32, 40, 0),
    ('medium',  40, 22, 31, 1),
    ('low_res', 35, 14, 21, 1),
]
_ADDR_WEIGHTS = [p[1] for p in _ADDR_PROFILES]


def _pick_addr_profile():
    p = random.choices(_ADDR_PROFILES, weights=_ADDR_WEIGHTS, k=1)[0]
    _, _, fs_min, fs_max, blur_max = p
    return random.randint(fs_min, fs_max), blur_max


# ── Synthetic text augmentation ───────────────────────────────────────────────

_BN_MONTHS = [
    'জানুয়ারি', 'ফেব্রুয়ারি', 'মার্চ', 'এপ্রিল', 'মে', 'জুন',
    'জুলাই', 'আগস্ট', 'সেপ্টেম্বর', 'অক্টোবর', 'নভেম্বর', 'ডিসেম্বর',
]
_EN_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

_RE_EN_DIGIT  = re.compile(r'^[0-9]+$')
_RE_EN_SPACED = re.compile(r'^[0-9]+( [0-9]+)+$')
_RE_BN_DIGIT  = re.compile(r'^[০-৯]+$')
_RE_BN_SPACED = re.compile(r'^[০-৯]+( [০-৯]+)+$')
_RE_EN_DOB    = re.compile(
    r'^(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})$',
    re.I,
)
_RE_SLASH_DOB = re.compile(r'^\d{1,2}/\d{1,2}/\d{4}$')
_RE_DASH_DOB  = re.compile(r'^\d{1,2}-\d{1,2}-\d{4}$')
_RE_BN_DOB    = re.compile(
    r'^[০-৯]{1,2}\s+(' + '|'.join(_BN_MONTHS) + r')\s+[০-৯]{4}$'
)

_NAME_CLASSES = {'bn_name', 'en_name', 'father_name', 'mother_name'}


def _to_bn_digit(s):
    return s.translate(str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯'))


def _detect_nid_formats(texts):
    fmts = []
    for t in texts:
        t = t.strip()
        if _RE_EN_SPACED.match(t):
            fmts.append(('en_spaced', tuple(len(g) for g in t.split())))
        elif _RE_EN_DIGIT.match(t):
            fmts.append(('en_num', len(t)))
        elif _RE_BN_SPACED.match(t):
            fmts.append(('bn_spaced', tuple(len(g) for g in t.split())))
        elif _RE_BN_DIGIT.match(t):
            fmts.append(('bn_num', len(t)))
    return fmts or [('en_num', 10), ('en_num', 13)]


def _gen_nid(fmt):
    kind, spec = fmt
    n = sum(spec) if isinstance(spec, tuple) else spec
    digits = [str(random.randint(0, 9)) for _ in range(n)]
    if kind in ('en_num', 'bn_num'):
        s = ''.join(digits)
        return _to_bn_digit(s) if kind == 'bn_num' else s
    out, i = [], 0
    for size in spec:
        chunk = ''.join(digits[i:i + size])
        out.append(_to_bn_digit(chunk) if kind == 'bn_spaced' else chunk)
        i += size
    return ' '.join(out)


def _detect_dob_formats(texts):
    fmts = set()
    for t in texts:
        t = t.strip()
        m = _RE_EN_DOB.match(t)
        if m:
            fmts.add('en_text_zero' if len(m.group(1)) == 2 else 'en_text')
        elif _RE_SLASH_DOB.match(t):
            fmts.add('slash')
        elif _RE_DASH_DOB.match(t):
            fmts.add('dash')
        elif _RE_BN_DOB.match(t):
            fmts.add('bn_text')
    return list(fmts) or ['en_text', 'slash']


def _gen_dob(fmt):
    d = random.randint(1, 28)
    m = random.randint(1, 12)
    y = random.randint(1950, 2005)
    if fmt == 'en_text':
        return f"{d} {_EN_MONTHS[m - 1]} {y}"
    if fmt == 'en_text_zero':
        return f"{d:02d} {_EN_MONTHS[m - 1]} {y}"
    if fmt == 'slash':
        return f"{d:02d}/{m:02d}/{y}"
    if fmt == 'dash':
        return f"{d:02d}-{m:02d}-{y}"
    if fmt == 'bn_text':
        return f"{_to_bn_digit(str(d))} {_BN_MONTHS[m - 1]} {_to_bn_digit(str(y))}"
    return f"{d:02d}/{m:02d}/{y}"


def _augment_class_dict(class_dict, target_per_class, is_english=False):
    """Expand each class to target_per_class using synthetic generation."""
    result = {k: list(v) for k, v in class_dict.items()}

    # Names — split real names into tokens, recombine randomly
    for cls in _NAME_CLASSES:
        if cls not in result or len(result[cls]) >= target_per_class:
            continue
        tokens = list(set(
            part for name in result[cls]
            for part in name.strip().split() if len(part) > 1
        ))
        if not tokens:
            continue
        needed = target_per_class - len(result[cls])
        for _ in range(needed):
            n = random.randint(1, min(3, len(tokens)))
            name = ' '.join(random.sample(tokens, n))
            result[cls].append(name.upper() if is_english else name)

    # NID numbers — detect length/grouping formats, generate synthetics
    if 'nid_number' in result and len(result['nid_number']) < target_per_class:
        fmts = _detect_nid_formats(result['nid_number'])
        needed = target_per_class - len(result['nid_number'])
        for _ in range(needed):
            result['nid_number'].append(_gen_nid(random.choice(fmts)))

    # Dates of birth — detect date formats, generate synthetics
    if 'date_of_birth' in result and len(result['date_of_birth']) < target_per_class:
        fmts = _detect_dob_formats(result['date_of_birth'])
        needed = target_per_class - len(result['date_of_birth'])
        for _ in range(needed):
            result['date_of_birth'].append(_gen_dob(random.choice(fmts)))

    return result


# ── Bangla character coverage ─────────────────────────────────────────────────
# Target: every commonly-used char in U+0980–U+09FF should appear in training.
# Chars covered: all vowels, all consonants (incl. ৎ ড় ঢ় য়), all vowel
# diacritics, ঁ ং ঃ ্, and a broad set of conjuncts (যুক্তাক্ষর).

_BN_TARGET_CHARS = set(
    'ঁংঃ'                                      # ঁ ং ঃ
    'অআইঈউঊঋ'             # অ আ ই ঈ উ ঊ ঋ
    'এঐওঔ'                                # এ ঐ ও ঔ
    'কখগঘঙ'                         # ক খ গ ঘ ঙ
    'চছজঝঞ'                         # চ ছ জ ঝ ঞ
    'টঠডঢণ'                         # ট ঠ ড ঢ ণ
    'তথদধন'                         # ত থ দ ধ ন
    'পফবভম'                         # প ফ ব ভ ম
    'যরল'                                     # য র ল
    'শষসহ'                               # শ ষ স হ
    '়'                                                  # ় (nukta)
    'ািীুূৃ'                  # া ি ী ু ূ ৃ
    'েৈোৌ্'                         # ে ৈ ো ৌ ্
    'ৎ'                                                  # ৎ
    '০১২৩৪'                         # ০ ১ ২ ৩ ৪
    '৫৬৭৮৯'                         # ৫ ৬ ৭ ৮ ৯
)

_BN_COVERAGE_WORDS = [
    # ৎ (khanda ta) — key missing char
    "হঠাৎ", "অকস্মাৎ", "বাৎসরিক", "তৎক্ষণাৎ", "তৎপর",
    "তৎকাল", "তৎসম", "মাৎস্য", "সাৎ", "কাৎলা",
    "নিমেষাৎ", "সহসাৎ", "তৎপরতা",
    # ঁ (chandrabindu)
    "চাঁদ", "বাঁশ", "গাঁও", "হাঁটা", "পাঁচ",
    "তাঁত", "মাঁ", "চাঁপা", "কাঁচ", "বাঁকা",
    "ধাঁধা", "সাঁতার", "পাঁজর", "হাঁস", "ঘাঁটি",
    "কাঁদা", "বাঁধ", "সাঁঝ",
    # ঃ (visarga)
    "দুঃখ", "নিঃশ্বাস", "প্রাতঃ", "অতঃপর",
    "পুনঃ", "বস্তুতঃ", "মূলতঃ", "সংক্ষেপতঃ",
    # ড় ঢ় য়
    "বড়", "ঘোড়া", "পড়া", "বাড়ি", "ভাড়া",
    "গড়", "পাড়া", "জুড়ি", "চড়া", "ছাড়া",
    "ওড়া", "নড়া", "তাড়া", "আড়াল",
    "ময়না", "রায়", "ছায়া", "নয়ন", "দয়া",
    "ভয়", "জয়", "সয়",
    # ৃ (ri-kar) — often missing
    "ঋতু", "ঋণ", "কৃষক", "কৃষি", "তৃণ",
    "গৃহ", "মৃত্যু", "বৃষ্টি", "হৃদয়", "নৃত্য",
    "পৃথিবী", "তৃতীয়", "ঘৃণা", "কৃত্রিম",
    "শৃঙ্খল", "বৃত্তান্ত", "কৃতজ্ঞ",
    # Vowels (standalone)
    "অথবা", "আকাশ", "ইচ্ছা", "ঈদ", "উৎসব",
    "ঊষা", "ঋষি", "এখন", "ঐতিহ্য", "ওষুধ", "ঔষধ",
    # All vowel diacritics in context
    "কাজ", "কিছু", "নীল", "কুল", "ভূমি",
    "কৃষি", "দেশ", "বৈদ্য", "বোন", "মৌসুম",
    # ক্ষ conjunct
    "ক্ষমা", "ক্ষতি", "ক্ষমতা", "শিক্ষা", "লক্ষ্য",
    "দক্ষতা", "রক্ষণ", "বিক্ষিপ্ত", "প্রতিরক্ষা",
    # জ্ঞ conjunct
    "জ্ঞান", "বিজ্ঞান", "প্রজ্ঞা", "কৃতজ্ঞ",
    "সংজ্ঞা", "বিজ্ঞপ্তি", "অজ্ঞতা",
    # ষ-conjuncts
    "কষ্ট", "নষ্ট", "শ্রেষ্ঠ", "পরিষ্কার",
    "ষষ্ঠ", "ষড়যন্ত্র", "ষোল", "বিষয়",
    # ত্ত ন্ন
    "উত্তর", "সত্ত্ব", "অন্ন", "পান্না",
    # ল্ল
    "উল্লাস", "আল্লাহ", "কল্লোল",
    # ম্ম
    "সম্মান", "সম্মতি",
    # হ্ন হ্ম
    "চিহ্ন", "ব্রহ্ম", "ব্রহ্মপুত্র",
    # র-ফলা (্র) conjuncts
    "প্রথম", "প্রতিষ্ঠান", "পত্র", "ত্রুটি",
    "দ্রুত", "শ্রম", "শ্রেণি", "ক্রয়",
    "গ্রাম", "ভ্রমণ", "মন্ত্র", "আক্রমণ",
    # য-ফলা (্য) conjuncts
    "ব্যবহার", "ব্যক্তি", "বিদ্যা", "বিদ্যালয়",
    "সত্য", "নিত্য", "ভাগ্য", "স্বাস্থ্য",
    # স্ত স্থ
    "স্তর", "স্থান", "স্থাপন", "স্পষ্ট",
    # ব্ল ফ্ল স্ক
    "ব্লক", "ফ্ল্যাট", "স্কুল",
    # ন্ত ন্দ ন্ধ
    "শান্ত", "অন্ত", "বন্দর", "সন্দেহ",
    "গন্ধ", "সন্ধ্যা", "বন্ধু",
    # ঞ্চ ঞ্জ
    "পঞ্চম", "বঞ্চিত", "গঞ্জ", "কুঞ্জ", "অঞ্জলি",
    # ণ্ট ণ্ড
    "ঘণ্টা", "কণ্ঠ", "মণ্ডল",
    # ব্দ ব্ধ
    "শব্দ", "লব্ধ", "সিদ্ধ",
    # শ্ব শ্ন
    "বিশ্ব", "অশ্ব", "প্রশ্ন", "কৃষ্ণ",
    # ক্ত
    "মুক্ত", "রক্ত", "ভক্তি",
    # ত্ব
    "সত্ব", "কর্তৃত্ব", "স্বাধীনতা",
    # ঙ ঞ standalone coverage
    "রং", "ঢং", "আঙ্গুর", "ভাঙ্গা",
    "মাঞ্জা", "পঞ্চ",
    # ষ ণ standalone
    "ষোল", "ষাট", "প্রাণ", "বর্ণ", "রণ",
    # Bangla digits (all 10)
    "০১২৩৪৫৬৭৮৯",
    # Common NID-context words with diverse chars
    "জাতীয়তা", "নাগরিকত্ব", "জন্মনিবন্ধন",
    "স্বাক্ষর", "পরিচয়পত্র",
    "বাংলাদেশ", "ঢাকা", "চট্টগ্রাম",
    "ময়মনসিংহ", "টাঙ্গাইল", "ফরিদপুর",
    "সংস্কৃতি", "বিশ্ববিদ্যালয়",
    # Names common in NID with rare chars
    "আব্দুল", "আক্তার", "মোস্তফা",
    "আফতাব", "মুহম্মদ", "রহমান",
    # চ্ছ
    "বিচ্ছেদ", "উচ্ছেদ", "সচ্ছল",
]


def _report_bn_coverage(texts_by_class):
    import unicodedata
    covered = set()
    for texts in texts_by_class.values():
        for t in texts:
            covered.update(unicodedata.normalize('NFC', t))
    missing = _BN_TARGET_CHARS - covered
    if missing:
        chars_str = ' '.join(
            f'{c}(U+{ord(c):04X})' for c in sorted(missing)
        )
        print(f"  Missing {len(missing)} Bangla chars: {chars_str}")
    else:
        print("  All target Bangla chars covered.")
    return missing


def _inject_char_coverage(bn_class_dict, min_reps=30):
    """Add coverage words to 'char_coverage' class so rare chars are rendered."""
    pool = _BN_COVERAGE_WORDS * min_reps
    random.shuffle(pool)
    bn_class_dict['char_coverage'] = pool
    print(f"  Injected char_coverage class: {len(pool):,} texts "
          f"({len(_BN_COVERAGE_WORDS)} unique × {min_reps} reps)")


# ── Confusion-pair training data ─────────────────────────────────────────────
# Short Bangla words that isolate visually similar characters so the model
# learns to distinguish them. Each key is the target character; value is a
# list of short words (≤3 syllables) where that character is prominent.

_BN_CONFUSION_CHARS = {
    # ── ঝ vs খ ────────────────────────────────────────────────────────────────
    'ঝ': ['ঝড়', 'ঝুড়ি', 'ঝামেলা', 'ঝরনা', 'ঝাঁপ', 'মাঝি', 'ঝলক', 'ঝুলন',
           'ঝিল', 'ঝাড়', 'ঝোপ', 'ঝাঁজ'],
    'খ': ['খবর', 'খাবার', 'খেলা', 'খুশি', 'খরচ', 'দেখা', 'রাখা', 'খামার',
           'খালি', 'খোলা', 'খাতা', 'খানা'],
    # ── ণ vs ন ────────────────────────────────────────────────────────────────
    'ণ': ['গণনা', 'বর্ণ', 'প্রাণ', 'ঘণ্টা', 'পণ', 'রণ', 'মণ', 'গণ',
           'বাণ', 'তাণ', 'কণা', 'ণত'],
    'ন': ['নাম', 'নদী', 'নিজ', 'বন', 'মন', 'দিন', 'খান', 'মানুষ',
           'নীল', 'নেই', 'তিন', 'নতুন'],
    # ── শ vs ষ ────────────────────────────────────────────────────────────────
    'শ': ['শব্দ', 'শহর', 'শেষ', 'শান্ত', 'শুরু', 'শাখা', 'শ্রম', 'শিক্ষা',
           'শীত', 'শোনা', 'শক্তি', 'শাড়ি'],
    'ষ': ['ষোল', 'বিষয়', 'কষ্ট', 'নষ্ট', 'ষষ্ঠ', 'বর্ষ', 'ষড়', 'বিষ',
           'আষাঢ়', 'দোষ', 'রোষ', 'ষাট'],
    # ── ড vs ঢ ────────────────────────────────────────────────────────────────
    'ড': ['ডাক', 'ডান', 'ডাকা', 'ডুব', 'ডিম', 'ডাল', 'ডোল', 'ডেকে',
           'ডাবা', 'ডুমুর', 'ডেরা', 'ডানা'],
    'ঢ': ['ঢাকা', 'ঢেউ', 'ঢোল', 'ঢাকনা', 'ঢং', 'ঢালু', 'ঢেলে', 'ঢুকে',
           'ঢিল', 'ঢোকা', 'ঢাল', 'ঢাকি'],
    # ── ব vs ভ ────────────────────────────────────────────────────────────────
    'ব': ['বাড়ি', 'বন', 'বয়স', 'বলা', 'বড়', 'বাজার', 'বারো', 'বেলা',
           'বাঘ', 'বাঁশ', 'বিষয়', 'বোন'],
    'ভ': ['ভাই', 'ভালো', 'ভূমি', 'ভয়', 'ভেতর', 'ভাষা', 'ভিড়', 'ভাগ',
           'ভোর', 'ভাড়া', 'ভালুক', 'ভিটা'],
    # ── ধ vs ঘ ────────────────────────────────────────────────────────────────
    'ধ': ['ধান', 'ধরন', 'ধীর', 'বিধি', 'ধাক্কা', 'ধোঁয়া', 'ধুলো', 'ধারা',
           'ধনী', 'ধারণ', 'ধরা', 'ধোয়া'],
    'ঘ': ['ঘর', 'ঘাস', 'ঘুম', 'ঘোড়া', 'মেঘ', 'ঘণ্টা', 'ঘড়ি', 'ঘেরা',
           'ঘাড়', 'ঘিরে', 'ঘটনা', 'ঘেঁটু'],
    # ── ত vs থ ────────────────────────────────────────────────────────────────
    'ত': ['তাই', 'তখন', 'তুমি', 'তালা', 'তিন', 'তারা', 'তলা', 'তবে',
           'তেল', 'তামা', 'তোলা', 'তাকা'],
    'থ': ['থাকা', 'থামা', 'থালা', 'পথ', 'থেকে', 'থোকা', 'থাবা', 'থোড়',
           'থানা', 'থামো', 'থুতু', 'থলে'],
    # ── ক vs ট ────────────────────────────────────────────────────────────────
    'ক': ['কাজ', 'কথা', 'কিছু', 'কেন', 'কাল', 'কানে', 'কেউ', 'কোথা',
           'কলম', 'কপাল', 'কাছে', 'কিনা'],
    'ট': ['টাকা', 'টেবিল', 'টান', 'টুকরো', 'পাট', 'টাটকা', 'টেনে', 'টিন',
           'টোপ', 'টানা', 'টাইম', 'টিকিট'],
    # ── দ vs ব ────────────────────────────────────────────────────────────────
    'দ': ['দিন', 'দেশ', 'দেখা', 'দুই', 'দাম', 'দল', 'দরজা', 'দূরে',
           'দাড়ি', 'দেরি', 'দানা', 'দোকান'],
    # ── র vs ব (low-res confusion) ────────────────────────────────────────────
    'র': ['রক্ত', 'রান্না', 'রাত', 'রঙ', 'রাস্তা', 'রোদ', 'রাখা', 'রোগ',
           'রাজা', 'রূপ', 'রিকশা', 'রসুন'],
    # ── গ vs ও ────────────────────────────────────────────────────────────────
    'গ': ['গাছ', 'গাড়ি', 'গল্প', 'গান', 'গরম', 'গভীর', 'গেল', 'গোলাপ',
           'গোড়া', 'গাল', 'গ্রাম', 'গণনা'],
    'ও': ['ওষুধ', 'ওজন', 'ওপর', 'ওঠা', 'ওই', 'ওরা', 'ওখানে', 'ওদের'],
    # ── Diacritics: ি vs ী ────────────────────────────────────────────────────
    'ি': ['কিছু', 'নিজ', 'বিশ্ব', 'দিন', 'রিকশা', 'চিঠি', 'ভিড়', 'নিম',
           'বিষয়', 'তিন', 'গির', 'পিতা'],
    'ী': ['নীল', 'জীবন', 'শীত', 'তীর', 'খীর', 'বীর', 'ধীর', 'নারী',
           'বাড়ী', 'পাখী', 'রানী', 'দাদী'],
    # ── Diacritics: ু vs ূ ────────────────────────────────────────────────────
    'ু': ['কুল', 'তুমি', 'দুই', 'মুখ', 'গুণ', 'ফুল', 'খুশি', 'সুর',
           'ভুল', 'কুকুর', 'তুলা', 'মুরগি'],
    'ূ': ['ভূমি', 'মূল', 'পূর্ব', 'শূন্য', 'সূর্য', 'তূণ', 'ভূল', 'মূলা',
           'কূল', 'ঝূলা', 'বূলি', 'পূর্ণ'],
    # ── Marks: ং vs ঃ vs ঁ ───────────────────────────────────────────────────
    'ং': ['বাংলা', 'রং', 'সংখ্যা', 'অংশ', 'ঢং', 'সংকট', 'রংপুর', 'ঢাকাং',
           'বাংলাদেশ', 'সংসার', 'সংঘ', 'মংলা'],
    'ঃ': ['দুঃখ', 'নিঃশ্বাস', 'অতঃপর', 'পুনঃ', 'প্রাতঃ', 'মূলতঃ',
           'বস্তুতঃ', 'সংক্ষেপতঃ'],
    'ঁ': ['চাঁদ', 'বাঁশ', 'হাঁটা', 'পাঁচ', 'মাঁ', 'সাঁতার', 'গাঁও', 'কাঁচ',
           'বাঁকা', 'ঘাঁটি', 'ধাঁধা', 'পাঁজর'],
    # ── ছ vs ঢ ────────────────────────────────────────────────────────────────
    'ছ': ['ছাত্র', 'ছবি', 'ছোট', 'ছেলে', 'ছাদ', 'ছায়া', 'ছুটি', 'ছুরি',
           'ছাপ', 'ছেড়া', 'ছিল', 'ছাগল'],
    # ── ফ vs ব ────────────────────────────────────────────────────────────────
    'ফ': ['ফুল', 'ফল', 'ফাঁকি', 'ফিরে', 'ফেরা', 'ফাঁদ', 'ফোন', 'ফজর',
           'ফ্লাট', 'ফাঁস', 'ফোঁটা', 'ফালা'],
    # ── ম vs ন (at small sizes) ───────────────────────────────────────────────
    'ম': ['মাঠ', 'মানুষ', 'মেঘ', 'মাছ', 'মন', 'মাস', 'মুখ', 'মাটি',
           'মেলা', 'মারা', 'মিষ্টি', 'মোড়'],
    # ── হ vs ব ────────────────────────────────────────────────────────────────
    'হ': ['হাত', 'হেঁটে', 'হওয়া', 'হলুদ', 'হিসাব', 'হাঁটা', 'হিম', 'হেলান',
           'হোক', 'হালকা', 'হিল', 'হামলা'],
    # ── ৎ vs ত (single-char isolation) ──────────────────────────────────────
    'ৎ': ['হঠাৎ', 'তৎপর', 'বাৎসরিক', 'তৎকাল', 'অকস্মাৎ', 'তৎক্ষণাৎ',
           'সাৎ', 'মাৎস্য', 'নিমেষাৎ', 'সহসাৎ', 'তৎসম', 'তৎপরতা'],
}


def _generate_confusion_data(confusion_count, output_dir, image_dir, reset):
    """Generate images targeting visually similar Bangla character pairs."""
    print(f"\n── Confusion-pair training ({confusion_count}/char × "
          f"{len(_BN_CONFUSION_CHARS)} chars) ────────────")

    fonts = load_fonts('bn')
    plan_path = os.path.join(output_dir, '.plan_confusion.json')

    if not reset and os.path.exists(plan_path):
        print(f"  Resuming from {plan_path}")
        plan = _load_json(plan_path)
    else:
        plan = []
        idx = 0
        for char, words in _BN_CONFUSION_CHARS.items():
            for _ in range(confusion_count):
                word = random.choice(words)
                # Vary font size: small (12-20), medium (22-36), large (38-54)
                size_band = random.choices(['s', 'm', 'l'], weights=[3, 5, 2])[0]
                if size_band == 's':
                    fs = random.randint(12, 20)
                elif size_band == 'm':
                    fs = random.randint(22, 36)
                else:
                    fs = random.randint(38, 54)
                plan.append({
                    'idx': idx,
                    'text': word,
                    'char': char,
                    'font_size': fs,
                    'image_path': os.path.join(
                        output_dir, 'images', f'bn_conf_{idx}.png'),
                    'label_path': os.path.join(
                        output_dir, 'labels', f'bn_conf_{idx}.txt'),
                })
                idx += 1
        _save_json(plan, plan_path)

    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    print(f"  {len(plan) - len(todo)} done, {len(todo)} remaining")

    for item in tqdm(todo, desc='Confusion training'):
        font = random.choice(fonts)
        img = None
        attempts = 0
        while img is None and attempts < 5:
            img = FakeTextDataGenerator.generate(
                index=item['idx'],
                text=item['text'],
                font=font,
                out_dir=None,
                size=item['font_size'],
                extension=None,
                skewing_angle=0,
                random_skew=False,
                blur=random.randint(0, 1),
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
                word_split=False,
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
        os.makedirs(os.path.dirname(item['image_path']), exist_ok=True)
        os.makedirs(os.path.dirname(item['label_path']), exist_ok=True)
        img.save(item['image_path'])
        with open(item['label_path'], 'w', encoding='utf-8') as f:
            f.write(item['text'])


def _rotate_image(img, max_angle=3):
    angle = random.uniform(-max_angle, max_angle)
    if abs(angle) < 0.3:
        return img
    w, h = img.size
    pad = 20  # sufficient for ±3° on any image size
    arr = np.array(img)
    if arr.ndim == 3:
        arr_padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    else:
        arr_padded = np.pad(arr, ((pad, pad), (pad, pad)), mode='edge')
    img_padded = PILImage.fromarray(arr_padded)
    img_rotated = img_padded.rotate(angle, expand=False, resample=PILImage.BILINEAR)
    return img_rotated.crop((pad, pad, pad + w, pad + h))


def _apply_augraphy(img, moire_prob=0.8, fade_prob=0.8):
    """Apply random moiré pattern and/or lighting-fade using augraphy. No-op if not installed."""
    try:
        from augraphy import MoirePattern, LightingGradient
    except ImportError:
        return img

    arr = np.array(img.convert('RGB'))
    changed = False

    if random.random() < moire_prob:
        result = MoirePattern()(arr)
        arr = result if isinstance(result, np.ndarray) else result.get('output', arr)
        changed = True

    if random.random() < fade_prob:
        result = LightingGradient()(arr)
        arr = result if isinstance(result, np.ndarray) else result.get('output', arr)
        changed = True

    if not changed:
        return img

    return PILImage.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


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
        skewing_angle=0,
        random_skew=False,
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
        img = _rotate_image(img)
        img = _apply_augraphy(img)
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
        profile_size, profile_blur = _pick_addr_profile()
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
                skewing_angle=0,
                random_skew=False,
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

        img = _rotate_image(img)
        img = _apply_augraphy(img)
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
    parser.add_argument('--confusion_count', type=int, default=0,
                        help='Images per confusable char (0=skip, e.g. 300)')
    parser.add_argument('--prepare_hf', action='store_true',
                        help='Run prepare_hf_dataset.py after generation')
    parser.add_argument('--shamadhan_dir', type=str, default='/app/data',
                        help='Real shamadhan image dir for prepare_hf_dataset.py')
    parser.add_argument('--hf_dir', type=str, default='/app/hf_dataset',
                        help='Output HuggingFace dataset dir for prepare_hf_dataset.py')
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

    print("\nShamadhan class distribution (before augmentation):")
    for lang in ['bn', 'en']:
        if texts_by_lang[lang]:
            total = sum(len(v) for v in texts_by_lang[lang].values())
            print(f"  {lang.upper()}: {len(texts_by_lang[lang])} classes, {total:,} texts")

    # ── Synthetic augmentation ─────────────────────────────────────────────────
    # Expand name/NID/DOB pools so text_count can be met from diverse samples.
    synth_target = max(500, args.text_count * 3)
    print(f"\nAugmenting to {synth_target} samples per class...")
    for lang in ['bn', 'en']:
        if not texts_by_lang[lang]:
            continue
        before = {k: len(v) for k, v in texts_by_lang[lang].items()}
        texts_by_lang[lang] = _augment_class_dict(
            texts_by_lang[lang], synth_target, is_english=(lang == 'en')
        )
        for cls, cnt_before in sorted(before.items()):
            cnt_after = len(texts_by_lang[lang][cls])
            if cnt_after > cnt_before:
                print(f"  [{lang.upper()}] {cls}: {cnt_before:,} → {cnt_after:,} (+{cnt_after - cnt_before:,})")

    # ── Bangla character coverage injection ───────────────────────────────────
    if texts_by_lang['bn']:
        print("\nBangla character coverage (before injection):")
        _report_bn_coverage(texts_by_lang['bn'])
        _inject_char_coverage(texts_by_lang['bn'], min_reps=30)
        print("Bangla character coverage (after injection):")
        _report_bn_coverage(texts_by_lang['bn'])

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

    # ── Confusion-pair training images ────────────────────────────────────────
    if args.confusion_count > 0:
        _generate_confusion_data(
            confusion_count=args.confusion_count,
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
        subprocess.run([
            sys.executable, 'prepare_hf_dataset.py',
            '--output_dir', args.output_dir,
            '--shamadhan_dir', args.shamadhan_dir,
            '--dataset_dir', args.hf_dir,
        ], check=True)

    img_dir = os.path.join(args.output_dir, 'images')
    total = len([f for f in os.listdir(img_dir) if f.endswith('.png')]) if os.path.isdir(img_dir) else 0
    print(f"\nDone. {total:,} total images in {args.output_dir}/")


if __name__ == '__main__':
    main()
