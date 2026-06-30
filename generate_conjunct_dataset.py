"""
generate_conjunct_dataset.py

5000 Bengali conjunct (যুক্তবর্ণ) OCR training images.
Every conjunct in _CONJUNCT_MAP gets at least one image; samples are distributed
evenly across all clusters (5000 / len(clusters) ≈ 21 each).

Output:
  {output_dir}/images/   — PNG images
  {output_dir}/labels/   — matching .txt labels
  {output_dir}/hf/       — sharded HF Parquet dataset

Usage:
  python3 generate_conjunct_dataset.py \\
    --output_dir /app/conjunct_output \\
    --total 5000 \\
    [--push_to_hub kavinh07/bn-conjunct-ocr] \\
    [--hf_token TOKEN] [--reset]
"""

import argparse
import io
import math
import os
import random
import sys

import numpy as np
from PIL import Image as PILImage, ImageFilter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_combined import (
    _apply_augraphy, _load_json, _maybe_downscale, _save_json,
)
from trdg.data_generator import FakeTextDataGenerator
from trdg.utils import load_fonts


# ── Conjunct → word list mapping ─────────────────────────────────────────────
# Every conjunct the user listed is covered. Words chosen so the cluster
# appears clearly (ideally not at a syllable boundary that gets split across lines).

_CONJUNCT_MAP = {
    # ক-series
    'ক্ক':    ['মক্কা', 'চাক্কা', 'ধাক্কা', 'টক্কর'],
    'ক্ট':    ['ডক্টর', 'ট্র্যাক্টর', 'স্ট্রিক্ট', 'ডিরেক্ট'],
    'ক্ত':    ['শক্তি', 'ভক্তি', 'রক্ত', 'মুক্তি', 'যুক্তি'],
    'ক্ত্র':   ['বক্ত্র', 'ক্ষত্রিয়'],
    'ক্ব':    ['পক্ব', 'ক্বচিৎ'],
    'ক্ম':    ['বাক্ময়', 'রুক্ম'],
    'ক্য':    ['বাক্য', 'বক্য', 'শক্য'],
    'ক্র':    ['চক্র', 'ক্রম', 'আক্রমণ', 'বিক্রম'],
    'ক্ল':    ['শুক্ল', 'ক্লান্ত', 'ক্লাস', 'ক্লেদ'],
    'ক্ষ':    ['ক্ষমা', 'লক্ষ্য', 'পক্ষ', 'রক্ষা', 'ক্ষতি'],
    'ক্ষ্ণ':   ['তীক্ষ্ণ', 'সূক্ষ্ণ'],
    'ক্ষ্ব':   ['ক্ষ্বেদ'],
    'ক্ষ্ম':   ['লক্ষ্মী', 'সূক্ষ্ম'],
    'ক্ষ্য':   ['লক্ষ্য', 'মোক্ষ্য', 'ক্ষ্যাপা'],
    'ক্স':    ['বক্স', 'ট্যাক্স', 'সিক্স', 'মিক্স'],
    # খ-series
    'খ্য':    ['বিখ্যাত', 'সুখ্যাত', 'সখ্য'],
    'খ্র':    ['খ্রিস্ট', 'খ্রিষ্টান', 'খ্রিস্টাব্দ'],
    # গ-series
    'গ্ণ':    ['ভগ্ণ', 'দগ্ণ', 'বিভগ্ণ'],
    'গ্ধ':    ['মুগ্ধ', 'দুগ্ধ', 'বিমুগ্ধ', 'মুগ্ধতা'],
    'গ্ধ্য':   ['মুগ্ধ্য', 'দুগ্ধ্য'],
    'গ্ধ্র':   ['দুগ্ধ্র'],
    'গ্ন':    ['মগ্ন', 'ভগ্ন', 'নিমগ্ন'],
    'গ্ন্য':   ['অগ্ন্যাশয়', 'প্রাগ্ন্য'],
    'গ্ব':    ['দিগ্বিদিক', 'বাগ্বিতণ্ডা'],
    'গ্ম':    ['যুগ্ম', 'বাগ্মী', 'প্রগ্ম'],
    'গ্য':    ['ভাগ্য', 'যোগ্য', 'অযোগ্য', 'সৌভাগ্য'],
    'গ্র':    ['গ্রাম', 'গ্রন্থ', 'আগ্রহ', 'বিগ্রহ'],
    'গ্র্য':   ['অগ্র্য', 'বৈগ্র্য'],
    'গ্ল':    ['গ্লাস', 'গ্লানি', 'গ্লোব', 'গ্লেসিয়ার'],
    # ঘ-series
    'ঘ্ন':    ['অশ্বঘ্ন', 'পাপঘ্ন', 'দুঃখঘ্ন'],
    'ঘ্য':    ['ঘৃণ্য', 'দুর্ঘ্য'],
    'ঘ্র':    ['ঘ্রাণ', 'আঘ্রাণ', 'ঘ্রাণশক্তি'],
    # ঙ-series
    'ঙ্ক':    ['শঙ্কা', 'আতঙ্ক', 'কলঙ্ক', 'শঙ্কিত'],
    'ঙ্ক্য':   ['শঙ্ক্য', 'কলঙ্ক্য'],
    'ঙ্ক্ষ':   ['আকাঙ্ক্ষা', 'প্রতিকাঙ্ক্ষা'],
    'ঙ্খ':    ['শঙ্খ', 'শঙ্খচিল', 'পঙ্খিরাজ'],
    'ঙ্খ্য':   ['শঙ্খ্য'],
    'ঙ্গ':    ['অঙ্গ', 'সঙ্গ', 'রঙিন', 'অঙ্গন'],
    'ঙ্গ্য':   ['অঙ্গ্য'],
    'ঙ্ঘ':    ['সঙ্ঘ', 'লঙ্ঘন', 'অতিলঙ্ঘন'],
    'ঙ্ঘ্য':   ['লঙ্ঘ্য', 'সঙ্ঘ্য'],
    'ঙ্ঘ্র':   ['সঙ্ঘ্র'],
    'ঙ্ম':    ['বাঙ্ময়', 'বাঙ্ময়'],
    # চ-series
    'চ্চ':    ['বাচ্চা', 'উচ্চ', 'স্বচ্ছ', 'উচ্চতা'],
    'চ্ছ':    ['ইচ্ছা', 'স্বচ্ছ', 'প্রচ্ছদ', 'সচ্ছল'],
    'চ্ছ্ব':   ['বিচ্ছ্বাস'],
    'চ্ছ্র':   ['প্রচ্ছ্র'],
    'চ্ঞ':    ['যাচ্ঞা'],
    'চ্ব':    ['চ্বিন্তা', 'কাচ্ব'],
    'চ্য':    ['বাচ্য', 'পাচ্য', 'অনুপাচ্য'],
    # জ-series
    'জ্জ':    ['সজ্জা', 'উজ্জ্বল', 'বিজ্জু'],
    'জ্জ্ব':   ['উজ্জ্বল', 'উজ্জ্বলতা'],
    'জ্ঝ':    ['কুজ্ঝটিকা'],
    'জ্ঞ':    ['জ্ঞান', 'বিজ্ঞান', 'আজ্ঞা', 'জ্ঞানী'],
    'জ্ব':    ['জ্বর', 'জ্বলন', 'জ্বালা', 'জ্বী'],
    'জ্য':    ['রাজ্য', 'সাজ্য', 'বাজ্য'],
    'জ্র':    ['বজ্র', 'বজ্রপাত'],
    # ঞ-series
    'ঞ্চ':    ['পঞ্চম', 'কাঞ্চন', 'অঞ্চল', 'পঞ্চাশ'],
    'ঞ্ছ':    ['মাঞ্ছা'],
    'ঞ্জ':    ['রঞ্জন', 'গঞ্জ', 'কুঞ্জ', 'গুঞ্জন'],
    'ঞ্ঝ':    ['ঝঞ্ঝা', 'ঝঞ্ঝাট'],
    # ট-series
    'ট্ট':    ['ভট্টাচার্য', 'ঘট্টন', 'ট্টমন'],
    'ট্ব':    ['খট্ব', 'ট্বিন'],
    'ট্ম':    ['কুট্মল'],
    'ট্য':    ['নাট্য', 'বাট্য', 'নাট্যশালা'],
    'ট্র':    ['ট্রেন', 'ট্রাক', 'ট্রাম', 'ট্রফি'],
    # ড-series
    'ড্ড':    ['আড্ডা', 'খড্ড', 'বাড্ডা'],
    'ড্ব':    ['দ্বীপ', 'ড্বিতীয়'],
    'ড্ম':    ['কড্ম'],
    'ড্য':    ['ড্যাম', 'মড্য'],
    'ড্র':    ['ড্রাম', 'ড্রাইভার', 'ড্রেস', 'ড্রয়িং'],
    # ঢ-series
    'ঢ্য':    ['ঢ্যাড়স', 'ঢ্যামনা'],
    'ঢ্র':    ['ঢ্রিম', 'ঢ্রিলিং'],
    # ণ-series
    'ণ্ট':    ['ঘণ্টা', 'দণ্ট', 'ঘণ্টাধ্বনি'],
    'ণ্ঠ':    ['কণ্ঠ', 'কণ্ঠস্বর', 'কণ্ঠহার'],
    'ণ্ঠ্য':   ['কণ্ঠ্য', 'কণ্ঠ্যধ্বনি'],
    'ণ্ড':    ['মণ্ডল', 'দণ্ড', 'খণ্ড', 'মণ্ডলী'],
    'ণ্ড্য':   ['দণ্ড্য', 'খণ্ড্য'],
    'ণ্ড্র':   ['মণ্ড্র', 'মণ্ড্রল'],
    'ণ্ঢ':    ['ণ্ঢাকা', 'বিণ্ঢ'],
    'ণ্ণ':    ['বিষণ্ণ', 'নিষণ্ণ', 'শূণ্য'],
    'ণ্ব':    ['গণ্বন', 'সণ্বত'],
    'ণ্ম':    ['ণ্মোচন', 'উণ্মোচন'],
    'ণ্য':    ['পণ্য', 'গণ্য', 'মান্য', 'পণ্যদ্রব্য'],
    # ত-series
    'ত্ত':    ['উত্তর', 'সত্তা', 'চিত্ত', 'বিত্ত'],
    'ত্ত্ব':   ['তত্ত্ব', 'সত্ত্ব', 'তত্ত্বাবধান'],
    'ত্ত্য':   ['তত্ত্য'],
    'ত্থ':    ['উত্থান', 'প্রত্যুত্থান', 'উত্থাপন'],
    'ত্ন':    ['যত্ন', 'রত্ন', 'রত্নখচিত'],
    'ত্ব':    ['স্বত্ব', 'ত্বক', 'কর্তৃত্ব'],
    'ত্ম':    ['আত্মা', 'পরমাত্মা', 'আত্মহত্যা'],
    'ত্ম্য':   ['আত্ম্য'],
    'ত্য':    ['সত্য', 'নিত্য', 'অবিত্য'],
    'ত্র':    ['মাত্র', 'ত্রাণ', 'চরিত্র', 'পত্র'],
    'ত্র্য':   ['ত্র্যম্বক', 'বৈচিত্র্য', 'স্বাতন্ত্র্য'],
    # থ-series
    'থ্ব':    ['পৃথ্বী', 'থ্বক'],
    'থ্য':    ['স্বাস্থ্য', 'তথ্য', 'মিথ্য'],
    'থ্র':    ['থ্রিলার', 'থ্রি', 'থ্রেড'],
    # দ-series
    'দ্গ':    ['উদ্গার', 'উদ্গম', 'উদ্গীরণ'],
    'দ্ঘ':    ['উদ্ঘাটন', 'উদ্ঘোষণা'],
    'দ্দ':    ['উদ্দেশ্য', 'আদ্দা', 'মদ্দ'],
    'দ্দ্ব':   ['উদ্দ্বেগ'],
    'দ্ধ':    ['বুদ্ধ', 'শুদ্ধ', 'রুদ্ধ', 'বিদ্ধ'],
    'দ্ব':    ['বিদ্বান', 'দ্বিধা', 'দ্বীপ', 'দ্বন্দ্ব'],
    'দ্ভ':    ['অদ্ভুত', 'উদ্ভব', 'উদ্ভিদ'],
    'দ্ভ্র':   ['উদ্ভ্রান্ত'],
    'দ্ম':    ['পদ্ম', 'পদ্মা', 'পদ্মফুল'],
    'দ্য':    ['বিদ্যা', 'বিদ্যালয়', 'মহাবিদ্যালয়'],
    'দ্র':    ['দ্রুত', 'সমুদ্র', 'আদ্র', 'দ্রব্য'],
    'দ্র্য':   ['দারিদ্র্য', 'দ্রব্য'],
    # ধ-series
    'ধ্ন':    ['ধন্য', 'ধন্যবাদ'],
    'ধ্ব':    ['ধ্বনি', 'ধ্বংস', 'ধ্বজা'],
    'ধ্ম':    ['ধ্মান', 'আধ্মান'],
    'ধ্য':    ['সাধ্য', 'মধ্য', 'অধ্যায়'],
    'ধ্র':    ['ধ্রুব', 'ধ্রুপদ', 'ধ্রুবক'],
    # ন-series
    'ন্ট':    ['প্রিন্ট', 'পেইন্ট', 'পয়েন্ট', 'পেন্টিং'],
    'ন্ট্র':   ['কন্ট্রোল', 'মন্ট্র', 'কন্ট্রাক্ট'],
    'ন্ঠ':    ['কণ্ঠ', 'ন্ঠের'],
    'ন্ড':    ['মান্ডা', 'ইন্ডিয়া', 'বান্ডেল', 'ইন্ডেক্স'],
    'ন্ড্র':   ['ইন্দ্র', 'সিলিন্ডার', 'ইন্ড্রাষ্ট্রি'],
    'ন্ত':    ['শান্ত', 'অন্ত', 'অন্তর', 'শান্তি'],
    'ন্ত্ব':   ['মন্ত্ব', 'প্রান্তব'],
    'ন্ত্য':   ['অন্ত্য', 'গন্ত্য'],
    'ন্ত্র':   ['মন্ত্র', 'যন্ত্র', 'তন্ত্র', 'মন্ত্রণা'],
    'ন্ত্র্য':  ['স্বাতন্ত্র্য'],
    'ন্থ':    ['গ্রন্থ', 'পন্থা', 'গ্রন্থাগার'],
    'ন্থ্র':   ['অ্যান্থ্র', 'ন্থ্রপলজি'],
    'ন্দ':    ['আনন্দ', 'বন্দর', 'চন্দন', 'বন্দনা'],
    'ন্দ্য':   ['বন্দ্যোপাধ্যায়', 'বন্দ্য'],
    'ন্দ্ব':   ['দ্বন্দ্ব'],
    'ন্দ্র':   ['চন্দ্র', 'ইন্দ্র', 'চন্দ্রমা'],
    'ন্ধ':    ['বন্ধু', 'সন্ধ্যা', 'অন্ধ', 'বন্ধন'],
    'ন্ধ্য':   ['সন্ধ্যা', 'বিন্ধ্য'],
    'ন্ধ্র':   ['অন্ধ্র', 'ন্ধ্রতা'],
    'ন্ন':    ['নিন্দা', 'ছিন্ন', 'ভিন্ন', 'খিন্ন'],
    'ন্ব':    ['অন্বয়', 'অন্বিত', 'অন্বেষণ'],
    'ন্ম':    ['জন্ম', 'উন্মাদ', 'উন্মোচন', 'জন্মদিন'],
    'ন্য':    ['মান্য', 'ধন্য', 'সম্মান্য'],
    # প-series
    'প্ট':    ['অ্যাপ্ট', 'কনসেপ্ট'],
    'প্ত':    ['গুপ্ত', 'লুপ্ত', 'আপ্ত', 'সুপ্ত'],
    'প্ন':    ['স্বপ্ন', 'তৃপ্ন'],
    'প্প':    ['চাপ্পা', 'টপ্পা', 'হাপ্পা'],
    'প্য':    ['প্রাপ্য', 'অপ্য', 'গ্রাহ্য'],
    'প্র':    ['প্রেম', 'প্রাণ', 'প্রতি', 'প্রবেশ'],
    'প্র্য':   ['প্র্যাক্টিস', 'প্র্যাকটিক্যাল'],
    'প্ল':    ['প্লেট', 'প্লান', 'প্লাস্টিক', 'প্লাটফর্ম'],
    'প্স':    ['গিপ্স', 'টিপ্স'],
    # ফ-series
    'ফ্র':    ['ফ্রান্স', 'ফ্রি', 'ফ্রেম', 'ফ্রিজ'],
    'ফ্ল':    ['ফ্লোর', 'ফ্লাই', 'ফ্লাট', 'ফ্লেক্স'],
    # ব-series
    'ব্জ':    ['ব্জার', 'বাব্জা'],
    'ব্দ':    ['শব্দ', 'অব্দ', 'প্রতিশব্দ'],
    'ব্ধ':    ['লব্ধ', 'প্রলব্ধ', 'প্রতিব্ধ'],
    'ব্ব':    ['ডাব্বা', 'তব্বা', 'জব্বার'],
    'ব্য':    ['ব্যক্তি', 'ব্যবসা', 'ব্যাপার', 'ব্যবস্থা'],
    'ব্র':    ['ব্রিজ', 'ব্রাহ্মণ', 'ব্রাকেট', 'ব্রেক'],
    'ব্ল':    ['ব্লক', 'ব্লু', 'ব্লাড', 'ব্লেড'],
    # ভ-series
    'ভ্ব':    ['ভ্বান'],
    'ভ্য':    ['সভ্য', 'অসভ্য', 'ভদ্রভ্য'],
    'ভ্র':    ['ভ্রমণ', 'ভ্রাতা', 'ভ্রূণ', 'ভ্রমর'],
    'ভ্ল':    ['ভ্লাদিমির'],
    # ম-series
    'ম্ন':    ['নিম্ন', 'নিম্নমান', 'অধোনিম্ন'],
    'ম্প':    ['কম্পন', 'তম্পা', 'ল্যাম্প', 'কম্পিউটার'],
    'ম্প্র':   ['সম্প্রতি', 'সম্প্রদায়', 'সম্প্রসারণ'],
    'ম্ফ':    ['ট্রাম্ফ', 'ম্ফল'],
    'ম্ব':    ['লম্বা', 'সম্বল', 'অম্বর', 'সম্বন্ধ'],
    'ম্ব্র':   ['অম্ব্র', 'চেম্ব্র'],
    'ম্ভ':    ['সম্ভব', 'সম্ভার', 'দম্ভ'],
    'ম্ভ্র':   ['সম্ভ্রম', 'গাম্ভীর্য'],
    'ম্ম':    ['সম্মান', 'সম্মতি', 'আম্মা', 'মম্মি'],
    'ম্য':    ['ম্যাচ', 'ম্যাপ', 'গ্রাম্য'],
    'ম্র':    ['ম্রিয়মাণ'],
    'ম্ল':    ['অম্ল', 'অম্লজান'],
    # য-series
    'য্য':    ['মহাশয্য', 'আয্য'],
    # ল-series
    'ল্ক':    ['শল্ক', 'ফল্ক'],
    'ল্ক্য':   ['যাজ্ঞবল্ক্য', 'শল্ক্য'],
    'ল্গ':    ['বল্গা', 'ল্গু'],
    'ল্ট':    ['বল্টু', 'ল্টার'],
    'ল্ড':    ['বল্ড', 'ল্ডার'],
    'ল্প':    ['কল্প', 'সংকল্প', 'কল্পনা'],
    'ল্ফ':    ['গল্ফ', 'স্কাল্ফ'],
    'ল্ব':    ['বল্ব', 'ল্বিত'],
    'ল্ভ':    ['দুর্লভ', 'সুলভ'],
    'ল্ম':    ['ফিল্ম', 'হেলমেট'],
    'ল্য':    ['বল্য', 'ল্যাব', 'ল্যান্ড'],
    'ল্ল':    ['উল্লাস', 'উল্লেখ', 'উল্লম্ব'],
    # শ-series
    'শ্চ':    ['নিশ্চিত', 'পশ্চিম', 'বিশ্চ'],
    'শ্ছ':    ['শ্ছেদ', 'বিশ্ছেদ'],
    'শ্ন':    ['প্রশ্ন', 'বিশ্ন', 'প্রশ্নপত্র'],
    'শ্ব':    ['বিশ্ব', 'বিশ্বাস', 'বিশ্বায়ন'],
    'শ্ম':    ['শ্মশান', 'শ্মাশান'],
    'শ্য':    ['শ্যামল', 'শ্যাম', 'শ্যামলী'],
    'শ্র':    ['শ্রম', 'শ্রেণী', 'শ্রমিক', 'শ্রদ্ধা'],
    'শ্ল':    ['শ্লোক', 'শ্লেষ'],
    # ষ-series
    'ষ্ক':    ['শুষ্ক', 'রুক্ষ', 'ভিষ্ক'],
    'ষ্ক্ব':   ['পষ্ক্ব'],
    'ষ্ক্র':   ['পুষ্ক্র'],
    'ষ্ট':    ['কষ্ট', 'স্পষ্ট', 'দুষ্ট', 'বিষ্ট'],
    'ষ্ট্য':   ['ষষ্ট্য'],
    'ষ্ট্র':   ['রাষ্ট্র', 'রাষ্ট্রীয়', 'রাষ্ট্রপতি'],
    'ষ্ঠ':    ['নিষ্ঠা', 'শ্রেষ্ঠ', 'প্রতিষ্ঠা'],
    'ষ্ঠ্য':   ['কনিষ্ঠ্য'],
    'ষ্ণ':    ['কৃষ্ণ', 'উষ্ণ', 'বিষ্ণু'],
    'ষ্ণ্ব':   ['কৃষ্ণ্বর্ণ'],
    'ষ্প':    ['পুষ্প', 'পুষ্পিত', 'পুষ্পমালা'],
    'ষ্প্র':   ['পুষ্প্রভা'],
    'ষ্ফ':    ['ষ্ফুট'],
    'ষ্ব':    ['বৈষ্ণব', 'ষ্বাস'],
    'ষ্ম':    ['উষ্মা', 'উষ্মীয়'],
    'ষ্য':    ['শিষ্য', 'শিষ্যত্ব', 'ষষ্ঠী'],
    # স-series
    'স্ক':    ['স্কুল', 'স্কার', 'স্কেল', 'স্কোর'],
    'স্ক্র':   ['স্ক্রিন', 'স্ক্রু', 'স্ক্রিপ্ট'],
    'স্ক্ল':   ['স্ক্লেরা'],
    'স্খ':    ['স্খলন', 'স্খলিত'],
    'স্ট':    ['স্টেশন', 'স্টার', 'স্টাফ', 'স্টোর'],
    'স্ট্র':   ['স্ট্রিট', 'স্ট্রোক', 'স্ট্রাকচার'],
    'স্ত':    ['স্তর', 'স্তব', 'বস্ত', 'স্তম্ভ'],
    'স্ত্ব':   ['বস্ত্ব', 'স্তব্ধ'],
    'স্ত্য':   ['সস্ত্য'],
    'স্ত্র':   ['বস্ত্র', 'অস্ত্র', 'শাস্ত্র'],
    'স্থ':    ['স্থান', 'স্থায়ী', 'স্বাস্থ্য', 'স্থগিত'],
    'স্থ্য':   ['স্বাস্থ্য', 'অস্থ্য'],
    'স্ন':    ['স্নান', 'স্নাতক', 'স্নাতকোত্তর'],
    'স্ন্য':   ['স্ন্য'],
    'স্প':    ['স্পষ্ট', 'স্পর্শ', 'স্পন্দন', 'স্পর্ধা'],
    'স্প্র':   ['স্প্রিং', 'স্প্রে'],
    'স্প্ল':   ['স্প্লিট'],
    'স্ফ':    ['স্ফীত', 'স্ফুলিঙ্গ'],
    'স্ব':    ['স্বাধীন', 'স্বপ্ন', 'স্বাস্থ্য', 'স্বদেশ'],
    'স্ম':    ['স্মরণ', 'স্মৃতি', 'স্মৃতিসৌধ'],
    'স্য':    ['স্যার', 'স্যামসাং'],
    'স্র':    ['স্রোত', 'স্রষ্টা'],
    'স্ল':    ['স্লোগান', 'স্লিপ'],
    # হ-series
    'হ্ণ':    ['ব্রাহ্মণ', 'হ্ণীকরণ'],
    'হ্ন':    ['চিহ্ন', 'লিখ্ন', 'চিহ্নিত'],
    'হ্ব':    ['বিহ্বল', 'হ্বান'],
    'হ্ম':    ['ব্রহ্ম', 'ব্রাহ্মণ', 'ব্রহ্মাণ্ড'],
    'হ্য':    ['সহ্য', 'অসহ্য', 'দুঃসহ্য'],
    'হ্র':    ['হ্রাস', 'হ্রদ', 'অবহ্রাস'],
    'হ্ল':    ['হ্লাদ', 'আহ্লাদ'],
    # ড়-series
    'ড়্গ':   ['ড়্গা', 'বাড়্গা'],
    # রেফ (র্ + consonant) — র comes as repha before the base consonant
    'র্ক':    ['কর্ক', 'মার্কিন', 'চিহ্নর্ক'],
    'র্ক্য':   ['র্ক্য'],
    'র্খ':    ['র্খত', 'কার্খানা'],
    'র্গ':    ['মর্গ', 'দর্গা', 'বার্গার'],
    'র্গ্য':   ['র্গ্য'],
    'র্গ্র':   ['র্গ্র'],
    'র্ঘ':    ['দীর্ঘ', 'দীর্ঘশ্বাস'],
    'র্ঘ্য':   ['দীর্ঘ্য'],
    'র্ঙ্গ':   ['বর্ঙ্গ'],
    'র্চ':    ['মার্চ', 'বার্চ', 'র্চনা'],
    'র্চ্য':   ['র্চ্য'],
    'র্ছ':    ['র্ছেদ', 'বিচ্ছেদ'],
    'র্জ':    ['র্জন', 'বর্জন', 'অর্জন'],
    'র্জ্য':   ['বর্জ্য', 'র্জ্য'],
    'র্জ্জ':   ['র্জ্জ'],
    'র্জ্ঞ':   ['র্জ্ঞ'],
    'র্ঝ':    ['র্ঝর'],
    'র্ট':    ['কার্ট', 'র্টার', 'স্মার্ট'],
    'র্ড':    ['কার্ড', 'র্ডার', 'বোর্ড'],
    'র্ঢ্য':   ['র্ঢ্য'],
    'র্ণ':    ['বর্ণ', 'কর্ণ', 'পর্ণ', 'বর্ণমালা'],
    'র্ণ্য':   ['বর্ণ্য', 'র্ণ্য'],
    'র্ত':    ['কর্তা', 'বর্তমান', 'কর্তৃপক্ষ'],
    'র্ত্য':   ['মর্ত্য', 'অমর্ত্য'],
    'র্ত্ম':   ['কর্ত্ম', 'বর্ত্ম'],
    'র্ত্র':   ['র্ত্র'],
    'র্থ':    ['অর্থ', 'সার্থক', 'অর্থনীতি'],
    'র্থ্য':   ['সামর্থ্য', 'দক্ষতার্থ্য'],
    'র্দ':    ['গর্দান', 'বর্দার', 'বর্দাশত'],
    'র্দ্ব':   ['র্দ্ব'],
    'র্দ্র':   ['আর্দ্র', 'আর্দ্রতা'],
    'র্ধ':    ['বর্ধন', 'অর্ধ', 'অর্ধেক'],
    'র্ধ্ব':   ['ঊর্ধ্ব', 'ঊর্ধ্বমুখী'],
    'র্ন':    ['পর্ন', 'বর্ন', 'মডার্ন'],
    'র্প':    ['কর্প', 'তর্পণ', 'শার্প'],
    'র্ফ':    ['স্কার্ফ', 'টার্ফ'],
    'র্ব':    ['গর্ব', 'সর্বনাম', 'সর্বোচ্চ'],
    'র্ব্য':   ['র্ব্য'],
    'র্ভ':    ['গর্ভ', 'গর্ভবতী', 'র্ভনী'],
    'র্ম':    ['ধর্ম', 'কর্ম', 'চর্ম', 'ধর্মীয়'],
    'র্ম্য':   ['ধর্ম্য', 'কর্ম্য'],
    'র্য':    ['কার্য', 'বার্য', 'কার্যকর'],
    'র্ল':    ['র্লাই', 'বার্লি'],
    'র্শ':    ['বর্শা', 'স্পর্শ', 'দর্শন'],
    'র্শ্য':   ['অদর্শ্য', 'দর্শ্য'],
    'র্শ্ব':   ['র্শ্ব'],
    'র্ষ':    ['বর্ষ', 'বর্ষণ', 'কর্ষণ', 'বর্ষা'],
    'র্ষ্ট':   ['আকর্ষ্ট'],
    'র্ষ্ণ':   ['কর্ষ্ণ'],
    'র্ষ্ণ্য':  ['র্ষ্ণ্য'],
    'র্ষ্য':   ['র্ষ্য'],
    'র্স':    ['র্স', 'কোর্স', 'নার্স'],
    'র্হ':    ['র্হণ', 'আর্হ'],
    'র্হ্য':   ['র্হ্য'],
    'র্ৎ':    ['কোর্ৎ', 'বিদ্যুর্ৎ'],
}


# ── Augmentation helpers ──────────────────────────────────────────────────────

def _apply_low_ink(img):
    arr = np.array(img.convert('RGB')).astype(np.float32)
    arr = np.clip(arr + np.random.normal(0, random.uniform(5, 15), arr.shape), 0, 255).astype(np.uint8)
    img = PILImage.fromarray(arr)
    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=random.randint(60, 80))
    buf.seek(0)
    return PILImage.open(buf).convert('RGB')


# ── Plan building ─────────────────────────────────────────────────────────────

def build_plan(out_img, out_lbl, total):
    """Distribute `total` samples evenly across all conjuncts."""
    conjuncts = list(_CONJUNCT_MAP.keys())
    n = len(conjuncts)
    per_cluster = max(1, math.ceil(total / n))

    plan = []
    idx = 0
    for conj in conjuncts:
        words = _CONJUNCT_MAP[conj]
        # Cycle through the word list
        for i in range(per_cluster):
            text = words[i % len(words)]
            low_ink = random.random() < 0.4   # 40% faded variants
            plan.append({
                'idx':        idx,
                'conjunct':   conj,
                'text':       text,
                'low_ink':    low_ink,
                'image_path': os.path.join(out_img, f'conj_{idx:06d}.png'),
                'label_path': os.path.join(out_lbl, f'conj_{idx:06d}.txt'),
            })
            idx += 1

    # Trim or top-up to exactly `total`
    random.shuffle(plan)
    if len(plan) > total:
        plan = plan[:total]
    elif len(plan) < total:
        extra = random.choices(plan, k=total - len(plan))
        for i, item in enumerate(extra):
            new_item = dict(item)
            new_item['idx'] = idx + i
            fname = f'conj_{new_item["idx"]:06d}'
            new_item['image_path'] = os.path.join(out_img, f'{fname}.png')
            new_item['label_path'] = os.path.join(out_lbl, f'{fname}.txt')
            plan.append(new_item)

    # Re-assign sequential idx for dedup
    for i, item in enumerate(plan):
        item['idx'] = i

    return plan


# ── Image generation ──────────────────────────────────────────────────────────

def generate_images(plan, fonts, image_dir, font_size, blur, reset):
    todo = [p for p in plan if not os.path.exists(p['image_path'])]
    print(f"  {len(plan) - len(todo)} done, {len(todo)} remaining")

    for item in tqdm(todo, desc='Conjunct images'):
        font = random.choice(fonts)
        low_ink = item['low_ink']
        color = f'#{random.randint(0x1a, 0x55):02x}' * 3 if low_ink else '#000000'
        stroke = 0 if low_ink else (1 if random.random() < 0.5 else 0)

        img = None
        for _ in range(5):
            try:
                img = FakeTextDataGenerator.generate(
                    index=item['idx'],
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
                    text_color=color,
                    orientation=0,
                    space_width=1.0,
                    character_spacing=0,
                    margins=(5, 5, 5, 5),
                    fit=True,
                    output_mask=False,
                    word_split=True,
                    image_dir=image_dir,
                    stroke_width=stroke,
                    stroke_fill=color,
                    image_mode='RGB',
                )
                if img is not None:
                    break
            except Exception:
                img = None

        if img is None:
            continue

        if low_ink:
            img = _apply_low_ink(img)
        else:
            img = _apply_augraphy(img)

        img = _maybe_downscale(img)
        img.save(item['image_path'])
        with open(item['label_path'], 'w', encoding='utf-8') as f:
            f.write(item['text'])


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
        'conjunct':   Value('string'),
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
                        'conjunct':   item['conjunct'],
                        'class_name': 'conjunct_training',
                        'source':     'synthetic_conjunct',
                    }
                except Exception as e:
                    print(f"  skip {item['image_path']}: {e}")

        ds = Dataset.from_generator(gen, features=features)
        tmp = shard_path + '.tmp'
        ds.to_parquet(tmp)
        os.replace(tmp, shard_path)

    _write_readme(hf_dir, len(done))
    print(f"Dataset ready in {hf_dir}/train/")


def _write_readme(hf_dir, total):
    with open(os.path.join(hf_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(f"""\
---
task_categories:
- image-to-text
language:
- bn
tags:
- ocr
- bangla
- conjunct
- yuktabarna
---

# Bengali Conjunct (যুক্তবর্ণ) OCR Dataset

{total:,} synthetic images covering all ~{len(_CONJUNCT_MAP)} Bengali consonant clusters.
Each row includes the `conjunct` field so you can filter/oversample by cluster.

| Split | Samples |
|:------|--------:|
| train | {total:,} |

## Columns
| Column | Type | Description |
|:---|:---|:---|
| `image` | Image | Cropped word image (RGB) |
| `text` | string | Bengali word containing the conjunct |
| `conjunct` | string | Target conjunct cluster (e.g. `ক্ষ`, `ন্ত্র`) |
| `class_name` | string | `conjunct_training` |
| `source` | string | `synthetic_conjunct` |

## Load
```python
from datasets import load_dataset
ds = load_dataset('parquet', data_files={{'train': '{hf_dir}/train/*.parquet'}})
```
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',  default='/app/conjunct_output')
    parser.add_argument('--font_size',   type=int,   default=32)
    parser.add_argument('--blur',        type=float, default=0.3)
    parser.add_argument('--total',       type=int,   default=5000,
                        help='Total images to generate')
    parser.add_argument('--shard_size',  type=int,   default=500)
    parser.add_argument('--push_to_hub', default=None)
    parser.add_argument('--hf_token',   default=None)
    parser.add_argument('--reset',       action='store_true')
    args = parser.parse_args()

    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trdg', 'images')
    fonts     = load_fonts('bn')
    out_img   = os.path.join(args.output_dir, 'images')
    out_lbl   = os.path.join(args.output_dir, 'labels')
    hf_dir    = os.path.join(args.output_dir, 'hf')
    plan_path = os.path.join(args.output_dir, '.plan_conjunct.json')

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    os.makedirs(hf_dir,  exist_ok=True)

    print(f"Conjuncts covered: {len(_CONJUNCT_MAP)}")
    print(f"Target total: {args.total}  (~{math.ceil(args.total/len(_CONJUNCT_MAP))} per cluster)")

    if not args.reset and os.path.exists(plan_path):
        print(f"Resuming from {plan_path}")
        plan = _load_json(plan_path)
    else:
        print("Building plan...")
        plan = build_plan(out_img, out_lbl, args.total)
        _save_json(plan, plan_path)
        print(f"Plan: {len(plan)} items")

    generate_images(plan, fonts, image_dir, args.font_size, args.blur, args.reset)

    done = [p for p in plan if os.path.exists(p['image_path'])]
    print(f"\n{len(done):,}/{len(plan):,} images generated")

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
