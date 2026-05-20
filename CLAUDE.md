# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Install
```bash
pip install -r requirements.txt
```

### Run tests
```bash
python tests.py
# Single test:
python -m unittest tests.DataGenerator.test_generate_data_with_format
```

### Generate images (main custom script)
```bash
# Auto-detect language from NID data (default)
python generate.py --text_count 1000

# Force language, custom font size
python generate.py --language bn --text_count 500 --font_size 32

# Custom JSON corpus {class_name: [texts]}
python generate.py --json_file data/my_data.json --separate_folders

# List-mode (from list_data/ flat lists)
python generate.py --use_list --language bn --text_count 2000

# With English-specific font
python generate.py --language en --en_font /path/to/Arial.ttf
```

### CLI (upstream trdg tool)
```bash
trdg -c 1000 -w 5 -f 64 -l en
# Or directly:
python trdg/run.py -c 100 -l bn -f 32
```

### Prepare list data
```bash
# Builds list_data/bangla_list.json and list_data/english_list.json from raw corpora
python generate_list_data.py
```

## Architecture

### Package: `trdg/`
The installable `trdg` Python package. Entry point for CLI is `trdg/run.py`.

**Core pipeline** (called for every image):
1. `trdg/computer_text_generator.py` — renders text string to RGBA PIL image + RGB mask using Pillow `ImageDraw`. Handles horizontal and vertical orientations. RTL languages (Arabic, Kurdish) are reshaped before reaching here via `arabic_reshaper` + `python-bidi` in the generators.
2. `trdg/distorsion_generator.py` — applies sine/cosine/random distortion to image+mask pair.
3. `trdg/background_generator.py` — produces background: Gaussian noise (0), plain white (1), quasicrystal (2), or random image from `trdg/images/` (3).
4. `trdg/data_generator.py` (`FakeTextDataGenerator.generate`) — orchestrates the above, applies skew + blur, composites text onto background, optionally saves mask and bounding box files.

**Generators** (`trdg/generators/`):
Iterator classes that produce `(PIL.Image, label_str)` tuples. Four variants: `GeneratorFromStrings`, `GeneratorFromDict`, `GeneratorFromRandom`, `GeneratorFromWikipedia`. All accept the same parameters as the CLI. Use `GeneratorFromStrings` when you have a prepared text list.

**Fonts**: `trdg/fonts/<lang>/` — only `.ttf`. Language code maps to folder in `trdg/utils.py:load_fonts()`.

**Dicts**: `trdg/dicts/<lang>.txt` — word lists for dictionary-mode generation.

### Custom scripts (root level)
These extend the base `trdg` package for Bangla/English NID OCR training data:

**`generate.py`** — main production generation script. Key behaviors:
- Data priority order: `data/nid_data_token_balanced.json` > `data/nid_data_texts.json` > language-specific fallbacks (`data/english_news.json`, `data/data_V2.json`). Custom JSON via `--json_file`.
- JSON format: `{class_name: [text_strings]}` — auto-detects language per string (Bangla: U+0980-U+09FF vs ASCII).
- Class-balanced sampling: picks class uniformly at random, then random text from that class.
- Token balancing (Efraimidis-Spirakis): weights texts by `1/sum(sqrt(freq(token)))` to undersample frequent tokens.
- Output structure: `output/{class_name}/images/` and `output/{class_name}/labels/` (or flat `output/images/` without `--separate_folders`).
- Generates `output/run_analytics.md` after each run via `create_analytics_report.py`.

**`generate_list_data.py`** — builds flat list corpora. Reads `data/data_v2.json` (Bangla news) and `data/train_labels.csv`, chunks long texts at <=30 chars on natural boundaries (space/comma/daari), deduplicates, shuffles, writes `list_data/bangla_list.json` and `list_data/english_list.json`.

**Data files** (not tracked in git, must be placed manually):
- `data/nid_data_token_balanced.json` — primary corpus
- `data/nid_data_texts.json` — fallback NID corpus
- `data/data_V2.json` / `data/data_v2.json` — Bangla news corpus
- `data/english_news.json` — English news corpus
- `data/train_labels.csv` — CSV with `words` column
- `list_data/bangla_list.json`, `list_data/english_list.json` — prebuilt flat lists

### Analyzer scripts (root level)
`analyze_*.py`, `balance_*.py`, `filter_token_length.py` — standalone dataset analysis and rebalancing utilities. Not part of the generation pipeline; run independently to inspect or modify JSON corpora.
