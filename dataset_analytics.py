"""
Analytics for the generated OCR dataset.
Scans output/ directory, computes label + Qwen-tokenizer statistics,
writes a markdown report.

Standalone:
    python dataset_analytics.py
    python dataset_analytics.py --output_dir output --report dataset_analytics.md
    python dataset_analytics.py --tokenizer Qwen/Qwen2.5-0.5B --top_tokens 30

Called from generate.py after each run automatically.
"""

import argparse
import collections
import math
import os
import statistics
from pathlib import Path

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Collect samples from output/
# ---------------------------------------------------------------------------

def collect_samples(output_dir: str):
    """
    Returns list of dicts: {image, label, class, lang}
    Supports flat and class-separated layouts from generate.py.
    """
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
        class_name = parts[-3] if len(parts) >= 3 and parts[-2] == 'images' else 'flat'
        stem = img_path.stem
        lang = stem.split('_')[0] if stem.split('_')[0] in ('bn', 'en') else 'unknown'

        samples.append({'image': str(img_path), 'label': label,
                        'class': class_name, 'lang': lang})
    return samples


# ---------------------------------------------------------------------------
# Token statistics helpers
# ---------------------------------------------------------------------------

def shannon_entropy(counter: collections.Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def coefficient_of_variation(counter: collections.Counter) -> float:
    vals = list(counter.values())
    if len(vals) < 2 or statistics.mean(vals) == 0:
        return 0.0
    return statistics.stdev(vals) / statistics.mean(vals)


def balance_score_str(counter: collections.Counter) -> str:
    entropy = shannon_entropy(counter)
    cv = coefficient_of_variation(counter)
    max_e = math.log2(len(counter)) if len(counter) > 1 else 1
    norm_e = entropy / max_e if max_e > 0 else 0
    score = norm_e * (1 / (1 + cv))
    return f"{score:.4f}  (entropy={entropy:.2f} bits, CV={cv:.3f})"


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def md_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join([':---'] * len(headers)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return lines


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------

def build_report(samples, tokenizer, top_n: int) -> list:
    md = []
    total = len(samples)

    md.append('# Dataset Analytics')
    md.append(f'**Total images:** {total:,}')
    md.append('')

    if total == 0:
        md.append('No samples found.')
        return md

    # Language distribution
    md.append('## Language Distribution')
    lang_counts = collections.Counter(s['lang'] for s in samples)
    md += md_table(
        ['Language', 'Count', '%'],
        [(lang, f'{cnt:,}', f'{100*cnt/total:.1f}%')
         for lang, cnt in sorted(lang_counts.items())]
    )
    md.append('')

    # Class distribution
    md.append('## Class Distribution')
    class_counts = collections.Counter(s['class'] for s in samples)
    md += md_table(
        ['Class', 'Count', '%'],
        [(cls, f'{cnt:,}', f'{100*cnt/total:.1f}%')
         for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])]
    )
    md.append('')

    # Multi-line
    double = sum(1 for s in samples if '\n' in s['label'])
    md.append('## Multi-line Images')
    md.append(f'- Double-line: **{double:,}** ({100*double/total:.1f}%)')
    md.append(f'- Single-line: **{total-double:,}** ({100*(total-double)/total:.1f}%)')
    md.append('')

    # Label text stats
    md.append('## Label Text Statistics')
    for lang in sorted(lang_counts):
        s_lang = [s for s in samples if s['lang'] == lang]
        char_lens = [len(s['label']) for s in s_lang]
        word_lens = [len(s['label'].split()) for s in s_lang]
        md.append(f'### {lang.upper()}')
        md += md_table(
            ['Metric', 'Avg', 'Min', 'Max', 'Median'],
            [
                ['Char length', f'{statistics.mean(char_lens):.1f}',
                 min(char_lens), max(char_lens), f'{statistics.median(char_lens):.0f}'],
                ['Word count', f'{statistics.mean(word_lens):.1f}',
                 min(word_lens), max(word_lens), f'{statistics.median(word_lens):.0f}'],
            ]
        )
        md.append('')

    # Qwen token stats
    if tokenizer is not None:
        md.append(f'## Qwen Tokenizer Statistics')
        md.append(f'*Model: `{tokenizer.name_or_path}`*')
        md.append('')

        all_ids: collections.Counter = collections.Counter()

        for lang in sorted(lang_counts):
            s_lang = [s for s in samples if s['lang'] == lang]
            lang_ids: collections.Counter = collections.Counter()
            tok_counts = []

            for s in tqdm(s_lang, desc=f'Tokenizing {lang}'):
                ids = tokenizer.encode(s['label'], add_special_tokens=False)
                tok_counts.append(len(ids))
                lang_ids.update(ids)
                all_ids.update(ids)

            md.append(f'### {lang.upper()} — tokens per label')
            md += md_table(
                ['Metric', 'Value'],
                [
                    ['Avg tokens/label', f'{statistics.mean(tok_counts):.2f}'],
                    ['Min', min(tok_counts)],
                    ['Max', max(tok_counts)],
                    ['Median', f'{statistics.median(tok_counts):.0f}'],
                    ['Unique token IDs', f'{len(lang_ids):,}'],
                    ['Total token occurrences', f'{sum(tok_counts):,}'],
                    ['Balance score', balance_score_str(lang_ids)],
                ]
            )
            md.append('')

            md.append(f'#### Top {top_n} Tokens')
            total_toks = sum(lang_ids.values())
            rows = []
            for tid, cnt in lang_ids.most_common(top_n):
                tok_str = tokenizer.decode([tid])
                rows.append([repr(tok_str), tid, f'{cnt:,}', f'{100*cnt/total_toks:.2f}%'])
            md += md_table(['Token', 'ID', 'Count', '%'], rows)
            md.append('')

        md.append('## Overall Token Balance')
        md.append(f'- Unique token IDs: **{len(all_ids):,}**')
        md.append(f'- Total occurrences: **{sum(all_ids.values()):,}**')
        md.append(f'- Balance score: **{balance_score_str(all_ids)}**')
        md.append('')
        md.append('> Score: 0 = fully skewed, 1 = perfectly uniform.')
        md.append('> Formula: normalised entropy × 1/(1+CV).')
        md.append('')

    return md


# ---------------------------------------------------------------------------
# Public entry point (called from generate.py)
# ---------------------------------------------------------------------------

def run_analytics(output_dir: str, report_path: str,
                  tokenizer_id: str = 'Qwen/Qwen2.5-0.5B',
                  top_n: int = 20):
    """Scan output_dir, build report, write to report_path."""
    print('\nCollecting generated samples for analytics...')
    samples = collect_samples(output_dir)
    print(f'Found {len(samples):,} samples')

    tokenizer = None
    if tokenizer_id:
        try:
            from transformers import AutoTokenizer
            print(f'Loading tokenizer: {tokenizer_id}')
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id, trust_remote_code=True)
        except Exception as e:
            print(f'Tokenizer unavailable ({e}), skipping token stats.')

    md_lines = build_report(samples, tokenizer, top_n)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'Analytics saved → {report_path}')


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='output')
    parser.add_argument('--report', default='dataset_analytics.md')
    parser.add_argument('--tokenizer', default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--top_tokens', type=int, default=20)
    args = parser.parse_args()
    run_analytics(args.output_dir, args.report, args.tokenizer, args.top_tokens)


if __name__ == '__main__':
    main()
