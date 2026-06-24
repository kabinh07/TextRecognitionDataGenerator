#!/bin/bash
# Full pipeline: generate synthetic data → build extended HF dataset
# Env overrides: FONT_SIZE, BLUR, STROKE_WIDTH, TEXT_COUNT, ADDR_COUNT, CONFUSION_COUNT, HARDNEG_COUNT
set -e

FONT_SIZE="${FONT_SIZE:-18}"
BLUR="${BLUR:-0.3}"
STROKE_WIDTH="${STROKE_WIDTH:-1}"
TEXT_COUNT="${TEXT_COUNT:-10}"
ADDR_COUNT="${ADDR_COUNT:-10}"
CONFUSION_COUNT="${CONFUSION_COUNT:-10}"
HARDNEG_COUNT="${HARDNEG_COUNT:-500}"

echo "=== Pipeline config: font_size=$FONT_SIZE blur=$BLUR stroke=$STROKE_WIDTH text=$TEXT_COUNT addr=$ADDR_COUNT confusion=$CONFUSION_COUNT hardneg=$HARDNEG_COUNT ==="

echo ""
echo "=== Step 1: Generate synthetic NID data ==="
python3 generate_combined.py \
  --text_count "$TEXT_COUNT" \
  --address_count "$ADDR_COUNT" \
  --output_dir /app/output \
  --font_size "$FONT_SIZE" \
  --blur "$BLUR" \
  --stroke_width "$STROKE_WIDTH" \
  --csv_path /app/postal_codes.csv \
  --reset

echo ""
echo "=== Step 1b: Generate hard-negative images (repeated digits, গ/প, ড/দ, low-ink) ==="
python3 generate_hard_negatives.py \
  --output_dir /app/output \
  --font_size "$FONT_SIZE" \
  --blur "$BLUR" \
  --count "$HARDNEG_COUNT" \
  --reset

echo ""
echo "=== Step 2: Build extended HF dataset (p2 + local synth + shamadhan + confusion) ==="
python3 build_confusion_dataset.py \
  --source_dataset kavinh07/ocr_dataset_shamadhan_synth_30k_p2 \
  --confusion_count "$CONFUSION_COUNT" \
  --font_size "$FONT_SIZE" \
  --blur "$BLUR" \
  --stroke_width "$STROKE_WIDTH" \
  --synth_output_dir /app/output \
  --shamadhan_dir /app/data \
  --work_dir /app/confusion_work \
  --output_dir /app/hf_extended \
  --shard_size 500 \
  --reset

echo ""
echo "=== Pipeline complete. Dataset at /app/hf_extended/ ==="
