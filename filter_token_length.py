"""
Filter NID data to keep only texts with <= 32 XLM-Roberta tokens.
"""
import json
import os
from transformers import AutoTokenizer

INPUT_FILE = 'data/nid_data_token_balanced.json'
OUTPUT_FILE = 'data/nid_data_token_balanced.json'  # Overwrite
MAX_TOKENS = 32

def filter_by_token_length(data_file, max_tokens=32):
    print("Loading XLM-Roberta tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    print(f"Loading data from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered_data = {}
    stats = {}
    
    for class_name, texts in data.items():
        if not isinstance(texts, list):
            filtered_data[class_name] = texts
            continue
        
        original_count = len(texts)
        filtered_texts = []
        
        for text in texts:
            if isinstance(text, str) and text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=True)
                token_count = len(tokens)
                
                if token_count <= max_tokens:
                    filtered_texts.append(text)
        
        filtered_data[class_name] = filtered_texts
        stats[class_name] = {
            'original': original_count,
            'filtered': len(filtered_texts),
            'removed': original_count - len(filtered_texts)
        }
    
    # Save filtered data
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print("\n" + "="*60)
    print(f"Filtered to max {max_tokens} tokens")
    print("="*60)
    for class_name, stat in stats.items():
        removed_pct = (stat['removed'] / stat['original'] * 100) if stat['original'] > 0 else 0
        print(f"{class_name}:")
        print(f"  Original: {stat['original']}")
        print(f"  Kept: {stat['filtered']}")
        print(f"  Removed: {stat['removed']} ({removed_pct:.1f}%)")
    
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    filter_by_token_length(INPUT_FILE, MAX_TOKENS)
