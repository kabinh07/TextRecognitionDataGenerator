"""
Generate longer text variations (15-20 tokens) for better distribution.
Combines shorter texts or extends existing ones.
"""
import json
import os
import random
from transformers import AutoTokenizer

INPUT_FILE = 'data/nid_data_token_balanced.json'
OUTPUT_FILE = 'data/nid_data_token_balanced.json'
TARGET_TOKEN_RANGE = (15, 20)

def main():
    print("Loading XLM-Roberta tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Only extend addresses - concatenated names look fake
    target_classes = ['address']
    
    for class_name in target_classes:
        if class_name not in data:
            continue
        
        texts = data[class_name]
        print(f"\nProcessing {class_name} ({len(texts)} items)...")
        
        # Count current distribution
        token_counts = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts if isinstance(t, str)]
        in_range = sum(1 for c in token_counts if TARGET_TOKEN_RANGE[0] <= c <= TARGET_TOKEN_RANGE[1])
        print(f"  Current 15-20 token texts: {in_range} ({in_range/len(token_counts)*100:.1f}%)")
        
        # Generate longer versions by combining
        longer_texts = []
        for _ in range(len(texts) // 2):  # Generate 50% more as longer versions
            # Combine 2-3 random texts
            num_combine = random.randint(2, 3)
            combined = ', '.join(random.sample([t for t in texts if isinstance(t, str)], min(num_combine, len(texts))))
            
            tokens = tokenizer.encode(combined, add_special_tokens=True)
            if TARGET_TOKEN_RANGE[0] <= len(tokens) <= TARGET_TOKEN_RANGE[1]:
                longer_texts.append(combined)
            elif len(tokens) < TARGET_TOKEN_RANGE[0]:
                # Too short, try adding more
                extra = random.choice([t for t in texts if isinstance(t, str)])
                combined = f"{combined}, {extra}"
                tokens = tokenizer.encode(combined, add_special_tokens=True)
                if len(tokens) <= 32:  # Don't exceed max
                    longer_texts.append(combined)
        
        # Add to dataset
        data[class_name] = texts + longer_texts
        print(f"  Added {len(longer_texts)} longer texts")
        print(f"  New total: {len(data[class_name])}")
    
    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved to {OUTPUT_FILE}")
    
    # Verify
    print("\n" + "="*60)
    print("Verification:")
    for class_name in target_classes:
        if class_name in data:
            texts = data[class_name]
            token_counts = [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts if isinstance(t, str)]
            in_range = sum(1 for c in token_counts if TARGET_TOKEN_RANGE[0] <= c <= TARGET_TOKEN_RANGE[1])
            print(f"{class_name}: {in_range}/{len(token_counts)} ({in_range/len(token_counts)*100:.1f}%) in 15-20 token range")

if __name__ == "__main__":
    main()
