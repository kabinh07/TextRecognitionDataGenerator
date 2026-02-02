"""
Analyze max token counts for NID data using XLM-Roberta tokenizer.
This helps understand the token length distribution for model training.
"""
import json
import os
from transformers import AutoTokenizer

DATA_FILE = 'data/nid_data_token_balanced.json'

def analyze_token_lengths(data_file):
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print("Loading XLM-Roberta tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    print(f"Loading data from {data_file}...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Categorize by language
    bangla_classes = ['name_bn', 'father_name', 'mother_name', 'address']
    english_classes = ['name_en']
    mixed_classes = ['dob', 'nid_no', 'blood_group', 'place_of_birth', 'issue_date']
    
    results = {
        'Bangla': {'texts': [], 'token_counts': []},
        'English': {'texts': [], 'token_counts': []},
        'Mixed': {'texts': [], 'token_counts': []}
    }
    
    # Collect texts and count tokens
    for class_name, texts in data.items():
        if not isinstance(texts, list):
            continue
            
        category = None
        if class_name in bangla_classes:
            category = 'Bangla'
        elif class_name in english_classes:
            category = 'English'
        elif class_name in mixed_classes:
            category = 'Mixed'
        else:
            continue
        
        for text in texts:
            if isinstance(text, str) and text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=True)
                token_count = len(tokens)
                results[category]['texts'].append(text)
                results[category]['token_counts'].append(token_count)
    
    # Print statistics
    print("\n" + "="*70)
    print("XLM-Roberta Token Count Analysis")
    print("="*70)
    
    for category, data in results.items():
        if not data['token_counts']:
            continue
            
        token_counts = data['token_counts']
        print(f"\n{category}:")
        print(f"  Total samples: {len(token_counts)}")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Average tokens: {sum(token_counts) / len(token_counts):.2f}")
        print(f"  Median tokens: {sorted(token_counts)[len(token_counts)//2]}")
        
        # Show distribution
        percentiles = [50, 75, 90, 95, 99]
        sorted_counts = sorted(token_counts)
        print(f"  Token count percentiles:")
        for p in percentiles:
            idx = int(len(sorted_counts) * p / 100)
            print(f"    {p}th: {sorted_counts[idx]}")
        
        # Show examples of longest texts
        max_idx = token_counts.index(max(token_counts))
        print(f"\n  Longest text ({max(token_counts)} tokens):")
        print(f"    {data['texts'][max_idx][:100]}...")

if __name__ == "__main__":
    analyze_token_lengths(DATA_FILE)
