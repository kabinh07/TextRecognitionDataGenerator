import json
import os
from collections import Counter
import re

DATA_FILES = [
    'data/nid_data_texts.json',
    'data/nid_data_augmented.json',
    'data/nid_data_token_balanced.json'
]

def tokenize_bangla(text):
    """Simple tokenization by spaces and common Bangla punctuation"""
    # Split by spaces and common punctuation, keep the tokens
    tokens = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z]+|\d+', text)
    return tokens

def analyze_token_distribution(data_file):
    if not os.path.exists(data_file):
        print(f"File not found: {data_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {data_file}")
    print(f"{'='*60}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for class_name in ['name_bn', 'address']:
        if class_name not in data:
            continue
            
        print(f"\n{class_name.upper()} Token Analysis:")
        print(f"Total samples: {len(data[class_name])}")
        
        all_tokens = []
        for text in data[class_name]:
            if isinstance(text, str):
                tokens = tokenize_bangla(text)
                all_tokens.extend(tokens)
        
        token_counts = Counter(all_tokens)
        total_tokens = sum(token_counts.values())
        unique_tokens = len(token_counts)
        
        print(f"Total tokens: {total_tokens}")
        print(f"Unique tokens: {unique_tokens}")
        print(f"Diversity ratio: {unique_tokens/total_tokens:.3f}")
        print(f"\nTop 20 most frequent tokens:")
        
        for token, count in token_counts.most_common(20):
            percentage = (count / total_tokens) * 100
            print(f"  {token:20s} : {count:6d} ({percentage:5.2f}%)")

if __name__ == "__main__":
    for data_file in DATA_FILES:
        analyze_token_distribution(data_file)
