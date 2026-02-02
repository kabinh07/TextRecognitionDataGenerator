import json
import os
import random
from collections import Counter
import re

INPUT_FILE = 'data/nid_data_texts.json'
OUTPUT_FILE = 'data/nid_data_token_balanced.json'

def tokenize_bangla(text):
    """Tokenize by spaces"""
    return text.strip().split()

def remove_prefix(name, prefixes=['মোঃ', 'মোঃ', 'মোছাঃ', 'মোছাঃ']):
    """Remove common prefixes from names"""
    tokens = tokenize_bangla(name)
    if tokens and tokens[0] in prefixes:
        return ' '.join(tokens[1:])
    return name

def generate_name_variations(names, target_multiplier=3):
    """
    Generate variations to balance token distribution:
    1. Keep originals
    2. Add versions WITHOUT prefix (মোঃ removed)
    3. Add split parts
    """
    variations = set()
    
    # Keep all originals
    variations.update(names)
    
    # Generate no-prefix versions
    for name in names:
        if not isinstance(name, str): continue
        
        # Remove prefix version
        no_prefix = remove_prefix(name)
        if no_prefix and no_prefix != name:
            variations.add(no_prefix)
        
        # Split into parts (single names and bigrams)
        tokens = tokenize_bangla(name)
        if len(tokens) > 1:
            # Skip prefix if present
            start_idx = 1 if tokens[0] in ['মোঃ', 'মোঃ', 'মোছাঃ', 'মোছাঃ'] else 0
            
            # Add individual tokens (except very short ones)
            for token in tokens[start_idx:]:
                if len(token) >= 2:
                    variations.add(token)
            
            # Add bigrams
            for i in range(start_idx, len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                variations.add(bigram)
    
    return list(variations)

def generate_address_variations(addresses, target_multiplier=2):
    """
    Generate address variations by:
    1. Removing common structural words (গ্রাম, রাস্তা, ডাকঘর)
    2. Splitting by comma
    3. Creating sub-components
    4. Creating "clean" versions with ALL structural words removed
    """
    variations = set()
    
    # Keep originals
    variations.update(addresses)
    
    common_words = ['গ্রাম', 'রাস্তা', 'ডাকঘর', 'সদর', 'পো', 'উপজেলা', 'জেলা', 'থানা', 
                    'বাজার', 'গ্রামঃ', 'রাস্তাঃ', 'চর', 'ইউনিয়ন']
    
    for addr in addresses:
        if not isinstance(addr, str): continue
        
        # Split by comma (typical address structure)
        parts = [p.strip() for p in addr.split(',')]
        
        for part in parts:
            if len(part) >= 3:
                variations.add(part)
                
                # Also add version without common structural words
                tokens = tokenize_bangla(part)
                filtered = [t for t in tokens if t not in common_words]
                if filtered and len(filtered) < len(tokens):
                    clean_part = ' '.join(filtered)
                    if len(clean_part) >= 3:
                        variations.add(clean_part)
        
        # Generate FULLY CLEAN version (remove ALL structural words from entire address)
        all_tokens = tokenize_bangla(addr)
        clean_tokens = [t for t in all_tokens if t not in common_words]
        if clean_tokens and len(clean_tokens) >= 2:
            clean_address = ' '.join(clean_tokens)
            variations.add(clean_address)
        
        # Generate combinations of non-consecutive parts
        if len(parts) >= 3:
            # Take parts that skip middle structural words
            for i in range(len(parts)):
                for j in range(i+2, len(parts)):
                    combo = f"{parts[i]}, {parts[j]}"
                    if len(combo) >= 5:
                        variations.add(combo)
    
    return list(variations)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Original counts:")
    for k, v in data.items():
        print(f"  {k}: {len(v)}")

    balanced_data = {}

    # Balance Bangla names - reduce মোঃ dominance
    for key in ['name_bn', 'father_name', 'mother_name']:
        if key in data:
            print(f"\nGenerating variations for {key}...")
            balanced_data[key] = generate_name_variations(data[key])
    
    # Balance English names
    if 'name_en' in data:
        print(f"\nGenerating variations for name_en...")
        # For English, just split and create variations
        variations = set(data['name_en'])
        for name in data['name_en']:
            if isinstance(name, str):
                parts = name.split()
                for p in parts:
                    if len(p) >= 2:
                        variations.add(p)
                # Bigrams
                for i in range(len(parts) - 1):
                    variations.add(f"{parts[i]} {parts[i+1]}")
        balanced_data['name_en'] = list(variations)

    # Balance Addresses - reduce গ্রাম/রাস্তা/ডাকঘর dominance
    if 'address' in data:
        print(f"\nGenerating variations for address...")
        balanced_data['address'] = generate_address_variations(data['address'])

    # Copy others unchanged
    for key in data:
        if key not in balanced_data:
            balanced_data[key] = data[key]

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("Token-balanced counts:")
    for k, v in balanced_data.items():
        print(f"  {k}: {len(v)}")
    
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
