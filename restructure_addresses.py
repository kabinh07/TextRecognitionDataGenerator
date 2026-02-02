"""
Restructure addresses to match detection model requirements:
- Remove addresses starting with structural words (ডাকঘর, গ্রাম, রাস্তা)
- Format as 1-5 lines with 7-8 words per line
"""
import json
import os
import random
from transformers import AutoTokenizer

INPUT_FILE = 'data/nid_data_token_balanced.json'
OUTPUT_FILE = 'data/nid_data_token_balanced.json'

def tokenize_bangla(text):
    """Simple word tokenization"""
    return text.strip().split()

def starts_with_structural_word(text):
    """Check if text starts with unwanted structural words"""
    structural_words = ['ডাকঘর', 'গ্রাম', 'রাস্তা', 'গ্রামঃ', 'রাস্তাঃ', 'ডাকঘরঃ']
    tokens = tokenize_bangla(text)
    return tokens and tokens[0] in structural_words

def restructure_address(text, max_words_per_line=8):
    """
    Restructure address into 1-5 lines with ~7-8 words per line.
    Returns formatted address with line breaks.
    """
    # Remove structural prefixes like "গ্রাম/রাস্তা:", "ডাকঘর:"
    text = text.replace('গ্রাম/রাস্তা:', '').replace('ডাকঘর:', '').replace('গ্রামঃ', '').replace('রাস্তাঃ', '').replace('ডাকঘরঃ', '')
    
    # Split by comma first (typical address structure)
    parts = [p.strip() for p in text.split(',') if p.strip()]
    
    if not parts:
        return None
    
    # Rebuild into lines
    lines = []
    current_line = []
    
    for part in parts:
        part_words = tokenize_bangla(part)
        
        # If adding this part exceeds max words, start new line
        if current_line and len(current_line) + len(part_words) > max_words_per_line:
            lines.append(' '.join(current_line))
            current_line = part_words
        else:
            current_line.extend(part_words)
        
        # If current line is at good length, save it
        if len(current_line) >= 5:
            lines.append(' '.join(current_line))
            current_line = []
    
    # Add remaining
    if current_line:
        lines.append(' '.join(current_line))
    
    # Limit to 5 lines max
    lines = lines[:5]
    
    if not lines:
        return None
    
    return '\n'.join(lines)

def main():
    print("Loading XLM-Roberta tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    print(f"Loading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'address' not in data:
        print("No address data found")
        return
    
    original_addresses = data['address']
    print(f"Original address count: {len(original_addresses)}")
    
    restructured = []
    removed_count = 0
    
    for addr in original_addresses:
        if not isinstance(addr, str) or not addr.strip():
            continue
        
        # Skip if starts with structural word
        if starts_with_structural_word(addr):
            removed_count += 1
            continue
        
        # Restructure
        new_addr = restructure_address(addr)
        
        if new_addr:
            # Check token count
            tokens = tokenizer.encode(new_addr, add_special_tokens=True)
            if len(tokens) <= 32:
                restructured.append(new_addr)
            else:
                removed_count += 1
    
    data['address'] = restructured
    
    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\nAddress restructuring complete:")
    print(f"  Original: {len(original_addresses)}")
    print(f"  Kept: {len(restructured)}")
    print(f"  Removed: {removed_count}")
    print(f"\nSample restructured addresses:")
    for i, addr in enumerate(random.sample(restructured, min(3, len(restructured)))):
        print(f"\n{i+1}. {addr}")
    
    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
