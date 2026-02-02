import json
import random
import os

INPUT_FILE = 'data/nid_data_texts.json'
OUTPUT_FILE = 'data/nid_data_augmented.json'

def split_generate_names(name_list, min_len=2):
    """
    Splits names into parts and generates variations.
    E.g. "A B C" -> "A", "B", "C", "A B", "B C"
    """
    new_items = set(name_list) # Use set to avoid duplicates
    for name in name_list:
        if not isinstance(name, str): continue
        parts = name.split()
        if len(parts) > 1:
            # Add individual parts
            for p in parts:
                if len(p) >= min_len:
                    new_items.add(p)
            # Add pairs (bigrams)
            for i in range(len(parts) - 1):
                bigram = f"{parts[i]} {parts[i+1]}"
                if len(bigram) >= min_len:
                    new_items.add(bigram)
    return list(new_items)

def split_generate_addresses(addr_list, min_len=3):
    """
    Splits addresses by comma usually found in addresses.
    """
    new_items = set(addr_list)
    for addr in addr_list:
        if not isinstance(addr, str): continue
        # Split by comma
        parts = [p.strip() for p in addr.split(',')]
        for p in parts:
            if len(p) >= min_len:
                new_items.add(p)
        
        # Also split by space if the part is long enough
        # (Maybe too noisy for addresses? Lets stick to comma chunks first as they are semantic units)
        
    return list(new_items)

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Original counts:")
    for k, v in data.items():
        print(f"  {k}: {len(v)}")

    augmented_data = {}

    # Augment Names (BN, EN, Father, Mother)
    for key in ['name_bn', 'name_en', 'father_name', 'mother_name']:
        if key in data:
            augmented_data[key] = split_generate_names(data[key])
    
    # Augment Addresses
    if 'address' in data:
        augmented_data['address'] = split_generate_addresses(data['address'])

    # Copy others
    for key in data:
        if key not in augmented_data:
            augmented_data[key] = data[key]

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)

    print("\nAugmented counts:")
    for k, v in augmented_data.items():
        print(f"  {k}: {len(v)}")
    
    print(f"\nSaved augmented data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
