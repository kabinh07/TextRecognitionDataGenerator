import json
import random
import os

# Complete list of 64 districts of Bangladesh in Bengali
DISTRICTS = [
    "ঢাকা", "ফরিদপুর", "গাজীপুর", "গোপালগঞ্জ", "কিশোরগঞ্জ", "মাদারীপুর", "মানিকগঞ্জ", "মুন্সিগঞ্জ", "নারায়ণগঞ্জ", "নরসিংদী", "রাজবাড়ী", "শরীয়তপুর", "টাঙ্গাইল",
    "বান্দরবান", "ব্রাহ্মণবাড়িয়া", "চাঁদপুর", "চট্টগ্রাম", "কুমিল্লা", "কক্সবাজার", "ফেনী", "খাগড়াছড়ি", "লক্ষ্মীপুর", "নোয়াখালী", "রাঙ্গামাটি",
    "বাগেরহাট", "চুয়াডাঙ্গা", "যশোর", "ঝিনাইদহ", "খুলনা", "কুষ্টিয়া", "মাগুরা", "মেহেরপুর", "নড়াইল", "সাতক্ষীরা",
    "বগুড়া", "চাঁপাইনবাবগঞ্জ", "জয়পুরহাট", "নওগাঁ", "নাটোর", "পাবনা", "রাজশাহী", "সিরাজগঞ্জ",
    "দিনাজপুর", "গাইবান্ধা", "কুড়িগ্রাম", "লালমনিরহাট", "নীলফামারী", "পঞ্চগড়", "রংপুর", "ঠাকুরগাঁও",
    "জামালপুর", "ময়মনসিংহ", "নেত্রকোণা", "শেরপুর",
    "বরগুনা", "বরিশাল", "ভোলা", "ঝালকাঠি", "পটুয়াখালী", "পিরোজপুর",
    "হবিগঞ্জ", "মৌলভীবাজার", "সুনামগঞ্জ", "সিলেট"
]

BLOOD_GROUPS = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']

def balance_dataset(input_file, output_file, target_count=None):
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Determine target count if not provided
    if target_count is None:
        target_count = 0
        for texts in data.values():
            target_count = max(target_count, len(texts))
    
    print(f"Target samples per class: {target_count}")
    
    balanced_data = {}
    
    for class_name, texts in data.items():
        # Make a copy to avoid modifying original info directly yet
        current_texts = list(texts)
        
        # Augmentation for specific classes
        if class_name == 'place_of_birth':
            print(f"Augmenting {class_name} with full district list...")
            original_set = set(current_texts)
            for district in DISTRICTS:
                if district not in original_set:
                    current_texts.append(district)
        
        elif class_name == 'blood_group':
            print(f"Augmenting {class_name} with all blood groups...")
            original_set = set(current_texts)
            for bg in BLOOD_GROUPS:
                if bg not in original_set:
                    current_texts.append(bg)

        # Remove duplicates just in case
        unique_texts = list(set(current_texts))
        
        # Token Capping Logic
        high_freq_texts = []
        other_texts = []
        
        if class_name == 'name_en':
            for text in unique_texts:
                # Check for Md generally at start
                if text.lower().startswith('md') or text.lower().startswith('md.'):
                    high_freq_texts.append(text)
                else:
                    other_texts.append(text)
            print(f"  - 'name_en' split: {len(high_freq_texts)} Md-prefixed, {len(other_texts)} others")
            
        elif class_name == 'name_bn':
            for text in unique_texts:
                if text.startswith('মোঃ') or text.startswith('মোহাম্মদ'):
                    high_freq_texts.append(text)
                else:
                    other_texts.append(text)
            print(f"  - 'name_bn' split: {len(high_freq_texts)} মোঃ-prefixed, {len(other_texts)} others")
            
        else:
            other_texts = unique_texts

        # Assemble final list
        final_texts = []
        
        if high_freq_texts:
            # Cap high freq items to 15% of target_count
            cap_limit = int(target_count * 0.15)
            if len(high_freq_texts) > cap_limit:
                selected_high_freq = random.sample(high_freq_texts, cap_limit)
            else:
                selected_high_freq = list(high_freq_texts) # Take all if less than cap, upsample strictly if needed? No, let upside logic handle.
                # Actually if we want to FORCE them to be low freq, we just take them once.
            
            final_texts.extend(selected_high_freq)
            remaining_slots = target_count - len(final_texts)
        else:
            remaining_slots = target_count

        # Fill remaining slots with other_texts (upsampling if needed)
        if other_texts:
            if len(other_texts) >= remaining_slots:
                final_texts.extend(random.sample(other_texts, remaining_slots))
            else:
                # Need to upsample
                final_texts.extend(other_texts) # Add all once first
                slots_still_needed = target_count - len(final_texts)
                if slots_still_needed > 0:
                    final_texts.extend(random.choices(other_texts, k=slots_still_needed))
        else:
            # Fallback if NO other texts exist (unlikely for names, but possible for small classes)
            # Just fill with high_freq if that's all we have
            if len(final_texts) < target_count:
                 slots_still_needed = target_count - len(final_texts)
                 # We already took some high freq, so we take more from the original high_freq pool
                 if high_freq_texts:
                     final_texts.extend(random.choices(high_freq_texts, k=slots_still_needed))

        random.shuffle(final_texts)
        balanced_data[class_name] = final_texts
        print(f"Class '{class_name}': {len(unique_texts)} unique items -> {len(final_texts)} samples (Capped Md/মোঃ)")

    print(f"Saving balanced dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == "__main__":
    input_path = 'data/nid_data_texts.json'
    output_path = 'data/nid_data_texts_balanced.json'
    
    if os.path.exists(input_path):
        balance_dataset(input_path, output_path)
    else:
        print(f"Input file not found: {input_path}")
