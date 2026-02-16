from trdg.generators import GeneratorFromStrings
import json
import collections
import random
import os
import math
from tqdm import tqdm
import pandas as pd
import re
import string
import argparse
import sys
from create_analytics_report import generate_markdown_report

def clean_english_words(texts):
    new_texts = []
    for text in tqdm(texts, total=len(texts), desc="Cleaning english texts"):
        text = re.sub(r'[a-zA-Z0-9]+', '', text)
        if text.strip() != '' and not re.fullmatch(rf"[{re.escape(string.punctuation)}]+", text.strip()):
            new_texts.append(text)
            new_texts.append(text)
    return new_texts

def clean_bangla_words(texts):
    new_texts = []
    for text in tqdm(texts, total=len(texts), desc="Cleaning english texts"):
        text = re.sub(r'[\u0980-\u09FF]+', '', text)
        if text.strip() != '' and not re.fullmatch(rf"[{re.escape(string.punctuation)}]+", text.strip()):
            new_texts.append(text)
            new_texts.append(text)
    return new_texts

def detect_text_language(text):
    """
    Detect language of a single text.
    Returns 'bn' for Bangla or 'en' for English based on character analysis.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: 'bn' for Bangla, 'en' for English
    """
    if not text or not isinstance(text, str):
        return 'en'
    # Check for Bengali characters (Unicode range \u0980-\u09FF)
    bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
    # Check for English/ASCII letters
    english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
    
    return 'bn' if bengali_chars > english_chars else 'en'

def detect_language(texts):
    """
    Auto-detect language from text content.
    Returns 'bn' for Bangla or 'en' for English based on character analysis.
    
    Args:
        texts (list): List of text strings to analyze
    
    Returns:
        str: 'bn' for Bangla, 'en' for English
    """
    bengali_count = 0
    english_count = 0
    
    # Sample up to 10 texts for language detection
    sample_texts = texts[:min(10, len(texts))]
    
    for text in sample_texts:
        if not text or not isinstance(text, str):
            continue
        # Check for Bengali characters (Unicode range \u0980-\u09FF)
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        # Check for English/ASCII letters
        english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        
        if bengali_chars > english_chars:
            bengali_count += 1
        else:
            english_count += 1
    
    return 'bn' if bengali_count > english_count else 'en'

def split_paragraph_randomly(paragraph, min_words=1, max_words=10, line_break_chance=0.2):
    """
    Splits a paragraph into lines with random lengths (between min_words and max_words).
    Randomly inserts paragraph breaks as well.

    Args:
        paragraph (str): The input paragraph.
        min_words (int): Minimum words per line.
        max_words (int): Maximum words per line.
        line_break_chance (float): Probability of inserting an extra line break after a line.

    Returns:
        str: The transformed paragraph with random line splits and breaks.
    """
    words = paragraph.split()
    lines = []
    i = 0

    while i < len(words):
        line_len = random.randint(min_words, max_words)
        line_words = words[i:i + line_len]
        i += line_len
        line = ' '.join(line_words)
        if line.strip() == '':
            continue
        lines.append(line)

        # Randomly add extra line break (simulate paragraph)
        if random.random() < line_break_chance:
            lines.append("\n")  # An empty string becomes a paragraph break on join

    return lines

def balance_tokens(texts, target_count):
    """
    Balances token distribution by sampling texts based on inverse token frequency.
    
    Args:
        texts (list): List of text strings.
        target_count (int): Number of texts to sample.
        
    Returns:
        list: Sampled list of texts.
    """
    if not texts:
        return []
    
    # Ensure items are unique before balancing
    texts = list(set([t for t in texts if isinstance(t, str) and t.strip()]))
    
    if len(texts) <= target_count:
        return texts

    # Tokenize and count frequencies
    token_counts = collections.Counter()
    for text in tqdm(texts, desc="Calculating token frequencies"):
        if not isinstance(text, str):
            continue
        # Split by non-alphanumeric/non-Bangla characters
        tokens = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z0-9]+', text)
        token_counts.update(tokens)
    
    # Calculate weights for each text
    # Penalty for frequent tokens: 1 / sum(sqrt(freq(token)))
    weights = []
    for text in tqdm(texts, desc="Calculating sampling weights"):
        if not isinstance(text, str):
            weights.append(0)
            continue
            
        tokens = re.findall(r'[\u0980-\u09FF]+|[a-zA-Z0-9]+', text)
        if not tokens:
            weights.append(0)
            continue
        
        # Score is sum of sqrt(counts) for each token
        score = sum(math.sqrt(token_counts[t]) for t in tokens)
        weights.append(1.0 / score if score > 0 else 1.0)
    
    # Weighted sampling WITHOUT replacement using Efraimidis and Spirakis algorithm
    # Each item gets a score: random.random() ^ (1/weight)
    # The higher the weight, the more likely it is to have a high score.
    print(f"Sampling {target_count} items without replacement...")
    weighted_scores = []
    for i in range(len(texts)):
        if weights[i] > 0:
            # Score = r^(1/w). We can use log transformed version for stability: log(r)/w
            score = math.log(random.random()) / weights[i]
            weighted_scores.append((score, texts[i]))
    
    # Highest scores win
    weighted_scores.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in weighted_scores[:target_count]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic text images.")
    parser.add_argument('--language', type=str, choices=['en', 'bn'], default=None, help='Language of the text (en for English, bn for Bangla). Auto-detected if not provided.')
    parser.add_argument('--text_count', type=int, default=100, help='Number of text images to generate')
    parser.add_argument('--orientation', type=int, choices=[0, 1], default=0, help='Orientation of the text (0: horizontal, 1: vertical)')
    parser.add_argument('--font_size', type=int, default=16, help='Font size of the text')
    parser.add_argument('--blur', type=float, default=0.8, help='Blur level of the text images')
    parser.add_argument('--use_list', action='store_true', help='Use list data for text generation')
    parser.add_argument('--json_file', type=str, default=None, help='Path to JSON file with format {class_name: [list_of_texts]}')
    parser.add_argument('--separate_folders', action='store_true', help='Save each class in separate folders instead of together')
    parser.add_argument('--en_font', type=str, default=None, help='Path to Arial.ttf or any font file to use ONLY for English text generation')
    args = parser.parse_args()

    language = args.language
    text_count = args.text_count
    orietation = args.orientation
    font_size = args.font_size
    blur = args.blur
    use_list = args.use_list
    json_file = args.json_file
    separate_folders = args.separate_folders
    en_font = args.en_font

    print("\n\nParameters:\nOrientation:", orietation, "\nFont Size:", font_size, "\nBlur:", blur, "\nUse_list:", use_list, "\nJSON File:", json_file, "\nSeparate Folders:", separate_folders, "\nEnglish Font:", en_font, "\n\n")

    # Validate English font path if provided
    if en_font and not os.path.exists(en_font):
        print(f"ERROR: English font file not found: {en_font}")
        sys.exit(1)

    # Loading JSON corpuses from files
    if json_file:
        # Load from user-provided JSON file with format {class_name: [list_of_texts]}
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print(f"Loaded custom JSON file with {len(json_data)} classes")
    else:
        language_provided = args.language is not None
        
        # Default to token-balanced NID data if available
        nid_data_token_balanced = os.path.join(os.path.dirname(__file__), 'data/nid_data_token_balanced.json')
        nid_data_path = os.path.join(os.path.dirname(__file__), 'data/nid_data_texts.json')
        
        if os.path.exists(nid_data_token_balanced):
             print(f"Loading token-balanced NID data from {nid_data_token_balanced}")
             with open(nid_data_token_balanced, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        elif os.path.exists(nid_data_path):
             print(f"Loading NID data from {nid_data_path}")
             with open(nid_data_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        elif language == 'en' and not use_list:
            with open(os.path.join(os.path.dirname(__file__), ('data/english_news.json')), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        elif language == 'bn' and not use_list:
            with open(os.path.join(os.path.dirname(__file__), ('data/data_V2.json')), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        elif language == 'bn' and use_list:
            with open(os.path.join(os.path.dirname(__file__), ('list_data/bangla_list.json')), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        elif language == 'en' and use_list:
            with open(os.path.join(os.path.dirname(__file__), ('list_data/english_list.json')), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        elif language is None:
            # If language not provided, try to load any available data first for detection
            try:
                with open(os.path.join(os.path.dirname(__file__), ('data/data_V2.json')), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except:
                try:
                    with open(os.path.join(os.path.dirname(__file__), ('data/english_news.json')), 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                except:
                    print("Could not load default JSON files. Please specify --language explicitly.")
                    sys.exit(1)
        else:
            print("Invalid language choice. Please choose 'en' or 'bn'.")
            sys.exit(1)

    # Loading JSON corpuses from files and preparing texts_by_language
    texts_by_language = {'bn': {}, 'en': {}}

    if use_list:
        print("Using list data from list_data directory...")
        # Bangla
        bn_path = os.path.join(os.path.dirname(__file__), 'list_data/bangla_list.json')
        if os.path.exists(bn_path) and (language in ['bn', None]):
             print(f"Loading and balancing Bangla list from {bn_path}...")
             with open(bn_path, 'r', encoding='utf-8') as f:
                 bn_data = json.load(f)
             bn_balanced = balance_tokens(bn_data, text_count)
             texts_by_language['bn']['list_data'] = bn_balanced
             
        # English
        en_path = os.path.join(os.path.dirname(__file__), 'list_data/english_list.json')
        if os.path.exists(en_path) and (language in ['en', None]):
             print(f"Loading and balancing English list from {en_path}...")
             with open(en_path, 'r', encoding='utf-8') as f:
                 en_data = json.load(f)
             en_balanced = balance_tokens(en_data, text_count)
             texts_by_language['en']['list_data'] = en_balanced
        
        if not texts_by_language['bn'] and not texts_by_language['en']:
            print("No list data found in list_data/ directory.")
            sys.exit(1)
            
        # Define json_data for consistency with following logic
        json_data = texts_by_language
            
    elif json_file:
        # Load from user-provided JSON file with format {class_name: [list_of_texts]}
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        print(f"Loaded custom JSON file with {len(json_data)} classes")
        
        # Group by language and class
        for class_name, class_texts in json_data.items():
            if class_name == "name_en":
                class_texts += [text.upper() for text in class_texts if isinstance(text, str)]
            
            for text in class_texts:
                val = ""
                if isinstance(text, dict):
                    # Use the first string value found in the dict
                    for v in text.values():
                        if isinstance(v, str) and v.strip():
                            val = v.strip()
                            break
                elif isinstance(text, str) and text.strip():
                    val = text.strip()
                
                if val:
                    detected_lang = detect_text_language(val)
                    if class_name not in texts_by_language[detected_lang]:
                        texts_by_language[detected_lang][class_name] = []
                    texts_by_language[detected_lang][class_name].append(val)
    else:
        # Default behavior: NID or news data
        nid_data_token_balanced = os.path.join(os.path.dirname(__file__), 'data/nid_data_token_balanced.json')
        nid_data_path = os.path.join(os.path.dirname(__file__), 'data/nid_data_texts.json')
        
        if os.path.exists(nid_data_token_balanced):
            print(f"Loading token-balanced NID data from {nid_data_token_balanced}")
            with open(nid_data_token_balanced, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        elif os.path.exists(nid_data_path):
            print(f"Loading NID data from {nid_data_path}")
            with open(nid_data_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        else:
            # Fallback to language-specific news
            if language == 'en':
                with open(os.path.join(os.path.dirname(__file__), 'data/english_news.json'), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            elif language == 'bn':
                with open(os.path.join(os.path.dirname(__file__), 'data/data_V2.json'), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            else:
                # Try to load any available
                try:
                    with open(os.path.join(os.path.dirname(__file__), 'data/data_V2.json'), 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                except:
                    print("Could not load default JSON files. Please specify --language or --use_list.")
                    sys.exit(1)
        
        # Process json_data into texts_by_language
        if isinstance(json_data, dict):
            for class_name, texts in json_data.items():
                for t in texts:
                    if isinstance(t, str) and t.strip():
                        det_lang = detect_text_language(t)
                        if class_name not in texts_by_language[det_lang]:
                            texts_by_language[det_lang][class_name] = []
                        texts_by_language[det_lang][class_name].append(t)
        else:
            # List format
            for t in json_data:
                if isinstance(t, str) and t.strip():
                    det_lang = detect_text_language(t)
                    if 'unknown' not in texts_by_language[det_lang]:
                        texts_by_language[det_lang]['unknown'] = []
                    texts_by_language[det_lang]['unknown'].append(t)

    # Print distribution info
    for lang in ['bn', 'en']:
        if texts_by_language[lang]:
            print(f"\n{lang.upper()} distribution:")
            for class_name, texts in texts_by_language[lang].items():
                print(f"  {class_name}: {len(texts)} items")

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate images for both languages with CLASS-BALANCED sampling
    for lang, class_dict in texts_by_language.items():
        if not class_dict:
            print(f"No {lang} texts to generate")
            continue
        
        print(f"\nGenerating {lang.upper()} images with class-balanced sampling...")
        
        # Prepare class-balanced sampling
        class_names = list(class_dict.keys())
        if not class_names:
            continue
        
        print(f"Classes: {class_names}")
        
        # Use Arial.ttf for English if specified or available
        fonts_to_use = []
        if lang == 'en' and en_font:
            fonts_to_use = [en_font]
            print(f"Using font: {en_font}")
        
        # Generate texts by sampling from classes
        balanced_texts = []
        if use_list and 'list_data' in class_dict:
            # For use_list, we already have perfectly balanced and unique texts in 'list_data'
            balanced_texts = [(t, 'list_data') for t in class_dict['list_data']]
            # In case list_data has more or fewer than text_count (though it shouldn't), 
            # we should cap it or just use it.
            balanced_texts = balanced_texts[:text_count]
        else:
            # Sampling from classes uniformly
            for i in range(text_count):
                # Pick random class
                class_name = random.choice(class_names)
                # Pick random text from that class
                if class_dict[class_name]:
                    text = random.choice(class_dict[class_name])
                    balanced_texts.append((text, class_name))
        
        # Extract just strings for the generator
        lang_texts = [x[0] for x in balanced_texts]
        
        generator = GeneratorFromStrings(
            strings=lang_texts,
            count=text_count,
            fonts=fonts_to_use,
            size=font_size,
            language=lang,
            skewing_angle=3,
            random_skew=True,
            distorsion_type=0,
            blur=blur,
            random_blur=True,
            fit=True,
            word_split=True,
            background_type=3,
            orientation=orietation
        )
        count = 0
        output_dir = 'output'

        
        for img, lbl in tqdm(generator, total=text_count, desc=f"Generating {lang} images"):
            try:
                # Get class_name from balanced_texts
                class_name = balanced_texts[count][1] if count < len(balanced_texts) else 'unknown'


                # Determine output subdirectory based on separate_folders flag
                if (json_file or isinstance(json_data, dict)) and separate_folders:
                    if class_name:
                        class_output_dir = os.path.join(output_dir, class_name)
                    else:
                        class_output_dir = os.path.join(output_dir, 'unknown')
                else:
                    # Save all together in one directory
                    class_output_dir = output_dir
                
                image_dir = os.path.join(class_output_dir, 'images')
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                label_dir = os.path.join(class_output_dir, 'labels')
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir)
                
                img.save(os.path.join(image_dir, f'{lang}_img_{count}.png'))
                
                # Save label with class tag if using custom JSON and not separate folders
                if (json_file or isinstance(json_data, dict)) and not separate_folders:
                    label_with_class = lbl # Or f"{class_name}|{lbl}" if requested
                else:
                    label_with_class = lbl

                
                with open(os.path.join(label_dir, f'{lang}_img_{count}.txt'), 'w', encoding='utf-8') as f:
                    f.write(label_with_class)
            except Exception as e:
                print(f"Error saving image {count}: {e}")
                continue
            count += 1
    
    # Generate Analytics Report for the current run
    print("\nGenerating analytics report for this run...")
    analytics_data = collections.defaultdict(list)
    
    for lang, class_dict in texts_by_language.items():
        if not class_dict:
            continue
        
        # Sample from class_dict for analytics
        for _ in range(min(text_count, 100)):
            class_name = random.choice(list(class_dict.keys()))
            if class_dict[class_name]:
                text = random.choice(class_dict[class_name])
                analytics_data[class_name].append(text)
            
    report_path = os.path.join(output_dir, 'run_analytics.md')
    generate_markdown_report(dict(analytics_data), report_path)