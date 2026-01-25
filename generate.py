from trdg.generators import GeneratorFromStrings
import json
import random
import os
from tqdm import tqdm
import pandas as pd
import re
import string
import argparse
import sys

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic text images.")
    parser.add_argument('--language', type=str, choices=['en', 'bn'], default=None, help='Language of the text (en for English, bn for Bangla). Auto-detected if not provided.')
    parser.add_argument('--text_count', type=int, default=100, help='Number of text images to generate')
    parser.add_argument('--orientation', type=int, choices=[0, 1], default=0, help='Orientation of the text (0: horizontal, 1: vertical)')
    parser.add_argument('--font_size', type=int, default=32, help='Font size of the text')
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
        if language == 'en' and not use_list:
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

    print(f"Loaded {len(json_data)} items")

    texts = []
    class_mapping = {}  # Maps text index to class name
    
    # Handle custom JSON format: {class_name: [list_of_texts]}
    if json_file:
        class_list = list(json_data.keys())
        print("class list:", class_list)
        for class_name in class_list:
            class_texts = json_data[class_name]
            if class_name == "name_en":
                class_texts += [text.upper() for text in class_texts if isinstance(text, str)]
            if isinstance(class_texts, list):
                for idx, item in enumerate(class_texts):
                    if isinstance(item, dict):
                        # Extract each field separately to preserve language-specific text
                        for key, value in item.items():
                            if isinstance(value, str) and value.strip():
                                texts.append(value.strip())
                                class_mapping[len(texts) - 1] = class_name
                    elif isinstance(item, str) and item.strip():
                        texts.append(item)
                        class_mapping[len(texts) - 1] = class_name
        print(f"Extracted {len(texts)} field values from {len(class_list)} classes")
    
    else:
        # Original format handling
        if not use_list:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/train_labels.csv'), encoding='utf-8')
            texts = df['words'].to_list()
        
        print(f"Total texts from dataframe: {len(texts)}")
    
    # Auto-detect language if not provided
    if language is None and texts:
        language = detect_language(texts)
        print(f"Auto-detected language: {language}")
    # Only process paragraphs if not using custom JSON format
    if not json_file:
        BATCH_SIZE = 50000
        json_data = json_data[:text_count]
        for i in range(0, len(json_data), BATCH_SIZE):
            batch = json_data[i:i + BATCH_SIZE]
            for item in tqdm(batch, total=len(batch), desc="Processing items"):
                text = ""
                if isinstance(item, dict):
                    text = item['content']
                elif isinstance(item, str):
                    text = item
                else:
                    continue
                if orietation == 0:
                    if use_list:
                        transformed_text = [item]
                    else:
                        transformed_text = split_paragraph_randomly(text, min_words=1, max_words=10, line_break_chance=0.0)
                else:
                    if use_list:
                        transformed_text = [item]
                    else:
                        transformed_text = split_paragraph_randomly(text, min_words=1, max_words=30, line_break_chance=0.2)
                cleaned_text = [item for item in transformed_text if item.strip() != '']
                texts.extend(cleaned_text)
    
    # if language == 'bn':
    #     texts = clean_english_words(texts)
    # if language == 'en':
    #     texts = clean_bangla_words(texts)
    texts = list(set(texts))
    if not use_list and not json_file:
        random.shuffle(texts)

    # Separate texts by language if using custom JSON
    texts_by_language = {'bn': [], 'en': []}
    text_language_map = {}  # Maps text index to its detected language
    
    if json_file:
        # Auto-detect language for each text when using custom JSON
        for idx, text in enumerate(texts):
            detected_lang = detect_text_language(text)
            text_language_map[idx] = detected_lang
            texts_by_language[detected_lang].append(text)
        print(f"Detected {len(texts_by_language['bn'])} Bangla texts and {len(texts_by_language['en'])} English texts")
    else:
        # Use the specified/detected language for all texts
        texts_by_language[language] = texts
        for idx in range(len(texts)):
            text_language_map[idx] = language

    # Generate images for both languages
    for lang, lang_texts in texts_by_language.items():
        if not lang_texts:
            print(f"No {lang} texts to generate")
            continue
        
        print(f"\nGenerating {lang.upper()} images...")
        
        # Use Arial.ttf for English if specified or available
        fonts_to_use = []
        if lang == 'en' and en_font:
            fonts_to_use = [en_font]
            print(f"Using font: {en_font}")
        
        generator = GeneratorFromStrings(
            strings=lang_texts,
            count=min(text_count, len(lang_texts)),
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
        output_dir = '/app/output'
        
        for img, lbl in tqdm(generator, total=min(text_count, len(lang_texts)), desc=f"Generating {lang} images"):
            try:
                # Determine output subdirectory based on separate_folders flag
                if json_file and separate_folders:
                    # Save in separate folders per class
                    text_idx = None
                    for idx, text in enumerate(lang_texts):
                        if text == lbl:
                            text_idx = idx
                            break
                    
                    if text_idx is not None and text_idx in class_mapping:
                        class_name = class_mapping[text_idx]
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
                if json_file and not separate_folders:
                    text_idx = None
                    for idx, text in enumerate(lang_texts):
                        if text == lbl:
                            text_idx = idx
                            break
                    label_with_class = lbl
                    # if text_idx is not None and text_idx in class_mapping:
                    #     class_name = class_mapping[text_idx]
                    #     label_with_class = f"{class_name}|{lbl}"
                    # else:
                    #     label_with_class = f"unknown|{lbl}"
                else:
                    label_with_class = lbl
                
                with open(os.path.join(label_dir, f'{lang}_img_{count}.txt'), 'w', encoding='utf-8') as f:
                    f.write(label_with_class)
            except Exception as e:
                print(f"Error saving image {count}: {e}")
                continue
            count += 1