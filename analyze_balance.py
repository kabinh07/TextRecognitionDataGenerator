import json
import collections
import sys

def analyze_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"Total classes: {len(data)}")
    
    class_counts = {}
    char_counts = collections.Counter()
    
    for class_name, texts in data.items():
        class_counts[class_name] = len(texts)
        for text in texts:
            if isinstance(text, str):
                char_counts.update(text)
            elif isinstance(text, dict):
                 # Handle dictionary items if any (based on generate.py logic)
                 for val in text.values():
                     if isinstance(val, str):
                         char_counts.update(val)

    print("\n--- Class Distribution (Samples per Class) ---")
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_classes:
        print(f"{cls}: {count}")

    print("\n--- Character Distribution (Top 20) ---")
    for char, count in char_counts.most_common(20):
        print(f"{repr(char)}: {count}")

    print("\n--- Least Common Characters (Bottom 20) ---")
    for char, count in char_counts.most_common()[:-21:-1]:
        print(f"{repr(char)}: {count}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_json(sys.argv[1])
    else:
        print("Usage: python analyze_balance.py <json_file>")
