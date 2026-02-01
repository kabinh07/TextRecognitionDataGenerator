import json
import collections
import sys
import re
import os
import statistics

def generate_markdown_report(data_source, output_md_path):
    data = {}
    if isinstance(data_source, str):
        print(f"Reading {data_source}...")
        try:
            with open(data_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    elif isinstance(data_source, dict):
        data = data_source
    else:
        print("Invalid data source provided to analytics generator.")
        return

    md_content = []
    md_content.append(f"# Dataset Analytics Report")
    source_name = os.path.basename(data_source) if isinstance(data_source, str) else "Generated Run Data"
    md_content.append(f"**Source File:** `{source_name}`")
    md_content.append(f"**Generated:** `By Antigravity`")
    md_content.append("")

    # 1. Class Distribution
    md_content.append("## 1. Class Distribution")
    md_content.append("| Class Name | Sample Count |")
    md_content.append("| :--- | :--- |")
    
    total_samples = 0
    class_stats = {}
    
    sorted_classes = sorted(data.keys())
    
    for class_name in sorted_classes:
        samples = data[class_name]
        count = len(samples)
        total_samples += count
        class_stats[class_name] = {
            'count': count,
            'tokens': collections.Counter(),
            'lengths': []
        }
        md_content.append(f"| `{class_name}` | {count} |")
        
        # Analyze content
        for text in samples:
            if not isinstance(text, str):
                continue
            
            # Simple tokenization
            tokens = re.findall(r'\w+', text)
            class_stats[class_name]['tokens'].update(tokens)
            class_stats[class_name]['lengths'].append(len(tokens))

    md_content.append("")
    md_content.append(f"**Total Samples:** {total_samples}")
    md_content.append("")

    # 2. Token Analysis
    md_content.append("## 2. Token Analysis per Class")
    
    for class_name in sorted_classes:
        stats = class_stats[class_name]
        md_content.append(f"### Class: `{class_name}`")
        
        # Length stats
        if stats['lengths']:
            avg_len = statistics.mean(stats['lengths'])
            max_len = max(stats['lengths'])
        else:
            avg_len = 0
            max_len = 0
            
        md_content.append(f"- **Avg Tokens/Sample:** {avg_len:.2f}")
        md_content.append(f"- **Max Tokens/Sample:** {max_len}")
        md_content.append(f"- **Unique Tokens:** {len(stats['tokens'])}")
        
        # Top Tokens
        md_content.append("")
        md_content.append("#### Top 10 Frequent Tokens")
        md_content.append("| Token | Count | Frequency (%) |")
        md_content.append("| :--- | :--- | :--- |")
        
        total_tokens_in_class = sum(stats['tokens'].values())
        for token, count in stats['tokens'].most_common(10):
            freq = (count / total_tokens_in_class * 100) if total_tokens_in_class > 0 else 0
            md_content.append(f"| {token} | {count} | {freq:.2f}% |")
        
        md_content.append("")

    # 3. Specific Analysis (Md / মোঃ)
    md_content.append("## 3. High-Frequency Prefix Analysis")
    md_content.append("Checking for 'Md' (English) and 'মোঃ' (Bengali) distribution.")
    md_content.append("")
    
    # Check English Names
    if 'name_en' in data:
        md_content.append("### `name_en`")
        en_samples = data['name_en']
        md_count = sum(1 for t in en_samples if isinstance(t, str) and t.lower().startswith('md'))
        md_content.append(f"- **'Md' prefix count:** {md_count} / {len(en_samples)}")
        md_content.append(f"- **Percentage:** {(md_count/len(en_samples)*100):.2f}%")
        md_content.append("")

    # Check Bangla Names
    if 'name_bn' in data:
        md_content.append("### `name_bn`")
        bn_samples = data['name_bn']
        mo_count = sum(1 for t in bn_samples if isinstance(t, str) and (t.startswith('মোঃ') or t.startswith('মোহাম্মদ')))
        md_content.append(f"- **'মোঃ' prefix count:** {mo_count} / {len(bn_samples)}")
        md_content.append(f"- **Percentage:** {(mo_count/len(bn_samples)*100):.2f}%")
        md_content.append("")

    # Write file
    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    
    print(f"Report generated at: {output_md_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_analytics_report.py <input_json> <output_md>")
    else:
        generate_markdown_report(sys.argv[1], sys.argv[2])
