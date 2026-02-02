import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

text = "কক্সবাজার" 
font_path = "trdg/fonts/bn/kalpurush.ttf" # Matplotlib works well with ttf

if not os.path.exists(font_path):
    print(f"Font not found: {font_path}")
    exit(1)

prop = FontProperties(fname=font_path, size=60)

fig, ax = plt.subplots(figsize=(5, 2))
ax.text(0.5, 0.5, text, fontproperties=prop, ha='center', va='center')
ax.axis('off')

output_name = "debug_bangla_mpl.png"
plt.savefig(output_name, dpi=100)
print(f"Saved {output_name}")
