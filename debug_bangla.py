from PIL import Image, ImageDraw, ImageFont
import os

text = "কক্সবাজার" # Cox's Bazar - has conjuncts
fonts = [
    "trdg/fonts/bn/kalpurush.ttf",
    "trdg/fonts/bn/Nikosh.ttf"
]

for font_path in fonts:
    if not os.path.exists(font_path):
        print(f"Font not found: {font_path}")
        continue
    
    try:
        font = ImageFont.truetype(font_path, 60)
    except OSError:
        print(f"Could not load font: {font_path}")
        continue
        
    img = Image.new('RGB', (400, 100), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), text, font=font, fill=(0, 0, 0))
    
    output_name = f"debug_bangla_{os.path.basename(font_path)}.png"
    img.save(output_name)
    print(f"Saved {output_name}")
