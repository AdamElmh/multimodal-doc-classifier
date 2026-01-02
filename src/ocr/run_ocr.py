import pandas as pd
import os
from ocr_engine import OCREngine

ocr = OCREngine()
raw_path = "data/raw"
data = []

for category in os.listdir(raw_path):
    cat_folder = os.path.join(raw_path, category)
    if os.path.isdir(cat_folder):
        print(f"Processing {category}...")
        for file in os.listdir(cat_folder):
            text = ocr.extract_text(os.path.join(cat_folder, file))
            data.append({"text": text, "label": category})

df = pd.DataFrame(data)
df.to_csv("data/processed_text.csv", index=False)
print("Extraction complete. saved to data/processed_text.csv")