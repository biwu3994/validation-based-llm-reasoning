import pandas as pd
import json
import ast
import os
import glob

INPUT_DIR = "data/raw/clutrr_clean"
OUTPUT_FILE = "data/processed/clutrr_clean_processed.json"

processed = []

for file_path in glob.glob(os.path.join(INPUT_DIR, "*.csv")):
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        try:
            entity_1, entity_2 = ast.literal_eval(row["query"])
        except Exception:
            entity_1, entity_2 = "", ""

        story = "" if pd.isna(row["story"]) else str(row["story"])
        genders = "" if pd.isna(row["genders"]) else str(row["genders"])
        label = "" if pd.isna(row["target"]) else str(row["target"])

        context = f"""Story:
{story}

Gender information:
{genders}"""

        question = f"What is the relationship of {entity_2} to {entity_1}?"

        processed.append({
            "id": row["id"],
            "context": context,
            "question": question,
            "label": label,
            "source_file": os.path.basename(file_path)
        })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=2, ensure_ascii=False)

print(f"Saved {len(processed)} samples to {OUTPUT_FILE}")
print(processed[0])