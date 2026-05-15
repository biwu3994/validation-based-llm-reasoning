import json

INPUT_FILE = "data/raw/neurl/deductive_neutral.json"
OUTPUT_FILE = "data/processed/neulr_deductive_processed.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

processed = []

for sample in data:
    processed.append({
        "id": sample.get("id", ""),
        "context": sample.get("context", ""),
        "question": sample.get("question", ""),
        "label": sample.get("label", "")
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=2, ensure_ascii=False)

print(f"Saved {len(processed)} samples to {OUTPUT_FILE}")
print(processed[0])