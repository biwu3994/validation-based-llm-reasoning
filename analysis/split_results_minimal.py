import json

INPUT_FILE = "results/results_clutrr_mixed_framework.json"

OUTPUT_CORRECT = "results/correct_min_clutrr_mixed_framework.json"
OUTPUT_ERROR = "results/error_min_clutrr_mixed_framework.json"
OUTPUT_REFUSAL = "results/refusal_min_clutrr_mixed_framework.json"

def extract_minimal(item):
    return {
        "id": item.get("id"),
        "gold": item.get("gold"),
        "prediction": item.get("prediction"),
        "raw_prediction": item.get("raw_prediction")
    }


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Compatible with summary or list
if isinstance(data, dict):
    results = data.get("results", [])
else:
    results = data


correct_list = []
error_list = []
refusal_list = []


for item in results:
    status = item.get("status", "")

    minimal = extract_minimal(item)

    if status == "correct":
        correct_list.append(minimal)
    elif status == "error":
        error_list.append(minimal)
    elif status == "refusal":
        refusal_list.append(minimal)


with open(OUTPUT_CORRECT, "w", encoding="utf-8") as f:
    json.dump(correct_list, f, indent=2, ensure_ascii=False)

with open(OUTPUT_ERROR, "w", encoding="utf-8") as f:
    json.dump(error_list, f, indent=2, ensure_ascii=False)

with open(OUTPUT_REFUSAL, "w", encoding="utf-8") as f:
    json.dump(refusal_list, f, indent=2, ensure_ascii=False)


print("Done.")
print(f"Correct: {len(correct_list)}")
print(f"Error: {len(error_list)}")
print(f"Refusal: {len(refusal_list)}")