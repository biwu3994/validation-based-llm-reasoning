import json

INPUT_FILE = "results/results_clutrr_mixed_framework.json"

OUTPUT_ERROR = "results/error_full_clutrr_mixed_framework.json"
OUTPUT_REFUSAL = "results/refusal_full_clutrr_mixed_framework.json"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, list):
        results = data
    else:
        raise ValueError("Unsupported JSON format: expected a dict with 'results' or a list.")

    error_samples = []
    refusal_samples = []

    for item in results:
        if not isinstance(item, dict):
            continue

        status = item.get("status", "")

        if status == "error":
            error_samples.append(item)
        elif status == "refusal":
            refusal_samples.append(item)

    with open(OUTPUT_ERROR, "w", encoding="utf-8") as f:
        json.dump(error_samples, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_REFUSAL, "w", encoding="utf-8") as f:
        json.dump(refusal_samples, f, indent=2, ensure_ascii=False)

    print(f"Done.")
    print(f"Error samples: {len(error_samples)} → {OUTPUT_ERROR}")
    print(f"Refusal samples: {len(refusal_samples)} → {OUTPUT_REFUSAL}")


if __name__ == "__main__":
    main()