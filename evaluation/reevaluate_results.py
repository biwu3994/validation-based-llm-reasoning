import json
import re

INPUT_FILE = "results_neulr_deductive_framework.json"
OUTPUT_FILE = "results_neulr_deductive_framework_fixed.json"


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()

    if text.endswith("."):
        text = text[:-1]

    text = re.sub(r"\bthe\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_refusal(raw_prediction: str) -> bool:
    refusal_patterns = [
        "cannot answer",
        "can't answer",
        "can not answer",
        "unable to answer",
        "do not know",
        "don't know",
        "not enough information",
        "insufficient information",
        "cannot determine",
        "can't determine",
        "can not determine",
        "unknown",
        "i cannot answer",
        "i can't answer",
        "i do not know",
        "i don't know"
    ]

    prediction_lower = str(raw_prediction).strip().lower()
    return any(p in prediction_lower for p in refusal_patterns)


def recompute_summary(summary: dict) -> dict:
    results = summary["results"]

    correct = sum(1 for r in results if r.get("status") == "correct")
    error = sum(1 for r in results if r.get("status") == "error")
    refusal = sum(1 for r in results if r.get("status") == "refusal")
    total = len(results)

    summary["total_samples"] = total
    summary["correct_count"] = correct
    summary["error_count"] = error
    summary["refusal_count"] = refusal
    summary["answer_correctness_rate"] = correct / total if total else 0
    summary["answer_error_rate"] = error / total if total else 0
    summary["refusal_rate"] = refusal / total if total else 0

    return summary


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, list):
    summary = {
        "dataset_name": "",
        "model_name": "",
        "results": data
    }
else:
    summary = data

results = summary["results"]

fixed_count = 0

for r in results:
    old_prediction = r.get("prediction", "")
    old_status = r.get("status", "")

    raw = r.get("raw_prediction", "")
    gold = normalize_text(r.get("gold", ""))

    pred = normalize_text(raw)

    if is_refusal(raw):
        status = "refusal"
    elif pred == gold:
        status = "correct"
    else:
        status = "error"

    r["gold"] = gold
    r["prediction"] = pred
    r["status"] = status

    if old_prediction != pred or old_status != status:
        fixed_count += 1

summary = recompute_summary(summary)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Re-evaluation done.")
print("Fixed samples:", fixed_count)
print("Total:", summary["total_samples"])
print("Correct:", summary["correct_count"])
print("Error:", summary["error_count"])
print("Refusal:", summary["refusal_count"])
print("Accuracy:", summary["answer_correctness_rate"])