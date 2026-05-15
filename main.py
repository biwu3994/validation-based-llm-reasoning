from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re

error_streak = 0
MAX_ERROR_STREAK = 5

load_dotenv("config.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====== Switch the dataset ======
DATASET_NAME = "clutrr_mixed"
INPUT_FILE = "data/processed/clutrr_mixed_processed.json"
OUTPUT_FILE = "results/results_clutrr_mixed_baseline.json"
MODEL_NAME = "gpt-5"
# ======================================


def normalize_text(text: str) -> str:
    text = text.strip().lower()

    if text.endswith("."):
        text = text[:-1]

    # Remove all independent occurrences of "the"
    text = re.sub(r"\bthe\b", "", text)
    
    # Compress extra spaces into a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text

def get_instruction(dataset_name: str) -> str:
    if dataset_name in ["clutrr_clean", "clutrr_mixed", "neulr_deductive", "neulr_inductive"]:
        return "Answer with one word only."
    elif dataset_name == "neulr_abductive":
        return "Answer with a single fact only. Copy the relation wording exactly as used in the context. Do not paraphrase or explain."
    else:
        raise ValueError(f"Unknown DATASET_NAME: {dataset_name}")


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
    prediction_lower = raw_prediction.strip().lower()
    return any(pattern in prediction_lower for pattern in refusal_patterns)


def load_existing_results(output_file: str) -> dict:
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {
                "dataset_name": DATASET_NAME,
                "model_name": MODEL_NAME,
                "total_samples": len(data),
                "correct_count": 0,
                "error_count": 0,
                "refusal_count": 0,
                "answer_correctness_rate": 0,
                "answer_error_rate": 0,
                "refusal_rate": 0,
                "results": data
            }
        return data
    else:
        return {
            "dataset_name": DATASET_NAME,
            "model_name": MODEL_NAME,
            "total_samples": 0,
            "correct_count": 0,
            "error_count": 0,
            "refusal_count": 0,
            "answer_correctness_rate": 0,
            "answer_error_rate": 0,
            "refusal_rate": 0,
            "results": []
        }


def recompute_summary(summary: dict) -> dict:
    results = summary["results"]

    correct_count = sum(1 for r in results if r["status"] == "correct")
    error_count = sum(1 for r in results if r["status"] == "error")
    refusal_count = sum(1 for r in results if r["status"] == "refusal")

    total = len(results)

    summary["total_samples"] = total
    summary["correct_count"] = correct_count
    summary["error_count"] = error_count
    summary["refusal_count"] = refusal_count
    summary["answer_correctness_rate"] = correct_count / total if total else 0
    summary["answer_error_rate"] = error_count / total if total else 0
    summary["refusal_rate"] = refusal_count / total if total else 0

    return summary


def save_summary(output_file: str, summary: dict) -> None:
    summary = recompute_summary(summary)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

instruction = get_instruction(DATASET_NAME)

summary = load_existing_results(OUTPUT_FILE)
results = summary["results"]

completed_ids = {str(r["id"]) for r in results}

print(f"Loaded {len(results)} existing results from {OUTPUT_FILE}")
print(f"Total input samples: {len(data)}")

for i, sample in enumerate(data, start=1):
    sample_id = str(sample["id"])

    if sample_id in completed_ids:
        print(f"[{i}/{len(data)}] Skipping completed sample: {sample_id}")
        continue

    prompt = f"""Context:
{sample['context']}

Question:
{sample['question']}

{instruction}
"""

    print(f"[{i}/{len(data)}] Processing sample: {sample_id}")

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt
        )
        raw_prediction = response.output_text.strip()
        
        # Success → Reset error count
        error_streak = 0

    except Exception as e:
        print(f"Error on sample {sample_id}: {e}")
        raw_prediction = "ERROR"
        
        # Error → Increment error count
        error_streak += 1
        print(f"Consecutive errors: {error_streak}/{MAX_ERROR_STREAK}")
        
        if error_streak >= MAX_ERROR_STREAK:
            print("Too many consecutive errors, stopping to prevent waste.")
            break

    prediction = normalize_text(raw_prediction)
    gold = normalize_text(sample["label"])

    if is_refusal(raw_prediction):
        status = "refusal"
    elif prediction == gold:
        status = "correct"
    else:
        status = "error"

    results.append({
        "id": sample["id"],
        "gold": gold,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "status": status
    })

    save_summary(OUTPUT_FILE, summary)
    completed_ids.add(sample_id)

    print(f"Saved sample {sample_id} | status = {status}")

save_summary(OUTPUT_FILE, summary)

print("Answer correctness rate:", summary["answer_correctness_rate"])
print("Answer error rate:", summary["answer_error_rate"])
print("Refusal rate:", summary["refusal_rate"])
print(f"Results saved to {OUTPUT_FILE}")