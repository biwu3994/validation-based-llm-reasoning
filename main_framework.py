from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re

error_streak = 0
MAX_ERROR_STREAK = 5

load_dotenv("config.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATASET_NAME = "clutrr_mixed"
INPUT_FILE = "data/processed/clutrr_mixed_processed.json"
OUTPUT_FILE = "results/results_clutrr_mixed_framework.json"
MODEL_NAME = "gpt-5"
DEBUG_N = None   # None means run all samples, otherwise only run the first N samples.


def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    if text.endswith("."):
        text = text[:-1]
    text = re.sub(r"\bthe\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_graph_component(text: str) -> str:
    text = str(text).strip().lower()
    if text.endswith("."):
        text = text[:-1]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_answer_instruction(dataset_name: str) -> str:
    if dataset_name in ["clutrr_clean", "clutrr_mixed", "neulr_deductive", "neulr_inductive"]:
        return "Answer with one word only."
    elif dataset_name == "neulr_abductive":
        return (
            "Answer with a single fact only. "
            "Use the natural-language surface wording from the context, not predicate notation. "
            "For third-person singular subjects, preserve the surface verb form used in the context, such as 'needs', 'eats', 'likes', or 'visits'. "
            "Do not paraphrase or explain."
            )
    else:
        raise ValueError(f"Unknown DATASET_NAME: {dataset_name}")

def is_refusal(raw_prediction: str) -> bool:
    refusal_patterns = [
        "cannot answer", "can't answer", "can not answer",
        "unable to answer", "do not know", "don't know",
        "not enough information", "insufficient information",
        "cannot determine", "can't determine", "can not determine",
        "unknown", "i cannot answer", "i can't answer",
        "i do not know", "i don't know"
    ]
    prediction_lower = str(raw_prediction).strip().lower()
    return any(pattern in prediction_lower for pattern in refusal_patterns)


def make_empty_graph():
    return {"nodes": [], "edges": []}


def safe_json_load(text: str):
    text = str(text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


def call_model(prompt: str, sample_id: str) -> str:
    global error_streak

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt
        )
        raw_text = response.output_text.strip()
        error_streak = 0
        return raw_text

    except Exception as e:
        print(f"Error on sample {sample_id}: {e}")
        error_streak += 1
        print(f"Consecutive errors: {error_streak}/{MAX_ERROR_STREAK}")

        if error_streak >= MAX_ERROR_STREAK:
            raise RuntimeError("Too many consecutive API errors, stopping to prevent waste.")

        return "ERROR"


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


def clean_edge_obj(obj: dict):
    if not isinstance(obj, dict):
        return None

    source = str(obj.get("source", "")).strip()
    relation = str(obj.get("relation", "")).strip()
    target = str(obj.get("target", "")).strip()

    if not source or not relation or not target:
        return None

    out = {
        "source": source,
        "relation": relation,
        "target": target
    }

    if "role" in obj:
        out["role"] = str(obj.get("role", "")).strip()

    return out

def canonical_class_label(text: str) -> str:
    """
    Only used for neulr_deductive class-label surface alignment.
    Handles simple singular/plural variants such as SP2Luin0 vs SP2Luin0s.
    """
    x = normalize_graph_component(text)

    # Only apply to code-like class labels, not ordinary words.
    if re.fullmatch(r"[a-z0-9]+s", x):
        return x[:-1]

    return x


def normalize_graph_component_for_deductive(text: str) -> str:
    """
    Deductive-only normalization:
    - normal graph normalization
    - plus class-label singular/plural alignment
    """
    return canonical_class_label(text)


BINARY_PRED_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*(?:\s+[A-Za-z_][A-Za-z0-9_-]*)*)\((.+?),(.+?)\)\s*$")
UNARY_PRED_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_-]*(?:\s+[A-Za-z_][A-Za-z0-9_-]*)*)\((.+?)\)\s*$")
RULE_RE = re.compile(r"^\s*rule\((.+)\)\s*$")


def parse_predicate(expr: str):
    expr = str(expr).strip()

    m2 = BINARY_PRED_RE.match(expr)
    if m2:
        rel = m2.group(1).strip()
        arg1 = m2.group(2).strip()
        arg2 = m2.group(3).strip()
        if rel and arg1 and arg2:
            return ("binary", arg1, rel, arg2)

    m1 = UNARY_PRED_RE.match(expr)
    if m1:
        rel = m1.group(1).strip()
        arg1 = m1.group(2).strip()
        if rel and arg1:
            return ("unary", arg1, rel, None)

    return None


def normalize_predicate_for_match(expr: str) -> str:
    expr = str(expr).strip()
    parsed = parse_predicate(expr)

    if parsed is None:
        return normalize_graph_component(expr)

    kind, arg1, rel, arg2 = parsed
    rel = normalize_graph_component(rel)
    arg1 = normalize_graph_component(arg1)

    if kind == "binary":
        arg2 = normalize_graph_component(arg2)
        return f"{rel}({arg1},{arg2})"

    return f"{rel}({arg1})"


def normalize_graph_edge(edge: dict):
    parsed = clean_edge_obj(edge)
    if parsed is None:
        return None

    src = normalize_graph_component(parsed["source"])
    rel = normalize_graph_component(parsed["relation"])
    
    if rel == "gender information":
        rel = "gender"
    
    tgt_raw = str(parsed["target"]).strip()

    if not src or not rel or not tgt_raw:
        return None

    if rel in {"premise", "conclusion"}:
        src = "__rule__"
        tgt = normalize_predicate_for_match(tgt_raw)
    else:
        tgt = normalize_graph_component(tgt_raw)

    return (src, rel, tgt)

def normalize_graph_edge_deductive(edge: dict):
    parsed = clean_edge_obj(edge)
    if parsed is None:
        return None

    src = normalize_graph_component(parsed["source"])
    rel = normalize_graph_component(parsed["relation"])
    tgt_raw = str(parsed["target"]).strip()

    if not src or not rel or not tgt_raw:
        return None

    if rel in {"premise", "conclusion"}:
        src = "__rule__"
        tgt = normalize_predicate_for_match_deductive(tgt_raw)
    else:
        tgt = normalize_graph_component_for_deductive(tgt_raw)

    return (src, rel, tgt)


def normalize_predicate_for_match_deductive(expr: str) -> str:
    expr = str(expr).strip()
    parsed = parse_predicate(expr)

    if parsed is None:
        return normalize_graph_component(expr)

    kind, arg1, rel, arg2 = parsed

    rel = normalize_graph_component(rel)
    arg1 = normalize_graph_component(arg1)

    if kind == "binary":
        arg2 = normalize_graph_component_for_deductive(arg2)
        return f"{rel}({arg1},{arg2})"

    return f"{rel}({arg1})"

def storage_edge_key(edge: dict):
    parsed = clean_edge_obj(edge)
    if parsed is None:
        return None

    src = normalize_graph_component(parsed["source"])
    rel = normalize_graph_component(parsed["relation"])
    
    if rel == "gender information":
        rel = "gender"
    
    tgt = normalize_graph_component(parsed["target"])

    if not src or not rel or not tgt:
        return None

    return (src, rel, tgt)


def unique_edges(edges: list) -> list:
    deduped = []
    seen = set()

    for edge in edges:
        parsed = clean_edge_obj(edge)
        if parsed is None:
            continue

        key = storage_edge_key(parsed)
        if key is None or key in seen:
            continue

        seen.add(key)
        deduped.append(parsed)

    return deduped


def build_graph_from_edges(edges: list) -> dict:
    cleaned = unique_edges(edges)
    nodes = set()

    for edge in cleaned:
        nodes.add(edge["source"])
        nodes.add(edge["target"])

    return {
        "nodes": list(nodes),
        "edges": cleaned
    }


def merge_graphs(G1: dict, G2: dict) -> dict:
    nodes = set(G1.get("nodes", [])) | set(G2.get("nodes", []))
    edges = unique_edges(G1.get("edges", []) + G2.get("edges", []))
    return {"nodes": list(nodes), "edges": edges}


def graph_edges_to_text(edges: list) -> str:
    lines = []
    for edge in edges:
        if isinstance(edge, dict):
            lines.append(f'{edge.get("source", "")} -{edge.get("relation", "")}-> {edge.get("target", "")}')
    return "\n".join(lines)


def normalize_predicate_text(text: str) -> str:
    text = str(text).strip()
    parsed = parse_predicate(text)

    def norm_arg(arg: str) -> str:
        a = normalize_graph_component(arg)

        if re.fullmatch(r"\?[A-Za-z_][A-Za-z0-9_]*", a):
            return "var"
        if re.fullmatch(r"[xyzuvw]", a, flags=re.IGNORECASE):
            return "var"
        if a in {"someone", "somebody", "person", "entity"}:
            return "var"

        return a

    if parsed is None:
        return normalize_graph_component(text)

    kind, arg1, rel, arg2 = parsed
    rel = normalize_graph_component(rel)
    arg1 = norm_arg(arg1)

    if kind == "binary":
        arg2 = norm_arg(arg2)
        return f"{rel}({arg1},{arg2})"

    return f"{rel}({arg1})"


def extract_rule_signatures(graph: dict) -> set:
    rule_map = {}

    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue

        rel = normalize_graph_component(str(edge.get("relation", "")))
        src = str(edge.get("source", "")).strip()
        tgt = str(edge.get("target", "")).strip()

        if rel not in {"premise", "conclusion"}:
            continue

        if src not in rule_map:
            rule_map[src] = {
                "premises": [],
                "conclusions": []
            }

        tgt_norm = normalize_predicate_text(tgt)

        if rel == "premise":
            rule_map[src]["premises"].append(tgt_norm)
        else:
            rule_map[src]["conclusions"].append(tgt_norm)

    signatures = set()

    for _, content in rule_map.items():
        premises = tuple(sorted(p for p in content["premises"] if p))
        conclusions = [c for c in content["conclusions"] if c]

        for conclusion in conclusions:
            signatures.add((premises, conclusion))

    return signatures

def normalize_predicate_text_deductive(text: str) -> str:
    text = str(text).strip()
    parsed = parse_predicate(text)

    def norm_arg(arg: str) -> str:
        a = normalize_graph_component(arg)

        if re.fullmatch(r"\?[A-Za-z_][A-Za-z0-9_]*", a):
            return "var"
        if re.fullmatch(r"[xyzuvw]", a, flags=re.IGNORECASE):
            return "var"
        if a in {"someone", "somebody", "person", "entity"}:
            return "var"

        return canonical_class_label(a)

    if parsed is None:
        return normalize_graph_component(text)

    kind, arg1, rel, arg2 = parsed
    rel = normalize_graph_component(rel)
    arg1 = norm_arg(arg1)

    if kind == "binary":
        arg2 = norm_arg(arg2)
        return f"{rel}({arg1},{arg2})"

    return f"{rel}({arg1})"


def extract_rule_signatures_deductive(graph: dict) -> set:
    rule_map = {}

    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue

        rel = normalize_graph_component(str(edge.get("relation", "")))
        src = str(edge.get("source", "")).strip()
        tgt = str(edge.get("target", "")).strip()

        if rel not in {"premise", "conclusion"}:
            continue

        if src not in rule_map:
            rule_map[src] = {
                "premises": [],
                "conclusions": []
            }

        tgt_norm = normalize_predicate_text_deductive(tgt)

        if rel == "premise":
            rule_map[src]["premises"].append(tgt_norm)
        else:
            rule_map[src]["conclusions"].append(tgt_norm)

    signatures = set()

    for _, content in rule_map.items():
        premises = tuple(sorted(p for p in content["premises"] if p))
        conclusions = [c for c in content["conclusions"] if c]

        for conclusion in conclusions:
            signatures.add((premises, conclusion))

    return signatures

def extract_non_rule_edges(graph: dict) -> list:
    out = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue

        rel = normalize_graph_component(str(edge.get("relation", "")))
        if rel not in {"premise", "conclusion"}:
            out.append(edge)

    return out


def extract_rule_edges(graph: dict) -> list:
    out = []
    for edge in graph.get("edges", []):
        if not isinstance(edge, dict):
            continue

        rel = normalize_graph_component(str(edge.get("relation", "")))
        if rel in {"premise", "conclusion"}:
            out.append(edge)

    return out


def compare_graph_support(G_C: dict, G_base: dict, dataset_name: str = ""):
    unsupported = []

    if dataset_name == "neulr_deductive":
        edge_normalizer = normalize_graph_edge_deductive
    else:
        edge_normalizer = normalize_graph_edge

    base_non_rule_set = {
        edge_normalizer(edge)
        for edge in extract_non_rule_edges(G_base)
        if edge_normalizer(edge) is not None
    }

    for edge in extract_non_rule_edges(G_C):
        norm = edge_normalizer(edge)
        if norm is None or norm not in base_non_rule_set:
            unsupported.append(edge)

    c_rule_edges = extract_rule_edges(G_C)
    if c_rule_edges:
        c_rule_graph = {"nodes": G_C.get("nodes", []), "edges": c_rule_edges}
        base_rule_graph = {
            "nodes": G_base.get("nodes", []),
            "edges": extract_rule_edges(G_base)
        }

        if dataset_name == "neulr_deductive":
            c_rule_signatures = extract_rule_signatures_deductive(c_rule_graph)
            base_rule_signatures = extract_rule_signatures_deductive(base_rule_graph)
        else:
            c_rule_signatures = extract_rule_signatures(c_rule_graph)
            base_rule_signatures = extract_rule_signatures(base_rule_graph)

        if not c_rule_signatures.issubset(base_rule_signatures):
            unsupported.extend(c_rule_edges)

    valid = len(unsupported) == 0
    return valid, unsupported


def is_subgraph(G_C: dict, G_base: dict, dataset_name: str = ""):
    return compare_graph_support(G_C, G_base, dataset_name)


def extract_inductive_patterns(G_I: dict) -> dict:
    """
    通用归纳 pattern：
    category(entity, category_value) + property(entity, property_value)
    =>
    category_value maps_to property_value
    """
    categories_by_entity = {}
    props_by_entity = {}

    for edge in G_I.get("edges", []):
        if not isinstance(edge, dict):
            continue

        s = str(edge.get("source", "")).strip()
        r = normalize_graph_component(edge.get("relation", ""))
        t = str(edge.get("target", "")).strip()

        if not s or not r or not t:
            continue

        if r == "category":
            categories_by_entity.setdefault(s, set()).add(t)
        elif r == "property":
            props_by_entity.setdefault(s, set()).add(t)

    pattern_edges = []

    for entity, categories in categories_by_entity.items():
        props = props_by_entity.get(entity, set())

        for category in categories:
            for prop in props:
                pattern_edges.append({
                    "source": category,
                    "relation": "maps_to",
                    "target": prop
                })

    return build_graph_from_edges(pattern_edges)


def extract_inductive_query_entity(Q: str) -> str:
    m = re.search(r"What property is\s+([A-Za-z0-9]+)\s*\??", Q, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def build_GC_from_chain(C: list) -> dict:
    edges = []
    rule_idx = 1

    for step in C:
        if not isinstance(step, dict):
            continue

        step_type = str(step.get("type", "")).strip()
        expr = str(step.get("expr", "")).strip()

        if not step_type or not expr:
            continue

        if step_type == "rule":
            m_rule = RULE_RE.match(expr)
            if not m_rule:
                continue

            body = m_rule.group(1).strip()
            if "->" not in body:
                continue

            left, right = body.split("->", 1)
            left = left.strip()
            right = right.strip()

            if not left or not right:
                continue

            rule_node = f"RuleStep_{rule_idx}"
            rule_idx += 1

            premises = [p.strip() for p in left.split("&") if p.strip()]
            for premise in premises:
                edges.append({
                    "source": rule_node,
                    "relation": "premise",
                    "target": premise
                })

            edges.append({
                "source": rule_node,
                "relation": "conclusion",
                "target": right
            })
            continue

        if step_type in {"given_fact", "candidate_answer", "pattern"}:
            parsed = parse_predicate(expr)

            if parsed is None:
                if step_type == "candidate_answer":
                    edges.append({
                        "source": "__PARSE_FAILED__",
                        "relation": "__candidate_parse_failed__",
                        "target": expr,
                        "role": "candidate_answer"
                    })
                continue

            kind, arg1, rel, arg2 = parsed

            if kind == "binary":
                rel_norm = normalize_graph_component(rel)
                arg2_norm = normalize_graph_component(arg2)

                # CLUTRR: tolerate model writing is(Entity,male/female)
                # because G_I stores gender as gender(Entity,male/female).
                if (
                    DATASET_NAME in ["clutrr_clean", "clutrr_mixed"]
                    and rel_norm == "is"
                    and arg2_norm in {"male", "female"}
                ):
                    edge = {
                        "source": arg1,
                        "relation": "gender",
                        "target": arg2
                    }
                else:
                    edge = {
                        "source": arg1,
                        "relation": rel,
                        "target": arg2
                    }

            else:
                # Inductive task: unary Property(Entity) becomes property(Entity,Property).
                if DATASET_NAME == "neulr_inductive" and step_type in {"given_fact", "candidate_answer"}:
                    edge = {
                        "source": arg1,
                        "relation": "property",
                        "target": rel
                    }
                else:
                    if rel.startswith("is_") and len(rel) > 3:
                        edge = {
                            "source": arg1,
                            "relation": "is",
                            "target": rel[3:]
                        }
                    else:
                        edge = {
                            "source": arg1,
                            "relation": "is",
                            "target": rel
                        }

            if step_type == "candidate_answer":
                edge["role"] = "candidate_answer"

            edges.append(edge)

    return build_graph_from_edges(edges)


def surface_answer_matches_edge(answer: str, edge: dict) -> bool:
    if not isinstance(edge, dict):
        return False

    source = str(edge.get("source", "")).strip()
    relation = str(edge.get("relation", "")).strip()
    target = str(edge.get("target", "")).strip()

    if not source or not relation or not target:
        return False

    answer_norm = normalize_text(answer)
    src_norm = normalize_text(source)
    rel_norm = normalize_text(relation)
    tgt_norm = normalize_text(target)

    expected = normalize_text(f"{source} {relation} {target}")
    if answer_norm == expected:
        return True

    prefix = src_norm + " "
    suffix = " " + tgt_norm

    if not answer_norm.startswith(prefix):
        return False
    if not answer_norm.endswith(suffix):
        return False

    middle = answer_norm[len(prefix): len(answer_norm) - len(suffix)].strip()
    return middle == rel_norm


def judge_conflict_with_graph_by_model(edge: dict, G_base: dict, sample_id: str) -> bool:
    edge_text = json.dumps(edge, ensure_ascii=False)
    base_text = graph_edges_to_text(G_base.get("edges", []))

    prompt = f"""You are given:
1. a candidate edge
2. a base graph G_base

Decide whether the candidate edge is in explicit conflict with G_base.

Rules:
- Conflict means G_base explicitly states information that contradicts the candidate edge.
- Do NOT reject an edge merely because it is missing from G_base.
- "Not supported" is NOT the same as "conflict".
- Only return conflict=true if there is explicit contradiction.
- Return JSON only.

Return JSON only in this format:
{{
  "conflict": true
}}

Candidate edge:
{edge_text}

G_base edges:
{base_text}
"""
    raw = call_model(prompt, sample_id)
    parsed = safe_json_load(raw)

    if parsed and isinstance(parsed.get("conflict"), bool):
        return parsed["conflict"]

    return False


def is_rule_edge(edge: dict) -> bool:
    if not isinstance(edge, dict):
        return False

    rel = normalize_graph_component(str(edge.get("relation", "")))
    return rel in {"premise", "conclusion"}


def validate_graph_match(G_C: dict, G_base: dict, dataset_name: str, answer: str, sample_id: str):
    valid, unsupported = is_subgraph(G_C, G_base, dataset_name)

    if valid:
        return True, unsupported, "strict_match"

    if dataset_name != "neulr_abductive":
        return False, unsupported, "invalid"

    if len(unsupported) != 1:
        return False, unsupported, "invalid"

    missing_edge = unsupported[0]

    if is_rule_edge(missing_edge):
        return False, unsupported, "invalid"

    if str(missing_edge.get("role", "")).strip() != "candidate_answer":
        return False, unsupported, "invalid"

    if not surface_answer_matches_edge(answer, missing_edge):
        return False, unsupported, "invalid"

    if judge_conflict_with_graph_by_model(missing_edge, G_base, sample_id):
        return False, unsupported, "conflict_rejected"

    return True, unsupported, "single_missing_fact_release"


def validate_inductive_answer(G_I: dict, G_P: dict, Q: str, answer: str):
    query_entity = extract_inductive_query_entity(Q)
    pred = str(answer).strip()

    if not query_entity or not pred:
        return False, [], "invalid"

    query_norm = normalize_graph_component(query_entity)
    pred_norm = normalize_graph_component(pred)

    entity_categories = set()

    for edge in G_I.get("edges", []):
        norm = normalize_graph_edge(edge)
        if norm is None:
            continue

        s, r, t = norm
        if s == query_norm and r == "category":
            entity_categories.add(t)

    if not entity_categories:
        return False, [], "invalid"

    for edge in G_P.get("edges", []):
        norm = normalize_graph_edge(edge)
        if norm is None:
            continue

        s, r, t = norm
        if r == "maps_to" and s in entity_categories and t == pred_norm:
            return True, [], "pattern_match"

    return False, [], "invalid"

def answer_matches_deductive_edge(answer: str, edge: dict) -> bool:
    if not isinstance(edge, dict):
        return False

    target = str(edge.get("target", "")).strip()
    if not target:
        return False

    answer_norm = normalize_text(answer)
    target_norm = normalize_text(target)
    target_singular_norm = canonical_class_label(target)

    return answer_norm == target_norm or answer_norm == target_singular_norm


def build_deductive_closure(G_I: dict) -> dict:
    """
    Deductive closure for NeuLR deductive task.

    Handles:
    - instance/class membership: X is a C
    - class-level relation inheritance: C relation Y => X relation Y
    - simple class-label singular/plural alignment: C vs Cs
    """

    original_edges = unique_edges(G_I.get("edges", []))
    closure_edges = list(original_edges)

    memberships = []
    class_relations = []

    for edge in original_edges:
        if not isinstance(edge, dict):
            continue

        s = str(edge.get("source", "")).strip()
        r = normalize_graph_component(edge.get("relation", ""))
        t = str(edge.get("target", "")).strip()

        if not s or not r or not t:
            continue

        if r == "is a":
            memberships.append((s, t))
        else:
            class_relations.append((s, r, t))

    extra_edges = []

    for entity, cls in memberships:
        cls_norm = canonical_class_label(cls)

        for rel_source, rel, rel_target in class_relations:
            rel_source_norm = canonical_class_label(rel_source)

            if cls_norm == rel_source_norm:
                extra_edges.append({
                    "source": entity,
                    "relation": rel,
                    "target": rel_target
                })

    closure_edges = unique_edges(closure_edges + extra_edges)
    return build_graph_from_edges(closure_edges)


def validate_deductive_answer(G_I: dict, G_C: dict, answer: str):
    """
    For deductive task:
    - Do not require G_C to be a strict subgraph of G_I.
    - Build a small deductive closure from G_I.
    - Accept only if the candidate_answer edge is supported by the closure
      and the final answer matches its target.
    """

    closure = build_deductive_closure(G_I)

    candidate_edges = [
        e for e in G_C.get("edges", [])
        if isinstance(e, dict)
        and str(e.get("role", "")).strip() == "candidate_answer"
    ]

    if len(candidate_edges) != 1:
        return False, candidate_edges, "invalid"

    candidate = candidate_edges[0]

    cand_norm = normalize_graph_edge_deductive(candidate)

    closure_set = {
        normalize_graph_edge_deductive(e)
        for e in closure.get("edges", [])
        if normalize_graph_edge_deductive(e) is not None
    }

    if cand_norm not in closure_set:
        return False, [candidate], "invalid"

    if not answer_matches_deductive_edge(answer, candidate):
        return False, [candidate], "invalid"

    return True, [], "deductive_closure_match"

def generate_inductive_information_graph(I: str, sample_id: str):
    """
    Rule-based G_I extraction for NeuLR inductive task.

    For this task:
    - "is a" means category
    - "is" without "a" means property
    """

    edges = []

    # split by sentence
    sentences = [s.strip() for s in str(I).split(".") if s.strip()]

    for sent in sentences:
        # Entity is a Category
        m_category = re.match(r"^([A-Za-z0-9]+)\s+is\s+a\s+([A-Za-z0-9]+)$", sent)
        if m_category:
            edges.append({
                "source": m_category.group(1),
                "relation": "category",
                "target": m_category.group(2)
            })
            continue

        # Entity is Property
        m_property = re.match(r"^([A-Za-z0-9]+)\s+is\s+([A-Za-z0-9]+)$", sent)
        if m_property:
            edges.append({
                "source": m_property.group(1),
                "relation": "property",
                "target": m_property.group(2)
            })
            continue

    G_I = build_graph_from_edges(edges)

    raw = json.dumps({"G_I": G_I}, ensure_ascii=False, indent=2)
    return raw, G_I

def generate_information_graph(I: str, sample_id: str):
    if DATASET_NAME == "neulr_inductive":
        fact_instruction = """
How to represent ordinary facts:
- Use abstract relation names only for category/property distinction.
- Distinguish these semantic roles:
  1. category or class membership
  2. property or value assignment
  3. ordinary entity-to-entity relations

Use these abstract relation names:
- category: when the sentence says an entity belongs to a class/category/type.
- property: when the sentence says an entity has a property/value/attribute.
- ordinary surface relation: when the sentence describes an entity-to-entity relation.

Important:
- Do NOT collapse category membership and property assignment into the same relation.
- Do NOT use the same relation name for category and property.
- Do NOT rely on a specific surface wording such as "is" or "is a"; infer the semantic role from context.
"""
    else:
        fact_instruction = """
How to represent ordinary facts:
- Use normal fact edges, for example:
  {"source": "Entity1", "relation": "relation_name", "target": "Entity2"}
  {"source": "Entity1", "relation": "is", "target": "Type1"}

Relation wording rule:
- Treat relation names as surface forms from the context.
- Copy relation wording from the context as closely as possible.
- Do NOT convert relation names to another grammatical form.
- Use the same relation wording consistently in ordinary facts, rule predicates, and reasoning-chain expressions.
"""

    prompt = f"""You are given a context I.

Construct an information graph G_I that preserves ALL structured information explicitly stated in I.

Core rule:
- If something is explicitly stated in the context, it must be represented in G_I.
- Do NOT drop information because it seems unimportant.
- Do NOT summarize.
- Do NOT simplify away rules.
- Do NOT omit asserted statements.
- Do NOT use outside knowledge.
- Do NOT invent unstated facts.

You must preserve all explicitly stated structured content, including:
1. ordinary facts
2. rules
3. any other explicitly asserted statements

Graph schema:
- G_I must contain:
  - nodes
  - edges
- Each edge must have:
  - source
  - relation
  - target

{fact_instruction}

Entity string rule:
- Preserve code-like entity names and constants exactly.
- Do NOT lowercase, uppercase, rename, or reformat code-like strings.
- Do NOT include articles such as "the", "The", "a", or "an" as part of entity names.
- Example:
  If the context says "The Entity1 chases the Entity2", use:
  {{"source": "Entity1", "relation": "chases", "target": "Entity2"}}

How to represent rules:
- Create a rule node such as Rule1, Rule2, etc.
- Represent each premise and conclusion explicitly as edges.
- Use this schema:
  - {{"source": "Rule1", "relation": "premise", "target": "predicate(...)"}}
  - {{"source": "Rule1", "relation": "conclusion", "target": "predicate(...)"}}
- If a rule has multiple premises, include multiple premise edges.
- IMPORTANT: rule targets must use the SAME predicate-style notation used in reasoning chains.
- Do NOT represent rule premises or conclusions in natural language.
- Use generic variables such as X, Y when needed.
- For rule predicates, follow the same relation policy as ordinary facts.

Good rule examples:
- {{"source": "Rule1", "relation": "premise", "target": "relation(Entity1,Entity2)"}}
- {{"source": "Rule1", "relation": "premise", "target": "is(Entity1,Type1)"}}
- {{"source": "Rule1", "relation": "conclusion", "target": "is(Entity2,Type2)"}}

Important:
- Every explicit sentence in the context that contains structured information should be reflected in G_I.
- Preserve information, do not rank information.

Return JSON only in this format:
{{
  "G_I": {{
    "nodes": ["node1", "node2"],
    "edges": [
      {{"source": "node1", "relation": "relation_name", "target": "node2"}}
    ]
  }}
}}

Context I:
{I}
"""
    raw = call_model(prompt, sample_id)
    parsed = safe_json_load(raw)

    if parsed and isinstance(parsed.get("G_I"), dict):
        g = parsed["G_I"]
        nodes = g.get("nodes", [])
        edges = g.get("edges", [])

        if isinstance(nodes, list) and isinstance(edges, list):
            cleaned_edges = unique_edges(edges)

            inferred_nodes = set()
            for n in nodes:
                inferred_nodes.add(str(n).strip())

            for e in cleaned_edges:
                inferred_nodes.add(str(e.get("source", "")).strip())
                inferred_nodes.add(str(e.get("target", "")).strip())

            return raw, {
                "nodes": list(inferred_nodes),
                "edges": cleaned_edges
            }

    return raw, make_empty_graph()


def get_reasoning_prompt(
    I: str,
    Q: str,
    answer_instruction: str,
    dataset_name: str,
    E: str = "",
    G_Q=None,
    G_P=None
) -> str:
    gq_edges = G_Q.get("edges", []) if isinstance(G_Q, dict) else []
    gp_edges = G_P.get("edges", []) if isinstance(G_P, dict) else []

    gq_text = graph_edges_to_text(gq_edges)
    gp_text = graph_edges_to_text(gp_edges)

    is_second_round = bool(E) or bool(gq_edges)

    # =====================================================
    # Inductive task prompt
    # =====================================================
    if dataset_name == "neulr_inductive":
        second_round_text = ""
        if is_second_round:
            second_round_text = f"""
Second-round correction context:
Error explanation E:
{E}

Question-focused graph G_Q:
{gq_text}

Pattern graph G_P:
{gp_text}

G_P contains abstract category-to-property patterns extracted from G_I.
You may use G_P as support for candidate_answer.
"""

        return f"""You are given information I and question Q.

Generate:
1. a semi-structured reasoning chain C
2. a final answer

Rules for C:
- C must be a list of typed steps.
- Each step must have:
  - type
  - expr

Allowed step types:
1. given_fact
   - A fact explicitly stated in I or retained in G_Q.
2. pattern
   - A pattern from G_P.
   - Only use this if G_P is non-empty.
3. candidate_answer
   - The single missing fact proposed as the final answer.
   - There must be at most one candidate_answer step.

For this inductive task:
- Use category(Entity,Category) for category/class/type membership.
- Use property(Entity,Property) for property/value/attribute assignment.
- Use maps_to(Category,Property) for extracted inductive patterns.
- Do NOT rely on a specific surface wording such as "is" or "is a"; infer the semantic role.

Do NOT include derived intermediate facts as separate steps.
If a conclusion is derived from a pattern, do not add it as a normal fact unless it is the final candidate_answer.

Expression format:
- given_fact uses:
  category(Entity,Category)
  property(Entity,Property)
- pattern uses:
  maps_to(Category,Property)
- candidate_answer uses:
  property(Entity,Property)

Important:
- If a fact is not explicitly stated in I/G_Q and is not the final missing answer, do NOT include it in C.
- If a pattern is used, include the pattern step.
- Do NOT use source/relation/target JSON edges.
- Do NOT output markdown.
- Return JSON only.

String consistency rules:
- Entity names and constants must be copied EXACTLY from I or G_Q.
- Do NOT normalize, lowercase, or reformat entity names.
- Do NOT invent new entity names.

Final answer rules:
- {answer_instruction}
- The final answer must follow the dataset answer format exactly.
- The final answer must be plain answer text, not predicate notation.
- Do NOT explain.

Return JSON only in this format:
{{
  "C": [
    {{"type": "given_fact", "expr": "category(Entity1,Category1)"}},
    {{"type": "pattern", "expr": "maps_to(Category1,Property1)"}},
    {{"type": "candidate_answer", "expr": "property(Entity1,Property1)"}}
  ],
  "answer": "final answer"
}}

{second_round_text}

Information I:
{I}

Question Q:
{Q}
"""

    # =====================================================
    # Non-inductive tasks prompt
    # =====================================================
    second_round_text = ""

    if is_second_round:
        second_round_text = f"""
Error explanation E:
{E}

G_Q edges:
{gq_text}
"""

        intro = (
            "You are given information I, question Q, an error explanation E, "
            "and a question-focused graph G_Q.\n\n"
            "Revise the reasoning and answer."
        )

        if dataset_name in ["clutrr_clean", "clutrr_mixed"]:
            second_guidance = """
Second-round guidance:
- Use E to correct the previous reasoning.
- Use G_Q to stay focused.
- If a rule is used, include it explicitly in C.
- If a fact is used, include it explicitly in C.
- If G_Q contains a direct kinship edge from the queried source entity to the queried target entity, use that edge as candidate_answer.
- Do NOT answer unknown when such a direct query-direction edge exists in G_Q.
- Do NOT use unrelated G_Q edges as candidate_answer.
- For gender information in CLUTRR, write gender(Entity,Gender), not is(Entity,Gender).
"""
        else:
            second_guidance = """
Second-round guidance:
- Use E to correct the previous reasoning.
- Use G_Q to stay focused.
- If a rule is used, include it explicitly in C.
- If a fact is used, include it explicitly in C.
"""
    else:
        intro = (
            "You are given information I and question Q.\n\n"
            "Generate:\n"
            "1. a semi-structured reasoning chain C\n"
            "2. a final answer"
        )
        second_guidance = ""

    if dataset_name == "neulr_deductive":
        deductive_answer_rule = """
- For neulr_deductive, if the inferred answer corresponds to a plural form of a class label, return its base (singular) form.
- Do not include plural markers in the final answer when the question asks for a class label.
- The answer must match the canonical class label format used in the context.
"""
    else:
        deductive_answer_rule = ""

    return f"""{intro}

Rules for C:
- C must be a list of typed steps.
- Each step must have:
  - type
  - expr

Allowed step types:
1. given_fact
   - A fact explicitly stated in I.
   - This must be directly supported by G_I.
2. rule
   - A rule explicitly stated in I.
   - This must be directly supported by G_I.
3. candidate_answer
   - The single missing fact proposed as the final answer.
   - There must be at most one candidate_answer step.

Do NOT include derived intermediate facts as separate steps.
If a conclusion is derived from a rule, do not add it as a normal fact unless it is the final candidate_answer.

Expression format:
- given_fact uses predicate notation:
  relation(Entity1,Entity2)
  is(Entity,Type)
  is not(Entity,Type)
  does not relation(Entity1,Entity2)
- rule uses:
  rule(premise1->conclusion)
  rule(premise1&premise2->conclusion)
- candidate_answer uses predicate notation:
  relation(Entity1,Entity2)
  is(Entity,Type)
  is not(Entity,Type)
  does not relation(Entity1,Entity2)

Surface relation preservation rule:
- If the context says "is a", write:
  is a(Entity,Class)
  NOT:
  is(Entity,Class)
- If the context says "is", write:
  is(Entity,Value)
- Treat "is" and "is a" as different relations.
- Preserve multi-word relation names exactly:
  are afraid of(Entity1,Entity2)
  does not chase(Entity1,Entity2)
  is a(Entity,Class)
- Do NOT convert multi-word relations into underscores.
  Write:
    are afraid of(Entity1,Entity2)
  NOT:
    afraid_of(Entity1,Entity2)

Negation rule:
- Do NOT use nested negation like not(relation(Entity1,Entity2)).
- Use direct relation names instead:
  does not visit(Entity1,Entity2)
  does not need(Entity1,Entity2)
  is not(Entity,Type)

Important:
- If a fact is not explicitly stated in I and is not the final missing answer, do NOT include it in C.
- If a rule is used, include the rule step.
- Do not encode type labels into predicate names.
  Write:
    is(Entity,Type)
  NOT:
    is_Type(Entity)

- Do NOT use source/relation/target JSON edges.
- Do NOT output markdown.
- Return JSON only.

{second_guidance}

Final answer rules:
- {answer_instruction}
- The final answer must follow the dataset answer format exactly.
- The final answer must be plain answer text, not predicate notation.
- For example, write:
  Entity1 relation Entity2
  not:
  relation(Entity1,Entity2)
- Do NOT explain.
{deductive_answer_rule}

Return JSON only in this format:
{{
  "C": [
    {{"type": "given_fact", "expr": "relation(Entity1,Entity2)"}},
    {{"type": "rule", "expr": "rule(relation(Entity1,Entity2)->is(Entity1,Type1))"}},
    {{"type": "candidate_answer", "expr": "relation(Entity1,Entity2)"}}
  ],
  "answer": "final answer"
}}

{second_round_text}

Information I:
{I}

Question Q:
{Q}
"""


def generate_reasoning_chain_round1(I: str, Q: str, sample_id: str, answer_instruction: str):
    prompt = get_reasoning_prompt(
        I=I,
        Q=Q,
        answer_instruction=answer_instruction,
        dataset_name=DATASET_NAME
    )

    raw = call_model(prompt, sample_id)
    parsed = safe_json_load(raw)

    if parsed:
        C = parsed.get("C", [])
        answer = parsed.get("answer", "")
        if not isinstance(C, list):
            C = []
        if not isinstance(answer, str):
            answer = str(answer)
        return raw, C, answer

    return raw, [], ""

def analyze_error_and_need_R(G_I: dict, G_C_1: dict, unsupported_edges: list, Q: str, sample_id: str):
    gi_text = graph_edges_to_text(G_I.get("edges", []))
    gc_text = graph_edges_to_text(G_C_1.get("edges", []))
    bad_text = graph_edges_to_text(unsupported_edges)

    prompt = f"""A first-round reasoning graph G_C^(1) failed validation against G_I.

Analyze the failure.

Return:
- E: a short explanation
- need_R: true or false
- R_reason: short reason, empty string if need_R is false

Guideline:
- need_R = false if the failure is mainly due to wrong reasoning, wrong target, wrong relation, reversed answer, bad assumption, or unsupported jump.
- need_R = true only if the key failure is that explicit graph G_I lacks relational expansion that could legitimately support a corrected second-round attempt.

Return JSON only in this format:
{{
  "E": "short explanation",
  "need_R": true,
  "R_reason": "why external relation expansion is needed"
}}

Question Q:
{Q}

G_I edges:
{gi_text}

G_C^(1) edges:
{gc_text}

Unsupported edges:
{bad_text}
"""
    raw = call_model(prompt, sample_id)
    parsed = safe_json_load(raw)

    if parsed:
        E = parsed.get("E", "")
        need_R = parsed.get("need_R", False)
        R_reason = parsed.get("R_reason", "")

        if not isinstance(E, str):
            E = str(E)
        if not isinstance(need_R, bool):
            need_R = False
        if not isinstance(R_reason, str):
            R_reason = str(R_reason)

        return raw, E, need_R, R_reason

    return raw, "Some edges in G_C^(1) are not supported by G_I.", False, ""


def generate_relation_expansion(G_I: dict, Q: str, E: str, R_reason: str, sample_id: str):
    gi_text = graph_edges_to_text(G_I.get("edges", []))

    if DATASET_NAME in ["clutrr_clean", "clutrr_mixed"]:
        extra_instruction = """
CLUTRR kinship-specific rules:
- Generate only kinship relation edges needed to answer Q.
- Prefer adding the inferred kinship relation in the query direction.
- If Q asks the relationship of EntityA to EntityB, include an edge:
  {"source": "EntityA", "relation": "kinship_relation", "target": "EntityB"}
- Use gender information in G_I to choose gendered kinship relations such as daughter, son, mother, father, wife, or husband.
- If useful, include both directions of an inferred relation.
  Example:
  {"source": "EntityB", "relation": "mother", "target": "EntityA"}
  {"source": "EntityA", "relation": "daughter", "target": "EntityB"}
- Do NOT use non-kinship story events as kinship relations.
- Do NOT infer step-relations such as stepdaughter, stepson, stepmother, or stepfather unless the context explicitly states remarriage, step-family, or non-biological parenthood. In CLUTRR, a parent's spouse should be treated as a parent for kinship inference.
"""
    else:
        extra_instruction = ""

    prompt = f"""You are given:
- an explicit graph G_I
- question Q
- an error explanation E
- a reason why external relational expansion is needed

Generate a relation expansion graph R.

Purpose of R:
- add only relation knowledge relevant to Q
- use R only as second-round support after first-round validation failed
- do not turn R into a full reasoning chain
- do not add unrelated knowledge

Rules:
- Prefer using entities already present in G_I.
- Do NOT introduce unrelated world knowledge.
- Use only the graph edge schema:
  - source
  - relation
  - target
- Return JSON only.

{extra_instruction}

Return JSON only in this format:
{{
  "R": {{
    "nodes": ["entity1", "entity2"],
    "edges": [
      {{"source": "entity1", "relation": "relation_name", "target": "entity2"}}
    ]
  }}
}}

Question Q:
{Q}

Error explanation E:
{E}

R_reason:
{R_reason}

G_I edges:
{gi_text}
"""
    raw = call_model(prompt, sample_id)
    parsed = safe_json_load(raw)

    if parsed and isinstance(parsed.get("R"), dict):
        r = parsed["R"]
        nodes = r.get("nodes", [])
        edges = r.get("edges", [])
        if isinstance(nodes, list) and isinstance(edges, list):
            return raw, {"nodes": nodes, "edges": unique_edges(edges)}

    return raw, make_empty_graph()


def fallback_extract_focus_entities(I: str, Q: str, G_base: dict):
    node_set = {str(n).strip() for n in G_base.get("nodes", [])}
    matched = []

    fact_match = re.search(r"The fact is:\s*(.+)", I, re.IGNORECASE)
    if fact_match:
        fact_text = fact_match.group(1)
        fact_tokens = re.findall(r"\b[A-Za-z0-9]{4,}\b", fact_text)
        for tok in fact_tokens:
            if tok in node_set and tok not in matched:
                matched.append(tok)

    q_tokens = re.findall(r"\b[A-Za-z0-9]{4,}\b", Q)
    for tok in q_tokens:
        if tok in node_set and tok not in matched:
            matched.append(tok)

    i_tokens = re.findall(r"\b[A-Za-z0-9]{4,}\b", I)
    for tok in i_tokens:
        if tok in node_set and tok not in matched:
            matched.append(tok)

    if len(matched) >= 2:
        return matched[0], matched[1]
    elif len(matched) == 1:
        return matched[0], ""
    else:
        return "", ""


def extract_query_focus(I: str, Q: str, sample_id: str):
    prompt = f"""You are given:
1. a context I
2. a question Q

Your task is to extract the query focus for reasoning.

Important:
- If Q explicitly mentions the target entities, use them.
- If Q asks for a property of an entity, return that entity as query_source.
- If Q is generic, extract the focus from the relevant statement in I.
- If one field cannot be identified reliably, use an empty string.
- Do NOT answer the question.
- Return JSON only.

Return JSON only in this format:
{{
  "query_source": "entity_or_empty",
  "query_relation": "relation_or_empty",
  "query_target": "entity_or_empty"
}}

Context I:
{I}

Question Q:
{Q}
"""
    raw = call_model(prompt, sample_id)
    parsed = safe_json_load(raw)

    if parsed:
        query_source = normalize_text(parsed.get("query_source", "")).strip()
        query_relation = normalize_text(parsed.get("query_relation", "")).strip()
        query_target = normalize_text(parsed.get("query_target", "")).strip()

        bad_relation_values = {"support", "fact", "relation", "information", "unknown", "missing"}
        if query_relation.lower() in bad_relation_values:
            query_relation = ""

        return raw, {
            "query_source": query_source,
            "query_relation": query_relation,
            "query_target": query_target
        }

    return raw, {
        "query_source": "",
        "query_relation": "",
        "query_target": ""
    }


def generate_question_focus_graph(G_base: dict, I: str, Q: str, sample_id: str):
    raw_focus_output, query_focus = extract_query_focus(I, Q, sample_id)

    query_source = normalize_text(query_focus.get("query_source", ""))
    query_relation = normalize_text(query_focus.get("query_relation", ""))
    query_target = normalize_text(query_focus.get("query_target", ""))

    if not query_source and not query_target:
        fb_source, fb_target = fallback_extract_focus_entities(I, Q, G_base)
        if fb_source:
            query_source = fb_source
        if fb_target:
            query_target = fb_target

    source_norm = normalize_graph_component(query_source) if query_source else ""
    target_norm = normalize_graph_component(query_target) if query_target else ""

    forced_edges = []
    forced_nodes = set()

    for edge in G_base.get("edges", []):
        if not isinstance(edge, dict):
            continue

        s_raw = str(edge.get("source", "")).strip()
        t_raw = str(edge.get("target", "")).strip()

        s = normalize_graph_component(s_raw)
        t = normalize_graph_component(t_raw)

        keep = False

        if source_norm and (s == source_norm or t == source_norm):
            keep = True
        if target_norm and (s == target_norm or t == target_norm):
            keep = True

        if keep:
            forced_edges.append(edge)
            forced_nodes.add(s_raw)
            forced_nodes.add(t_raw)

    forced_edges = unique_edges(forced_edges)

    base_edges_text = graph_edges_to_text(G_base.get("edges", []))
    forced_edges_text = graph_edges_to_text(forced_edges)

    prompt = f"""You are given:
1. a graph G_base
2. a context I
3. a question Q
4. an extracted query focus
5. a set of mandatory edges that must stay in G_Q

Construct a question-focused graph G_Q.

Purpose of G_Q:
- keep edges most relevant to answering Q
- reduce drift
- keep enough edges for second-round correction

Rules:
- G_Q must be a subgraph of G_base.
- All mandatory edges must be included in G_Q.
- Do NOT invent edges.
- Do NOT add outside knowledge.
- Only keep existing edges from G_base.
- Prefer edges around the query source and query target.
- Include a small number of bridge edges if they are plausibly relevant.
- Return JSON only.

Return JSON only in this format:
{{
  "G_Q": {{
    "nodes": ["entity1", "entity2"],
    "edges": [
      {{"source": "entity1", "relation": "relation_name", "target": "entity2"}}
    ]
  }}
}}

Context I:
{I}

Question Q:
{Q}

Extracted query focus:
source = {query_source}
relation = {query_relation}
target = {query_target}

Mandatory edges:
{forced_edges_text}

G_base edges:
{base_edges_text}
"""
    raw_gq_output = call_model(prompt, sample_id)
    parsed = safe_json_load(raw_gq_output)

    candidate_edges = []

    if parsed and isinstance(parsed.get("G_Q"), dict):
        gq = parsed["G_Q"]
        edges = gq.get("edges", [])
        if isinstance(edges, list):
            candidate_edges = edges

    merged_candidate_edges = unique_edges(candidate_edges + forced_edges)
    candidate_graph = build_graph_from_edges(merged_candidate_edges)

    base_edge_set = {
        normalize_graph_edge(edge)
        for edge in G_base.get("edges", [])
        if normalize_graph_edge(edge) is not None
    }

    filtered_edges = []
    for edge in candidate_graph.get("edges", []):
        norm = normalize_graph_edge(edge)
        if norm is not None and norm in base_edge_set:
            filtered_edges.append(edge)

    filtered_graph = build_graph_from_edges(filtered_edges)

    if not filtered_graph["edges"]:
        final_graph = G_base
    else:
        final_graph = filtered_graph

    raw_combined_output = json.dumps({
        "query_focus": {
            "query_source": query_source,
            "query_relation": query_relation,
            "query_target": query_target
        },
        "raw_focus_output": safe_json_load(raw_focus_output) if raw_focus_output else {},
        "model_G_Q_raw": parsed if parsed else {},
        "final_G_Q": final_graph
    }, ensure_ascii=False, indent=2)

    return raw_combined_output, final_graph


def generate_reasoning_chain_round2(
    I: str,
    Q: str,
    E: str,
    G_Q: dict,
    G_P: dict,
    sample_id: str,
    answer_instruction: str
):
    prompt = get_reasoning_prompt(
        I=I,
        Q=Q,
        answer_instruction=answer_instruction,
        dataset_name=DATASET_NAME,
        E=E,
        G_Q=G_Q,
        G_P=G_P
    )

    raw = call_model(prompt, sample_id)
    parsed = safe_json_load(raw)

    if parsed:
        C = parsed.get("C", [])
        answer = parsed.get("answer", "")
        if not isinstance(C, list):
            C = []
        if not isinstance(answer, str):
            answer = str(answer)
        return raw, C, answer

    return raw, [], ""


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

if DEBUG_N is not None:
    data = data[:DEBUG_N]

answer_instruction = get_answer_instruction(DATASET_NAME)
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

    print(f"[{i}/{len(data)}] Processing sample: {sample_id}")

    I = sample["context"]
    Q = sample["question"]
    gold = normalize_text(sample["label"])

    try:
        if DATASET_NAME == "neulr_inductive":
            raw_GI_output, G_I = generate_inductive_information_graph(I, sample_id)
        else:
            raw_GI_output, G_I = generate_information_graph(I, sample_id)

        if DATASET_NAME == "neulr_inductive":
            G_P = extract_inductive_patterns(G_I)
        else:
            G_P = make_empty_graph()

        raw_C1_output, C_1, answer_1 = generate_reasoning_chain_round1(
            I=I,
            Q=Q,
            sample_id=sample_id,
            answer_instruction=answer_instruction
        )

        if is_refusal(answer_1):
            G_C_1 = make_empty_graph()
            raw_GC1_output = ""
            valid_1 = False
            unsupported_1 = []
            validation_mode_1 = "direct_refusal_round_1"
        else:
            G_C_1 = build_GC_from_chain(C_1)
            raw_GC1_output = json.dumps({"G_C": G_C_1}, ensure_ascii=False, indent=2)

            if DATASET_NAME == "neulr_inductive":
                valid_1, unsupported_1, validation_mode_1 = validate_inductive_answer(
                    G_I=G_I,
                    G_P=G_P,
                    Q=Q,
                    answer=answer_1
                )
            elif DATASET_NAME == "neulr_deductive":
                valid_1, unsupported_1, validation_mode_1 = validate_deductive_answer(
                    G_I=G_I,
                    G_C=G_C_1,
                    answer=answer_1
                    )
            else:
                valid_1, unsupported_1, validation_mode_1 = validate_graph_match(
                    G_C_1, G_I, DATASET_NAME, answer_1, sample_id
                )

        if valid_1:
            raw_prediction = answer_1.strip()
            prediction = normalize_text(raw_prediction)

            if prediction == gold:
                status = "correct"
                refusal_stage = ""
                if validation_mode_1 == "pattern_match":
                    final_decision_reason = "The first reasoning result is accepted because the candidate property is supported by G_P, and the final answer matches the gold label."
                elif validation_mode_1 == "deductive_closure_match":
                    final_decision_reason = "The first reasoning result is accepted because the candidate answer is supported by the deductive closure of G_I, and the final answer matches the gold label."   
                elif validation_mode_1 == "strict_match":
                    final_decision_reason = "The first reasoning graph strictly matches G_I and the final answer matches the gold label."
                elif validation_mode_1 == "single_missing_fact_release":
                    final_decision_reason = "The first reasoning graph is accepted under the single-missing-fact release rule, and the final answer matches the gold label."
                else:
                    final_decision_reason = "The first reasoning graph is accepted and the final answer matches the gold label."
            else:
                status = "error"
                refusal_stage = ""
                if validation_mode_1 == "pattern_match":
                    final_decision_reason = "The first reasoning result is accepted because the candidate property is supported by G_P, but the final answer does not match the gold label."
                elif validation_mode_1 == "deductive_closure_match":
                    final_decision_reason = "The first reasoning result is accepted because the candidate answer is supported by the deductive closure of G_I, but the final answer does not match the gold label."
                elif validation_mode_1 == "strict_match":
                    final_decision_reason = "The first reasoning graph strictly matches G_I, but the final answer does not match the gold label."
                elif validation_mode_1 == "single_missing_fact_release":
                    final_decision_reason = "The first reasoning graph is accepted under the single-missing-fact release rule, but the final answer does not match the gold label."
                else:
                    final_decision_reason = "The first reasoning graph is accepted, but the final answer does not match the gold label."

            results.append({
                "id": sample["id"],
                "gold": gold,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "status": status,
                "refusal_stage": refusal_stage,
                "final_decision_reason": final_decision_reason,
                "need_R": False,
                "R_reason": "",
                "validation_mode_1": validation_mode_1,
                "validation_mode_2": "",
                "G_I": G_I,
                "G_P": G_P,
                "R": make_empty_graph(),
                "G_IR": make_empty_graph(),
                "G_Q": make_empty_graph(),
                "C_1": C_1,
                "G_C_1": G_C_1,
                "Valid_G_C_1": 1,
                "E": "",
                "C_2": [],
                "G_C_2": make_empty_graph(),
                "Valid_G_C_2": "",
                "unsupported_G_C_1": unsupported_1,
                "unsupported_G_C_2": [],
                "raw_GI_output": raw_GI_output,
                "raw_error_analysis_output": "",
                "raw_R_output": "",
                "raw_GQ_output": "",
                "raw_C1_output": raw_C1_output,
                "raw_GC1_output": raw_GC1_output,
                "raw_C2_output": "",
                "raw_GC2_output": ""
            })

        else:
            if validation_mode_1 == "direct_refusal_round_1":
                raw_error_analysis_output = ""
                E = "The model directly refused in the first reasoning round."
                need_R = False
                R_reason = ""
            else:
                raw_error_analysis_output, E, need_R, R_reason = analyze_error_and_need_R(
                    G_I=G_I,
                    G_C_1=G_C_1,
                    unsupported_edges=unsupported_1,
                    Q=Q,
                    sample_id=sample_id
                )
            if DATASET_NAME == "neulr_deductive":
                    need_R = False
                    R_reason = ""
            elif DATASET_NAME in ["clutrr_clean", "clutrr_mixed"]:
                need_R = True
                R_reason = (
                    "CLUTRR requires external kinship composition rules to infer unstated "
                    "family relations from explicit relations and gender information."
                    )
            
            if DATASET_NAME == "neulr_abductive":
                candidate_unsupported = [
                    e for e in unsupported_1
                    if str(e.get("role", "")).strip() == "candidate_answer"
                ]

                if candidate_unsupported:
                    need_R = False
                    R_reason = ""

            if need_R:
                raw_R_output, R = generate_relation_expansion(
                    G_I=G_I,
                    Q=Q,
                    E=E,
                    R_reason=R_reason,
                    sample_id=sample_id
                )
                validation_base = merge_graphs(G_I, R)
                G_IR = validation_base
            else:
                R = make_empty_graph()
                raw_R_output = ""
                validation_base = G_I
                G_IR = make_empty_graph()

            raw_GQ_output, G_Q = generate_question_focus_graph(
                G_base=validation_base,
                I=I,
                Q=Q,
                sample_id=sample_id
            )

            raw_C2_output, C_2, answer_2 = generate_reasoning_chain_round2(
                I=I,
                Q=Q,
                E=E,
                G_Q=G_Q,
                G_P=G_P,
                sample_id=sample_id,
                answer_instruction=answer_instruction
            )

            if is_refusal(answer_2):
                raw_prediction = answer_2.strip()
                prediction = normalize_text(raw_prediction)
                status = "refusal"
                refusal_stage = "direct_refusal_round_2"
                validation_mode_2 = "direct_refusal_round_2"

                final_decision_reason = "The first reasoning result was not accepted, and the model directly refused to answer in the second round."

                results.append({
                    "id": sample["id"],
                    "gold": gold,
                    "prediction": prediction,
                    "raw_prediction": raw_prediction,
                    "status": status,
                    "refusal_stage": refusal_stage,
                    "final_decision_reason": final_decision_reason,
                    "need_R": need_R,
                    "R_reason": R_reason,
                    "validation_mode_1": validation_mode_1,
                    "validation_mode_2": validation_mode_2,
                    "G_I": G_I,
                    "G_P": G_P,
                    "R": R,
                    "G_IR": G_IR,
                    "G_Q": G_Q,
                    "C_1": C_1,
                    "G_C_1": G_C_1,
                    "Valid_G_C_1": 0,
                    "E": E,
                    "C_2": C_2,
                    "G_C_2": make_empty_graph(),
                    "Valid_G_C_2": "",
                    "unsupported_G_C_1": unsupported_1,
                    "unsupported_G_C_2": [],
                    "raw_GI_output": raw_GI_output,
                    "raw_error_analysis_output": raw_error_analysis_output,
                    "raw_R_output": raw_R_output,
                    "raw_GQ_output": raw_GQ_output,
                    "raw_C1_output": raw_C1_output,
                    "raw_GC1_output": raw_GC1_output,
                    "raw_C2_output": raw_C2_output,
                    "raw_GC2_output": ""
                })

            else:
                G_C_2 = build_GC_from_chain(C_2)
                raw_GC2_output = json.dumps({"G_C": G_C_2}, ensure_ascii=False, indent=2)

                if DATASET_NAME == "neulr_inductive":
                    valid_2, unsupported_2, validation_mode_2 = validate_inductive_answer(
                        G_I=G_I,
                        G_P=G_P,
                        Q=Q,
                        answer=answer_2
                    )
                elif DATASET_NAME == "neulr_deductive":
                    valid_2, unsupported_2, validation_mode_2 = validate_deductive_answer(
                        G_I=validation_base,
                        G_C=G_C_2,
                        answer=answer_2
                        )
                else:
                    valid_2, unsupported_2, validation_mode_2 = validate_graph_match(
                        G_C_2, validation_base, DATASET_NAME, answer_2, sample_id
                    )

                if valid_2:
                    raw_prediction = answer_2.strip()
                    prediction = normalize_text(raw_prediction)

                    if prediction == gold:
                        status = "correct"
                        refusal_stage = ""
                        if validation_mode_2 == "pattern_match":
                            final_decision_reason = "The first reasoning result was not accepted. In round 2, the revised answer is accepted because the candidate property is supported by G_P, and the final answer matches the gold label."
                        elif validation_mode_2 == "strict_match":
                            final_decision_reason = "The first reasoning result was not accepted. In round 2, the revised reasoning graph strictly matches the validation graph and the final answer matches the gold label."
                        elif validation_mode_2 == "deductive_closure_match":
                            final_decision_reason = "The first reasoning result was not accepted. In round 2, the revised candidate answer is supported by the deductive closure of the validation graph, and the final answer matches the gold label."
                        elif validation_mode_2 == "single_missing_fact_release":
                            final_decision_reason = "The first reasoning result was not accepted. In round 2, the revised reasoning graph is accepted under the single-missing-fact release rule and the final answer matches the gold label."
                        else:
                            final_decision_reason = "The first reasoning result was not accepted. The second reasoning result is accepted and the final answer matches the gold label."
                    else:
                        status = "error"
                        refusal_stage = ""
                        if validation_mode_2 == "pattern_match":
                            final_decision_reason = "The first reasoning result was not accepted. In round 2, the revised answer is accepted because the candidate property is supported by G_P, but the final answer does not match the gold label."
                        elif validation_mode_2 == "strict_match":
                            final_decision_reason = "The first reasoning result was not accepted. In round 2, the revised reasoning graph strictly matches the validation graph, but the final answer does not match the gold label."
                        elif validation_mode_2 == "deductive_closure_match":
                            final_decision_reason = "The first reasoning result was not accepted. In round 2, the revised candidate answer is supported by the deductive closure of the validation graph, but the final answer does not match the gold label."
                        elif validation_mode_2 == "single_missing_fact_release":
                            final_decision_reason = "The first reasoning result was not accepted. In round 2, the revised reasoning graph is accepted under the single-missing-fact release rule, but the final answer does not match the gold label."
                        else:
                            final_decision_reason = "The first reasoning result was not accepted. The second reasoning result is accepted, but the final answer does not match the gold label."

                    results.append({
                        "id": sample["id"],
                        "gold": gold,
                        "prediction": prediction,
                        "raw_prediction": raw_prediction,
                        "status": status,
                        "refusal_stage": refusal_stage,
                        "final_decision_reason": final_decision_reason,
                        "need_R": need_R,
                        "R_reason": R_reason,
                        "validation_mode_1": validation_mode_1,
                        "validation_mode_2": validation_mode_2,
                        "G_I": G_I,
                        "G_P": G_P,
                        "R": R,
                        "G_IR": G_IR,
                        "G_Q": G_Q,
                        "C_1": C_1,
                        "G_C_1": G_C_1,
                        "Valid_G_C_1": 0,
                        "E": E,
                        "C_2": C_2,
                        "G_C_2": G_C_2,
                        "Valid_G_C_2": 1,
                        "unsupported_G_C_1": unsupported_1,
                        "unsupported_G_C_2": unsupported_2,
                        "raw_GI_output": raw_GI_output,
                        "raw_error_analysis_output": raw_error_analysis_output,
                        "raw_R_output": raw_R_output,
                        "raw_GQ_output": raw_GQ_output,
                        "raw_C1_output": raw_C1_output,
                        "raw_GC1_output": raw_GC1_output,
                        "raw_C2_output": raw_C2_output,
                        "raw_GC2_output": raw_GC2_output
                    })

                else:
                    raw_prediction = "cannot answer this question"
                    prediction = normalize_text(raw_prediction)
                    status = "refusal"
                    refusal_stage = "refusal_after_failed_validation"

                    if validation_mode_2 == "conflict_rejected":
                        if need_R:
                            final_decision_reason = (
                                "The second-round candidate fact conflicts with G_(I+R), "
                                "so the system refuses to answer."
                                )
                        else:
                            final_decision_reason = (
                                "The second-round candidate fact conflicts with G_I, "
                                "so the system refuses to answer."
                                )

                    elif DATASET_NAME == "neulr_inductive":
                        final_decision_reason = (
                            "The first reasoning result was not accepted, and the second-round answer "
                            "is still not supported by G_P, so the system refuses to answer."
                            )

                    elif need_R:
                        final_decision_reason = (
                            "The first reasoning result was not accepted, and the second reasoning graph "
                            "still failed validation against G_(I+R), so the system refuses to answer."
                            )

                    else:
                        final_decision_reason = (
                            "The first reasoning result was not accepted, and the second reasoning graph "
                            "still failed validation against G_I, so the system refuses to answer."
                            )

                    results.append({
                        "id": sample["id"],
                        "gold": gold,
                        "prediction": prediction,
                        "raw_prediction": raw_prediction,
                        "status": status,
                        "refusal_stage": refusal_stage,
                        "final_decision_reason": final_decision_reason,
                        "need_R": need_R,
                        "R_reason": R_reason,
                        "validation_mode_1": validation_mode_1,
                        "validation_mode_2": validation_mode_2,
                        "G_I": G_I,
                        "G_P": G_P,
                        "R": R,
                        "G_IR": G_IR,
                        "G_Q": G_Q,
                        "C_1": C_1,
                        "G_C_1": G_C_1,
                        "Valid_G_C_1": 0,
                        "E": E,
                        "C_2": C_2,
                        "G_C_2": G_C_2,
                        "Valid_G_C_2": 0,
                        "unsupported_G_C_1": unsupported_1,
                        "unsupported_G_C_2": unsupported_2,
                        "raw_GI_output": raw_GI_output,
                        "raw_error_analysis_output": raw_error_analysis_output,
                        "raw_R_output": raw_R_output,
                        "raw_GQ_output": raw_GQ_output,
                        "raw_C1_output": raw_C1_output,
                        "raw_GC1_output": raw_GC1_output,
                        "raw_C2_output": raw_C2_output,
                        "raw_GC2_output": raw_GC2_output
                    })

    except RuntimeError as stop_error:
        print(stop_error)
        break

    except Exception as e:
        print(f"Unexpected framework error on sample {sample_id}: {e}")

        results.append({
            "id": sample["id"],
            "gold": gold,
            "prediction": "error",
            "raw_prediction": "ERROR",
            "status": "error",
            "refusal_stage": "",
            "final_decision_reason": "Unexpected framework error.",
            "need_R": "",
            "R_reason": "",
            "validation_mode_1": "",
            "validation_mode_2": "",
            "G_I": make_empty_graph(),
            "G_P": make_empty_graph(),
            "R": make_empty_graph(),
            "G_IR": make_empty_graph(),
            "G_Q": make_empty_graph(),
            "C_1": [],
            "G_C_1": make_empty_graph(),
            "Valid_G_C_1": "",
            "E": "",
            "C_2": [],
            "G_C_2": make_empty_graph(),
            "Valid_G_C_2": "",
            "unsupported_G_C_1": [],
            "unsupported_G_C_2": [],
            "raw_GI_output": "",
            "raw_error_analysis_output": "",
            "raw_R_output": "",
            "raw_GQ_output": "",
            "raw_C1_output": "",
            "raw_GC1_output": "",
            "raw_C2_output": "",
            "raw_GC2_output": ""
        })

    save_summary(OUTPUT_FILE, summary)
    completed_ids.add(sample_id)
    print(f"Saved sample {sample_id} | status = {results[-1]['status']}")

save_summary(OUTPUT_FILE, summary)

print("Answer correctness rate:", summary["answer_correctness_rate"])
print("Answer error rate:", summary["answer_error_rate"])
print("Refusal rate:", summary["refusal_rate"])
print(f"Results saved to {OUTPUT_FILE}")