#!/usr/bin/env python3
"""
run_eval.py

Research-ready evaluation runner for SELF-RAG inference only, aligned with the Self-RAG README (7B model).

Usage:
    python run_eval.py \
      --model_name selfrag/selfrag_llama2_7b \
      --n_samples 200 \
      --max_new_tokens 512 \
      --output_dir ./selfrag_eval_outputs

Notes:
 - Requires: vllm, datasets, python-standard libs.
 - Log in to HuggingFace first if datasets require auth (`huggingface-cli login`).
 - For RAGTruth span-level metrics, this script currently does not compute span-level PRF (left as None).
 - EM is improved to account for surface wording differences using normalization, containment, and fuzzy ratio.
 - Follows the Self-RAG README formatting and defaults for vLLM: dtype default is "half" and skip_special_tokens=False.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# ---------------- Dependencies ----------------
try:
    from vllm import LLM, SamplingParams
except Exception as e:
    raise SystemExit(
        "vllm is required but not importable. Install with `pip install vllm` and ensure GPU drivers are set up. Error: {}".format(e)
    )

try:
    from datasets import load_dataset
except Exception as e:
    raise SystemExit("datasets is required. Install with `pip install datasets`. Error: {}".format(e))

# ---------------- Constants & thresholds ----------------
FUZZY_EM_RATIO = 0.92  # SequenceMatcher ratio threshold for fuzzy EM (conservative)
DEFAULT_BATCH_SIZE = 4
# Keep 'half' as the default to follow README, but coerce internally to a valid vLLM dtype.
DEFAULT_DTYPE = "half"

# ---------------- Normalization & scoring utilities ----------------
def normalize_answer(s: Optional[str]) -> str:
    """Normalize text: lower, remove punctuation, articles, extra spaces."""
    if s is None:
        return ""
    s = str(s)
    s = s.lower()
    # remove punctuation (keep word characters and spaces)
    s = re.sub(r"[^\w\s]", " ", s)
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def exact_match_improved(pred: str, gold: str) -> int:
    """Improved EM:
       - normalized equality OR
       - normalized containment (pred in gold or gold in pred) OR
       - fuzzy ratio >= FUZZY_EM_RATIO
    """
    npred = normalize_answer(pred)
    ngold = normalize_answer(gold)
    if not npred and not ngold:
        return 1
    if npred == ngold:
        return 1
    # containment: handles "obama" vs "barack obama"
    if npred in ngold or ngold in npred:
        return 1
    # fuzzy edit-similarity guard (conservative)
    if fuzzy_ratio(npred, ngold) >= FUZZY_EM_RATIO:
        return 1
    return 0

def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (standard)."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0
    common = Counter(p_tokens) & Counter(g_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec = same / len(p_tokens)
    rec = same / len(g_tokens)
    return 2 * prec * rec / (prec + rec)

def lcs_length(a_tokens: List[str], b_tokens: List[str]) -> int:
    """Length of the Longest Common Subsequence between two token lists (O(n*m))."""
    n, m = len(a_tokens), len(b_tokens)
    if n == 0 or m == 0:
        return 0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a_tokens[i - 1]
        for j in range(1, m + 1):
            if ai == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    return dp[n][m]

def rouge_l_f1(pred: str, gold: str) -> float:
    """Compute ROUGE-L F1 using LCS over normalized tokens."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    lcs = lcs_length(p_tokens, g_tokens)
    if lcs == 0:
        return 0.0
    prec = lcs / len(p_tokens)
    rec = lcs / len(g_tokens)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def best_rouge_l_over_golds(pred: str, golds: Optional[List[str]]) -> float:
    if not golds:
        return 0.0
    best = 0.0
    for g in golds:
        score = rouge_l_f1(pred, g)
        if score > best:
            best = score
    return best

def best_em_f1_over_golds(pred: str, golds: Optional[List[str]]) -> Tuple[int, float]:
    """Return best EM and best F1 across multiple gold answers"""
    if not golds:
        return 0, 0.0
    best_em = 0
    best_f1 = 0.0
    for g in golds:
        em = exact_match_improved(pred, g)
        f1 = token_f1(pred, g)
        if em > best_em:
            best_em = em
        if f1 > best_f1:
            best_f1 = f1
    return best_em, best_f1

def inclusion_match(pred: str, golds: Optional[List[str]]) -> int:
    """Return 1 if any normalized gold string is contained in normalized pred."""
    if not golds:
        return 0
    npred = normalize_answer(pred)
    for g in golds:
        if normalize_answer(g) and normalize_answer(g) in npred:
            return 1
    return 0

# ---------------- RAGTruth helpers (response-level & span-level) ----------------
def detect_ragtruth_response_label(example: dict) -> Optional[int]:
    """
    Heuristic: try to extract a binary label for hallucination existence.
    Return 1 if example marked hallucinated (positive), 0 if not hallucinated, None if unknown.
    """
    # common possible fields
    fields = ["is_hallucinated", "hallucinated", "label", "gold_label", "y", "has_hallucination"]
    for f in fields:
        if f in example:
            val = example[f]
            # numeric or boolean
            if isinstance(val, bool):
                return 1 if val else 0
            if isinstance(val, (int, float)):
                return int(bool(val))
            if isinstance(val, str):
                v = val.strip().lower()
                if v in ("yes", "true", "1", "hallucinated", "hallucination", "halluc"):
                    return 1
                if v in ("no", "false", "0", "not enough", "not_hallucinated", "non-hallucinated", "non-halluc"):
                    return 0
    # sometimes stored under nested annotations (try find any 'halluc' token)
    for k, v in example.items():
        if "halluc" in str(k).lower():
            if isinstance(v, bool):
                return 1 if v else 0
            if isinstance(v, (int, float)):
                return int(bool(v))
            if isinstance(v, str):
                vv = v.strip().lower()
                if vv in ("yes", "true", "1"):
                    return 1
                if vv in ("no", "false", "0"):
                    return 0
    return None

def extract_ragtruth_spans(example: dict) -> Optional[List[Tuple[int,int]]]:
    """
    Best-effort: find annotated hallucinated spans. Return list of (start_char, end_char) spans.
    If dataset stores token indices or char spans, try to convert.
    """
    # try common keys
    span_keys = ["hallucinated_spans", "spans", "annotations", "halluc_spans"]
    for k in span_keys:
        if k in example and example[k]:
            val = example[k]
            # If list of dicts with start/end
            if isinstance(val, list):
                spans = []
                for item in val:
                    if isinstance(item, dict):
                        # common keys: start, end, s, e
                        start = item.get("start") or item.get("s") or item.get("char_start")
                        end = item.get("end") or item.get("e") or item.get("char_end")
                        if start is not None and end is not None:
                            try:
                                spans.append((int(start), int(end)))
                            except Exception:
                                pass
                if spans:
                    return spans
            # If list of (start,end) tuples serialized as lists
            if isinstance(val, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in val):
                try:
                    return [(int(x[0]), int(x[1])) for x in val]
                except Exception:
                    pass
    return None

def compute_span_level_prf(pred_text: str, gold_spans: List[Tuple[int,int]]) -> Optional[Tuple[float,float,float]]:
    """
    Placeholder for span-level PRF computation.

    Real span-level evaluation requires alignment between gold spans and prediction text.
    This function currently returns None to indicate 'not computed'.
    """
    return None

# ---------------- Dataset extraction heuristics ----------------
def extract_fields_for_dataset(example: dict, dataset: str) -> Tuple[Optional[str], Optional[List[str]], Optional[str], dict]:
    """
    Return (instruction_text, list_of_gold_answers, optional_context_paragraph, metadata_dict)
    metadata_dict contains raw fields used to help later (e.g., labels for fever, ragtruth info)
    """
    # Generic candidates
    q_candidates = ["question", "query", "claim", "instruction", "input", "query_text", "question_text", "question_body"]
    ans_candidates = ["answers", "answer", "gold_answers", "label", "answer_text", "target_text"]
    context_candidates = ["contexts", "context", "retrieved_docs", "paragraph", "evidence", "document", "context_text", "passage"]
    meta = {}
    instruction = None

    # direct key matches preferred
    for k in q_candidates:
        if k in example and example[k]:
            instruction = example[k]
            break

    # dataset-specific fallbacks
    if instruction is None:
        if dataset.startswith("mwong/fever"):
            if "claim" in example:
                instruction = example["claim"]
        if dataset.startswith("mandarjoshi/trivia_qa"):
            if "question" in example:
                instruction = example["question"]
        if dataset.startswith("microsoft/ms_marco"):
            if "query" in example:
                instruction = example["query"]
    # Answers extraction
    answers = None
    if "answers" in example:
        a = example["answers"]
        if isinstance(a, dict):
            # many HF qa formats: {'text': [...]} or {'answer': ...}
            if "text" in a:
                answers = a["text"] if isinstance(a["text"], list) else [a["text"]]
            elif "answer" in a:
                answers = a["answer"] if isinstance(a["answer"], list) else [a["answer"]]
        elif isinstance(a, list):
            answers = a
        elif isinstance(a, str):
            answers = [a]
    else:
        for k in ans_candidates:
            if k in example and example[k]:
                v = example[k]
                if isinstance(v, list):
                    answers = v
                elif isinstance(v, dict):
                    if "text" in v and isinstance(v["text"], list):
                        answers = v["text"]
                    else:
                        answers = [str(x) for x in v.values() if x]
                else:
                    answers = [str(v)]
                break

    # context
    context = None
    for k in context_candidates:
        if k in example and example[k]:
            context = example[k]
            break

    # normalize answers to list[str] or None
    if answers is not None:
        clean = []
        for a in answers:
            if a is None:
                continue
            if isinstance(a, (list, dict)):
                clean.append(str(a))
            else:
                clean.append(str(a))
        answers = clean if clean else None

    # metadata for FEVER labels and RAGTruth
    if "label" in example:
        meta["label"] = example["label"]
    if "gold_label" in example:
        meta["gold_label"] = example["gold_label"]
    # RAGTruth detection fields
    meta["ragtruth_label"] = detect_ragtruth_response_label(example)
    meta["ragtruth_spans"] = extract_ragtruth_spans(example)

    return (str(instruction) if instruction else None, answers, (str(context) if context else None), meta)

# ---------------- vllm helpers ----------------
def get_text_from_vllm_result(pred_obj) -> str:
    """Safely extract text from a vllm result item."""
    try:
        # vllm generate returns objects with .outputs; each output is a list of result alternatives with .text
        if hasattr(pred_obj, "outputs"):
            out0 = pred_obj.outputs[0]
            if hasattr(out0, "text"):
                return out0.text.strip()
            return str(out0).strip()
        # fallback
        return str(pred_obj).strip()
    except Exception:
        try:
            return str(pred_obj).strip()
        except Exception:
            return ""

def batched_generate(model: LLM, prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    """
    Generate for a list of prompts using vllm. Returns list of strings (model outputs).
    If batch generation fails, fallback to single-call generation per prompt.
    """
    if not prompts:
        return []
    # Try batched generation
    try:
        preds = model.generate(prompts, sampling_params)
        # model.generate returns a list-like object; extract text for each
        texts = []
        for p in preds:
            texts.append(get_text_from_vllm_result(p))
        return texts
    except Exception as e:
        print("vllm batched generation failed (will fallback to single-call loop). Error:", e, file=sys.stderr)
        texts = []
        for p in prompts:
            try:
                single = model.generate([p], sampling_params)[0]
                texts.append(get_text_from_vllm_result(single))
            except Exception as e2:
                print("vllm single-call generation failed for one prompt:", e2, file=sys.stderr)
                texts.append("")
        return texts

# ---------------- Core evaluation per dataset (batched) ----------------
def _find_label_from_example(ex: dict, meta: dict) -> Optional[str]:
    """Look for FEVER label in meta and the raw example under many common keys."""
    # priority: meta if present
    candidates = []
    if isinstance(meta, dict):
        candidates.extend(["label", "gold_label", "ragtruth_label"])
        for k in ["label", "gold_label", "verifiable_label", "veracity", "verdict", "claim_label"]:
            if k in meta:
                candidates.append(k)
    # also check the raw example
    candidates.extend(["label", "gold_label", "verifiable_label", "veracity", "verdict", "claim_label", "verifiable"])
    # finally any key that looks promising
    for k in list(ex.keys()):
        if k.lower() in ("label", "gold_label", "verifiable_label", "veracity", "verdict", "claim_label"):
            candidates.append(k)
    # dedupe preserve order
    seen = set()
    final_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            final_candidates.append(c)
    # check them
    for k in final_candidates:
        if isinstance(meta, dict) and k in meta and meta[k] is not None:
            return meta[k]
        if k in ex and ex[k] is not None:
            return ex[k]
    # fallback: some datasets store label under 'annotations' or nested dicts
    for k, v in ex.items():
        if isinstance(k, str) and "label" in k.lower():
            return v
    return None

def evaluate_fever(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int):
    """FEVER: treat as classification (accuracy). The dataset may have claims and labels like SUPPORTS/REFUTES/NOT ENOUGH INFO."""
    print("Evaluating FEVER (accuracy)")
    split = choose_split(ds)
    data = ds[split]
    total = len(data)
    n = min(n_samples, total)
    indices = sample_indices(total, n)
    outputs = []
    correct = 0
    latencies = []
    label_mode_used = None  # '3way' or 'binary'

    # batch processing
    for i in range(0, len(indices), batch_size):
        batch_idxs = indices[i : i + batch_size]
        actual_prompts = []
        prompt_meta = []  # (idx, ex, meta, instruction)

        # build prompts for this batch
        for idx in batch_idxs:
            ex = data[int(idx)]
            instruction, answers, context, meta = extract_fields_for_dataset(ex, "mwong/fever-evidence-related")
            if instruction is None:
                outputs.append({"index": idx, "skipped": True, "reason": "no-claim"})
                continue
            prompt = format_prompt(f"Verify this claim: {instruction}\nAnswer with one of: SUPPORTS, REFUTES, NOT ENOUGH INFO.")
            actual_prompts.append(prompt)
            prompt_meta.append((idx, ex, meta, instruction))

        if not actual_prompts:
            continue

        t0 = time.perf_counter()
        texts = batched_generate(model, actual_prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(actual_prompts))
        latencies.extend([per_latency] * len(actual_prompts))

        for (idx, ex, meta, instruction), text in zip(prompt_meta, texts):
            mapped = map_to_fever_label(text)
            correct_flag = 0
            gold_label = _find_label_from_example(ex, meta)
            binary_gold = None
            mode = None
            # Determine if the found gold label is numeric/binary
            def as_binary(val):
                try:
                    if isinstance(val, bool):
                        return int(val)
                    if isinstance(val, (int, float)):
                        return int(val)
                    if isinstance(val, str) and val.strip() in {"0", "1"}:
                        return int(val.strip())
                except Exception:
                    pass
                return None

            if gold_label is not None:
                # If it's numeric-like, use binary comparison
                maybe_bin = as_binary(gold_label)
                if maybe_bin is not None:
                    binary_gold = maybe_bin
                    pred_binary = 1 if mapped == "SUPPORTS" else 0
                    if pred_binary == binary_gold:
                        correct_flag = 1
                    mode = "binary"
                else:
                    # 3-way textual gold label
                    try:
                        gold_label_str = str(gold_label)
                    except Exception:
                        gold_label_str = None
                    if gold_label_str and normalize_label_compare(mapped, gold_label_str):
                        correct_flag = 1
                    mode = "3way"
            else:
                # Try dataset-specific numeric label (0/1) under 'labels'
                if isinstance(ex, dict) and "labels" in ex:
                    binary_gold = as_binary(ex.get("labels"))
                    if binary_gold is not None:
                        pred_binary = 1 if mapped == "SUPPORTS" else 0
                        if pred_binary == binary_gold:
                            correct_flag = 1
                        mode = "binary"
            if mode and (label_mode_used is None):
                label_mode_used = mode
            outputs.append({
                "index": idx,
                "instruction": instruction,
                "model_raw": text,
                "pred_label": mapped,
                "gold_label": gold_label,
                "binary_gold": binary_gold,
                "mode_used": mode,
                "correct": correct_flag,
                "latency_s": per_latency
            })
            correct += correct_flag

    # Only count examples where a gold label exists
    evaluated = len([o for o in outputs if (not o.get("skipped", False)) and ((o.get("gold_label") is not None) or (o.get("binary_gold") is not None))])
    accuracy = correct / evaluated if evaluated else None
    out_file = out_dir / "fever_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for item in outputs:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {
        "dataset": "mwong/fever-evidence-related",
        "split": split,
        "examples_evaluated": evaluated,
        "accuracy": accuracy,
        "fever_label_mode": label_mode_used,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "output_file": str(out_file)
    }

def map_to_fever_label(text: str) -> str:
    """Map arbitrary model output to one of SUPPORTS / REFUTES / NOT ENOUGH INFO (NEI first; conservative, word-boundary based)."""
    t = normalize_answer(text)

    # Handle explicit 'not support' forms as REFUTES early
    if re.search(r"\b(not|no)\s+(support|supported|supports)\b", t):
        return "REFUTES"

    # NOT ENOUGH INFO (NEI) first
    if (
        "not enough" in t
        or "insufficient" in t
        or "no evidence" in t
        or "unknown" in t
        or "cannot determine" in t
        or "unable to determine" in t
        or "cannot tell" in t
        or "indeterminate" in t
        or re.search(r"\bnei\b", t) is not None
    ):
        return "NOT ENOUGH INFO"

    # REFUTES
    if (
        re.search(r"\brefut\w*\b", t) is not None
        or re.search(r"\bcontradict\w*\b", t) is not None
        or re.search(r"\bdisprov\w*\b", t) is not None
        or re.search(r"(?<!not\s)\bfalse\b", t) is not None
        or re.search(r"\bdoes\s+not\s+(hold|follow)\b", t) is not None
    ):
        return "REFUTES"

    # SUPPORTS
    if (
        re.search(r"(?<!not\s)\b(support|supports|supported|entail|entails|entailed)\b", t) is not None
        or re.search(r"(?<!not\s)\b(true|yes)\b", t) is not None
    ):
        return "SUPPORTS"

    # Default to NEI
    return "NOT ENOUGH INFO"

def normalize_label_compare(pred_label: Optional[str], gold_label: Optional[str]) -> bool:
    if pred_label is None or gold_label is None:
        return False

    def canonize(x: str) -> str:
        x = normalize_answer(x)
        # Map synonyms and variants to canonical set
        mapping = {
            "supported": "supports",
            "support": "supports",
            "entail": "supports",
            "entails": "supports",
            "entailed": "supports",
            "refuted": "refutes",
            "refute": "refutes",
            "contradiction": "refutes",
            "contradicts": "refutes",
            "nei": "not enough info",
        }
        return mapping.get(x, x)

    pl = canonize(pred_label)
    gl = canonize(gold_label)
    # Golds are commonly "SUPPORTS"/"REFUTES"/"NOT ENOUGH INFO"
    return pl == gl

def evaluate_generic_qa(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int, dataset_name: str, metric_mode: str):
    """
    Generic QA evaluation for MSMARCO, HotPotQA, TriviaQA, Natural Questions.
    metric_mode: 'em_f1' (EM+F1), 'inclusion' (match/inclusion)
    """
    print(f"Evaluating {dataset_name} with mode {metric_mode}")
    split = choose_split(ds)
    data = ds[split]
    total = len(data)
    n = min(n_samples, total)
    indices = sample_indices(total, n)
    outputs = []
    em_sum = 0
    f1_sum = 0.0
    rougeL_sum = 0.0
    inclusion_sum = 0
    latencies = []

    # process in batches
    for i in range(0, len(indices), batch_size):
        batch_idxs = indices[i : i + batch_size]
        actual_prompts = []
        meta_list = []  # tuples (idx, answers, instruction)

        for idx in batch_idxs:
            ex = data[int(idx)]
            instruction, answers, context, meta = extract_fields_for_dataset(ex, dataset_name)
            if instruction is None:
                outputs.append({"index": idx, "skipped": True, "reason": "no-question"})
                continue
            actual_prompts.append(format_prompt(instruction, paragraph=context))
            meta_list.append((idx, answers, instruction))

        if not actual_prompts:
            continue

        t0 = time.perf_counter()
        texts = batched_generate(model, actual_prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(actual_prompts))
        latencies.extend([per_latency] * len(actual_prompts))

        for (idx, answers, instruction), text in zip(meta_list, texts):
            if metric_mode == "em_f1":
                em, f1 = best_em_f1_over_golds(text, answers)
                em_sum += em
                f1_sum += f1
                # Add ROUGE-L for MS MARCO (common scoring)
                if "microsoft/ms_marco" in dataset_name or "msmarco" in dataset_name:
                    rougeL = best_rouge_l_over_golds(text, answers)
                    rougeL_sum += rougeL
                else:
                    rougeL = None
                outputs.append({
                    "index": idx,
                    "instruction": instruction,
                    "prediction": text,
                    "gold": answers,
                    "em": em,
                    "f1": f1,
                    **({"rougeL": rougeL} if rougeL is not None else {}),
                    "latency_s": per_latency
                })
            elif metric_mode == "inclusion":
                inc = inclusion_match(text, answers)
                inclusion_sum += inc
                outputs.append({
                    "index": idx,
                    "instruction": instruction,
                    "prediction": text,
                    "gold": answers,
                    "inclusion": inc,
                    "latency_s": per_latency
                })
            else:
                outputs.append({
                    "index": idx,
                    "instruction": instruction,
                    "prediction": text,
                    "gold": answers,
                    "latency_s": per_latency
                })

    # Only count examples that have any non-empty gold answers for metric calculations
    def has_gold(o):
        g = o.get("gold")
        return isinstance(g, list) and any((isinstance(x, str) and x.strip()) for x in g)
    evaled = len([o for o in outputs if (not o.get("skipped", False)) and has_gold(o)])
    summary = {
        "dataset": dataset_name,
        "split": split,
        "examples_evaluated": evaled,
        "output_file": str(out_dir / f"{safe_filename(dataset_name)}_outputs.jsonl"),
    }
    if metric_mode == "em_f1":
        summary["em"] = (em_sum / evaled) if evaled else None
        summary["f1"] = (f1_sum / evaled) if evaled else None
        if evaled and ("microsoft/ms_marco" in dataset_name or "msmarco" in dataset_name):
            summary["rougeL"] = (rougeL_sum / evaled)
        summary["avg_latency_s"] = (sum(latencies) / len(latencies)) if latencies else None
    elif metric_mode == "inclusion":
        summary["inclusion_rate"] = (inclusion_sum / evaled) if evaled else None
        summary["avg_latency_s"] = (sum(latencies) / len(latencies)) if latencies else None

    out_file = out_dir / f"{safe_filename(dataset_name)}_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for it in outputs:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return summary

def _parse_ragtruth_gold_label(ex: dict) -> Optional[int]:
    """Parse RAGTruth gold: 1 if any hallucination labels present, else 0. Returns None if unknown."""
    if not isinstance(ex, dict):
        return None
    # Candidates: 'hallucination_labels', 'hallucination_labels_processed'
    for k in ["hallucination_labels", "hallucination_labels_processed"]:
        if k in ex and ex[k] is not None:
            v = ex[k]
            try:
                # Some fields are JSON strings
                if isinstance(v, str):
                    v_parsed = json.loads(v)
                else:
                    v_parsed = v
                if isinstance(v_parsed, list):
                    return 1 if len(v_parsed) > 0 else 0
                # If dict with spans, consider any key presence
                if isinstance(v_parsed, dict):
                    return 1 if len(v_parsed) > 0 else 0
            except Exception:
                # If parsing fails, treat non-empty string as positive signal
                if isinstance(v, str) and v.strip():
                    # Heuristic: if it looks like a JSON list and contains '{' assume positive
                    return 1 if "{" in v or "[" in v else 0
    return None

def evaluate_ragtruth(ds, model, sampling_params, out_dir: Path, n_samples: int, batch_size: int):
    """
    RAGTruth: compute response-level detection P/R/F1, plus try span-level if annotations exist.
    We'll look for a label field indicating hallucinated vs not.
    """
    print("Evaluating RAGTruth (response-level + optional span-level)")
    split = choose_split(ds)
    data = ds[split]
    total = len(data)
    n = min(n_samples, total)
    indices = sample_indices(total, n)
    outputs = []
    tp = fp = fn = 0
    latencies = []
    span_prf_accum: List[Tuple[float, float, float]] = []

    # process batches
    for i in range(0, len(indices), batch_size):
        batch_idxs = indices[i : i + batch_size]
        actual_prompts = []
        meta_list = []  # (idx, ex, meta, instruction)

        for idx in batch_idxs:
            ex = data[int(idx)]
            instruction, answers, context, meta = extract_fields_for_dataset(ex, "wandb/RAGTruth-processed")
            if instruction is None and "query" in ex:
                instruction = ex.get("query")
            if instruction is None:
                outputs.append({"index": idx, "skipped": True, "reason": "no-query"})
                continue
            actual_prompts.append(format_prompt(instruction, paragraph=context))
            meta_list.append((idx, ex, meta, instruction))

        if not actual_prompts:
            continue

        t0 = time.perf_counter()
        texts = batched_generate(model, actual_prompts, sampling_params)
        t1 = time.perf_counter()
        per_latency = (t1 - t0) / max(1, len(actual_prompts))
        latencies.extend([per_latency] * len(actual_prompts))

        for (idx, ex, meta, instruction), text in zip(meta_list, texts):
            # robust gold parse from dataset-specific fields
            gold_binary: Optional[int] = _parse_ragtruth_gold_label(ex)
            gold_label = gold_binary  # store for output readability
            # naive predicted detection heuristic:
            lower_txt = text.lower()
            if any(phrase in lower_txt for phrase in ["i don't know", "i do not know", "cannot determine", "no retrieval", "no evidence", "i'm not sure", "not enough information", "i cannot find"]):
                pred_halluc = 0
            else:
                pred_halluc = 1
            # update confusion
            if gold_binary is not None:
                if pred_halluc == 1 and gold_binary == 1:
                    tp += 1
                elif pred_halluc == 1 and gold_binary == 0:
                    fp += 1
                elif pred_halluc == 0 and gold_binary == 1:
                    fn += 1
            outputs.append({
                "index": idx,
                "instruction": instruction,
                "prediction": text,
                "gold_label": gold_label,
                "pred_halluc": pred_halluc,
                "latency_s": per_latency,
                "ragtruth_spans": (meta.get("ragtruth_spans") if isinstance(meta, dict) else None)
            })
            if isinstance(meta, dict) and meta.get("ragtruth_spans"):
                prf = compute_span_level_prf(text, meta.get("ragtruth_spans"))
                if prf is not None:
                    span_prf_accum.append(prf)

    evaled = len([o for o in outputs if not o.get("skipped", False)])
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = None

    span_prf_avg = None
    span_prf_computed = False
    if span_prf_accum:
        ps = [p for p, r, f in span_prf_accum]
        rs = [r for p, r, f in span_prf_accum]
        fs = [f for p, r, f in span_prf_accum]
        span_prf_avg = {"p": sum(ps) / len(ps), "r": sum(rs) / len(rs), "f1": sum(fs) / len(fs)}
        span_prf_computed = True

    out_file = out_dir / "ragtruth_outputs.jsonl"
    with open(out_file, "w", encoding="utf-8") as fh:
        for it in outputs:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    return {
        "dataset": "wandb/RAGTruth-processed",
        "split": split,
        "examples_evaluated": evaled,
        "response_precision": precision,
        "response_recall": recall,
        "response_f1": f1,
        "span_prf_avg": span_prf_avg,
        "span_prf_computed": span_prf_computed,
        "avg_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "output_file": str(out_file)
    }

# ---------------- Helpers ----------------
def sample_indices(total: int, n: int, seed: int = 42) -> List[int]:
    random.seed(seed)
    idxs = list(range(total))
    random.shuffle(idxs)
    return idxs[:n]

def format_prompt(instruction: str, paragraph: Optional[str] = None) -> str:
    # Follows the Self-RAG README prompt format.
    p = "### Instruction:\n{0}\n\n### Response:\n".format(instruction)
    if paragraph:
        p += "[Retrieval]<paragraph>{0}</paragraph>".format(paragraph)
    return p

def choose_split(raw_ds) -> str:
    """Pick a split, preferring labeled splits over test.

    Many HF datasets hide labels in the test split. To avoid computing
    misleading zeros, we de-prioritize 'test' and prefer validation/dev/train.
    """
    for s in ["validation", "dev", "train", "validation_matched", "validation_unmatched", "test"]:
        if s in raw_ds:
            return s
    # default to first available
    return list(raw_ds.keys())[0]

def safe_filename(name: str) -> str:
    """Create a filesystem-safe filename from dataset name."""
    return re.sub(r"[^\w\-_\.]", "_", name)

def coerce_dtype_for_vllm(dtype_arg: str) -> str:
    """
    Accept README-style dtype aliases while ensuring compatibility with vLLM.
    - "half" or "fp16" -> "float16"
    - "bf16" -> "bfloat16"
    - pass-through: "auto", "float16", "bfloat16", "float32"
    """
    x = (dtype_arg or "").strip().lower()
    if x in ("half", "fp16"):
        return "float16"
    if x in ("bf16",):
        return "bfloat16"
    if x in ("auto", "float16", "bfloat16", "float32"):
        return x
    # Fall back to float16 with a warning
    print(f"Warning: dtype '{dtype_arg}' not recognized. Falling back to float16.", file=sys.stderr)
    return "float16"

# ---------------- Main CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Research-grade SELF-RAG evaluation (inference only).")
    parser.add_argument("--model_name", type=str, default="selfrag/selfrag_llama2_7b", help="HuggingFace model name for SELF-RAG")
    parser.add_argument("--download_dir", type=str, default=None, help="Optional model download cache directory")
    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples per dataset (max)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens per sample")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for generation")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, help="dtype for vllm (README default 'half'; accepted: half/fp16, bf16, auto, float16, bfloat16, float32)")
    parser.add_argument("--output_dir", type=str, default="./selfrag_eval_outputs", help="Directory to save outputs and summary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # instantiate vllm model
    llm_kwargs: Dict[str, Any] = {}
    if args.download_dir:
        llm_kwargs["download_dir"] = args.download_dir

    coerced_dtype = coerce_dtype_for_vllm(args.dtype)
    if coerced_dtype != (args.dtype or "").strip().lower():
        print(f"Coercing dtype '{args.dtype}' to '{coerced_dtype}' for vLLM compatibility.")

    print(f"Loading model {args.model_name} with dtype={args.dtype} (vLLM uses: {coerced_dtype}) ...")
    model = LLM(args.model_name, dtype=coerced_dtype, **llm_kwargs)
    # Follow README: skip_special_tokens=False to preserve Self-RAG reflection tokens in outputs.
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, skip_special_tokens=False)

    # Datasets list (keep names the same)
    datasets_to_run = [
        ("mwong/fever-evidence-related", "fever"),
        ("microsoft/ms_marco", "msmarco_v2.1"),
        ("hotpotqa/hotpot_qa", "hotpot_distractor_and_fullwiki"),
        ("wandb/RAGTruth-processed", "ragtruth"),
        ("mandarjoshi/trivia_qa", "trivia_rc"),
        ("sentence-transformers/natural-questions", "natural_questions")
    ]

    summary_list: List[Dict[str, Any]] = []

    for ds_id, tag in datasets_to_run:
        try:
            if ds_id == "hotpotqa/hotpot_qa":
                # evaluate both variants: distractor and fullwiki
                for cfg in ["distractor", "fullwiki"]:
                    print(f"\n--- Loading HotPotQA config: {cfg}")
                    raw = load_dataset(ds_id, cfg)
                    summ = evaluate_generic_qa(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size, dataset_name=f"{ds_id}__{cfg}", metric_mode="em_f1")
                    if summ:
                        summary_list.append(summ)
                continue

            if ds_id == "microsoft/ms_marco":
                print("\n--- Loading MS MARCO v2.1")
                raw = load_dataset(ds_id, "v2.1")
                summ = evaluate_generic_qa(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size, dataset_name=f"{ds_id}::v2.1", metric_mode="em_f1")
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "mwong/fever-evidence-related":
                print("\n--- Loading FEVER")
                raw = load_dataset(ds_id)
                summ = evaluate_fever(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size)
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "wandb/RAGTruth-processed":
                print("\n--- Loading RAGTruth")
                raw = load_dataset(ds_id)
                summ = evaluate_ragtruth(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size)
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "mandarjoshi/trivia_qa":
                print("\n--- Loading TriviaQA (rc)")
                raw = load_dataset(ds_id, "rc")
                summ = evaluate_generic_qa(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size, dataset_name=f"{ds_id}::rc", metric_mode="inclusion")
                if summ:
                    summary_list.append(summ)
                continue

            if ds_id == "sentence-transformers/natural-questions":
                print("\n--- Loading Natural Questions")
                raw = load_dataset(ds_id)
                summ = evaluate_generic_qa(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size, dataset_name=ds_id, metric_mode="em_f1")
                if summ:
                    summary_list.append(summ)
                continue

            # default fallback
            raw = load_dataset(ds_id)
            summ = evaluate_generic_qa(raw, model, sampling_params, out_dir, args.n_samples, args.batch_size, dataset_name=ds_id, metric_mode="em_f1")
            if summ:
                summary_list.append(summ)
        except Exception as e:
            print(f"Failed to evaluate {ds_id}: {e}", file=sys.stderr)

    # Save overall summary
    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary_list, fh, indent=2)

    # Save CSV summary (flatten)
    summary_csv = out_dir / "summary.csv"
    # collect union of possible keys
    keys = set()
    for s in summary_list:
        keys.update(s.keys())
    keys = sorted(list(keys))
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for s in summary_list:
            writer.writerow(s)

    print("Evaluation complete. Outputs and summary saved to:", out_dir.resolve())

if __name__ == "__main__":
    main()
