from vllm import LLM, SamplingParams
import json, time, os, sys, logging, re, string
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np
from datasets import load_dataset
from datasets.utils.file_utils import DownloadConfig
import time, random
from datasets import load_dataset
import itertools
import atexit, datetime
import torch.distributed as dist


def load_dataset_retry(*args, retries=5, base_sleep=2.0, jitter=0.75, **kwargs):
    """
    Retry wrapper for HF load_dataset with exponential backoff + jitter.
    Works with both normal and streaming modes.
    """
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                raise
            sleep = (base_sleep ** (attempt - 1)) + random.uniform(0.0, jitter)
            logger.warning(
                f"load_dataset failed (attempt {attempt}/{retries}): {e}. "
                f"Retrying in {sleep:.1f}s"
            )
            time.sleep(sleep)

# Optional metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except Exception:
    print("Warning: rouge_score not available. pip install rouge-score")
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except Exception:
    print("Warning: bert_score not available. pip install bert_score")
    BERTSCORE_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("selfrag_eval")

# Global download config (helps with transient 50x like your 504)
DC = DownloadConfig(max_retries=5)

# ----------------------- Model & Evaluator -----------------------

class SelfRAGModel:
    def __init__(self,
                 model_path: str = "selfrag/selfrag_llama2_7b",
                 download_dir: str = "/gscratch/h2lab/akari/model_cache",
                 dtype: str = "half"):
        self.model = LLM(model_path, download_dir=download_dir, dtype=dtype)
        # FIXED: Increased max_tokens from 512 to 1024 for better responses and scoring
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=2048, skip_special_tokens=False
        )

    def format_prompt(self, input_text, paragraph=None):
        prompt = f"### Instruction:\n{input_text}\n\n### Response:\n"
        if paragraph:
            prompt += f"[Retrieval]<paragraph>{paragraph}</paragraph>"
        return prompt

    def extract_utility_score(self, text: str) -> int:
        for i in range(5, 0, -1):
            if f"[Utility:{i}]" in text:
                return i
        return 0

    def extract_relevance(self, text: str) -> bool:
        return "[Relevant]" in text

    def extract_support(self, text: str) -> str:
        if "[Fully supported]" in text:
            return "fully_supported"
        if "[Partially supported]" in text:
            return "partially_supported"
        if "[No support / Contradictory]" in text:
            return "no_support"
        return "unknown"

    def uses_retrieval(self, text: str) -> bool:
        return "[Retrieve]" in text

    def extract_final_answer(self, text: str) -> str:
        """
        Extract the main answer from SelfRAG response, removing special tokens
        """
        # Remove SelfRAG special tokens
        cleaned = re.sub(r'\[.*?\]', '', text)
        # Remove paragraph markers
        cleaned = re.sub(r'<paragraph>.*?</paragraph>', '', cleaned, flags=re.DOTALL)
        # Clean up whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

class SelfRAGEvaluator:
    def __init__(self):
        self.rouge_scorer = (
            rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
            if ROUGE_AVAILABLE else None
        )

    def normalize_answer(self, s: str) -> str:
        def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t, flags=re.I)
        def white_space_fix(t): return " ".join(t.split())
        def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    def exact_match_score(self, pred, gt) -> float:
        return float(self.normalize_answer(pred) == self.normalize_answer(gt))

    def f1_score(self, pred, gt) -> float:
        p = self.normalize_answer(pred).split()
        g = self.normalize_answer(gt).split()
        if not p and not g: return 1.0
        if not p or not g: return 0.0
        common = Counter(p) & Counter(g)
        num_same = sum(common.values())
        if num_same == 0: return 0.0
        precision = num_same / len(p)
        recall = num_same / len(g)
        return 2 * precision * recall / (precision + recall)

    def evaluate_multiple_answers(self, prediction, ground_truths):
        if not ground_truths: return {'em': 0.0, 'f1': 0.0}
        best_em = 0.0; best_f1 = 0.0
        for gt in ground_truths:
            if not (gt and gt.strip()): continue
            best_em = max(best_em, self.exact_match_score(prediction, gt))
            best_f1 = max(best_f1, self.f1_score(prediction, gt))
        return {'em': best_em, 'f1': best_f1}

evaluator = SelfRAGEvaluator()

# Safe generation wrapper
def safe_generate(model: SelfRAGModel, prompt: str):
    out = model.model.generate([prompt], model.sampling_params)[0]
    if not getattr(out, "outputs", None):
        return "", 0
    first = out.outputs[0]
    # IMPROVED: Extract clean answer for better scoring
    raw_response = first.text or ""
    clean_response = model.extract_final_answer(raw_response)
    return clean_response, len(first.token_ids or [])

# ----------------------- Benchmarks -----------------------

def run_natural_questions_benchmark(model, sample_size: int = 200, streaming=False):
    """
    Use nq_open (stable schema): {'question': str, 'answers': list[str]}
    """
    logger.info(f"Running NQ-Open with sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("google-research-datasets/natural_questions", "default", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("google-research-datasets/natural_questions", "default", split="validation", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))

        results = []
        for i, item in enumerate(ds):
            try:
                question = item.get("question", "")
                answer_texts = [a for a in (item.get("answers") or []) if a]
                prompt = model.format_prompt(question)  # no built-in context
                t0 = time.time()
                resp, tok_count = safe_generate(model, prompt)
                dt = time.time() - t0
                scores = evaluator.evaluate_multiple_answers(resp, answer_texts) if answer_texts else {'em':0.0,'f1':0.0}
                results.append({
                    'dataset':'nq_open','question':question,'response':resp,
                    'ground_truth_answers':answer_texts,'exact_match':scores['em'],'f1_score':scores['f1'],
                    'inference_time':dt,'tokens_generated':tok_count,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp)
                })
                if (i+1)%10==0: logger.info(f"NQ processed {i+1}/{len(ds) if not streaming else sample_size}")
            except Exception as e:
                logger.error(f"NQ item {i} error: {e}", exc_info=True)
        logger.info(f"NQ-Open completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running NQ-Open: {e}", exc_info=True)
        return []

def run_trivia_qa_benchmark(model, sample_size: int = 200, streaming=False):
    """
    TriviaQA rc: {'question': str, 'context': str, 'answer': {'value', 'aliases'}}
    """
    logger.info(f"Running TriviaQA(rc) sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("mandarjoshi/trivia_qa", "rc", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("mandarjoshi/trivia_qa", "rc", split="validation", download_config=DC)
            if sample_size < len(ds): ds = ds.select(range(sample_size))

        results=[]
        for i,item in enumerate(ds):
            try:
                question = item.get("question","")
                context_text = item.get("context","") or ""
                ans = item.get("answer", {}) or {}
                answer_texts = []
                if ans.get("value"): answer_texts.append(ans["value"])
                answer_texts += [a for a in (ans.get("aliases") or []) if a]

                prompt = model.format_prompt(question, context_text if context_text.strip() else None)
                t0=time.time(); resp, tok = safe_generate(model, prompt); dt=time.time()-t0
                scores = evaluator.evaluate_multiple_answers(resp, answer_texts) if answer_texts else {'em':0.0,'f1':0.0}

                results.append({
                    'dataset':'trivia_qa','question':question,'response':resp,
                    'ground_truth_answers':answer_texts,'exact_match':scores['em'],'f1_score':scores['f1'],
                    'inference_time':dt,'tokens_generated':tok,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp),
                    'has_context': bool(context_text)
                })
                if (i+1)%10==0: logger.info(f"TriviaQA processed {i+1}/{len(ds) if not streaming else sample_size}")
            except Exception as e:
                logger.error(f"TriviaQA item {i} error: {e}", exc_info=True)
        logger.info(f"TriviaQA completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running TriviaQA: {e}", exc_info=True)
        return []

def run_hotpot_qa_benchmark(model, sample_size: int = 200, streaming=False):
    logger.info(f"Running HotpotQA(distractor) sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("hotpotqa/hotpot_qa","distractor", split="validation", download_config=DC)
            if sample_size < len(ds): ds = ds.select(range(sample_size))

        results=[]
        for i,item in enumerate(ds):
            try:
                question = item.get("question","")
                answer = item.get("answer","") or ""
                level = item.get("level","unknown")
                qtype = item.get("type","unknown")

                # context: list of [title, [sentences...]]
                context_texts=[]
                for pair in item.get("context", []):
                    if isinstance(pair,(list,tuple)) and len(pair)==2:
                        title, sentences = pair
                        if sentences:
                            context_texts.append(f"{title}: {' '.join(sentences)}")
                context_text = "\n".join(context_texts[:5])

                prompt = model.format_prompt(question, context_text if context_text.strip() else None)
                t0=time.time(); resp, tok = safe_generate(model, prompt); dt=time.time()-t0
                scores = evaluator.evaluate_multiple_answers(resp, [answer]) if answer else {'em':0.0,'f1':0.0}

                results.append({
                    'dataset':'hotpot_qa','question':question,'response':resp,'ground_truth_answer':answer,
                    'level':level,'type':qtype,'exact_match':scores['em'],'f1_score':scores['f1'],
                    'inference_time':dt,'tokens_generated':tok,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp),
                    'num_context_paragraphs': len(context_texts)
                })
                if (i+1)%10==0: logger.info(f"HotpotQA processed {i+1}/{len(ds) if not streaming else sample_size}")
            except Exception as e:
                logger.error(f"HotpotQA item {i} error: {e}", exc_info=True)
        logger.info(f"HotpotQA completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running HotpotQA: {e}", exc_info=True)
        return []

def run_squad_v2_benchmark(model, sample_size: int = 200, streaming=False):
    logger.info(f"Running SQuAD v2 sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("rajpurkar/squad_v2", split="validation", streaming=True)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("rajpurkar/squad_v2", split="validation", download_config=DC)
            if sample_size < len(ds): ds = ds.select(range(sample_size))

        results=[]
        for i,item in enumerate(ds):
            try:
                question = item.get("question","")
                context = item.get("context","") or ""
                answers = item.get("answers", {}) or {}
                answer_texts = [a for a in (answers.get("text") or []) if a]
                is_impossible = (len(answer_texts)==0)

                prompt = model.format_prompt(question, context if context.strip() else None)
                t0=time.time(); resp, tok = safe_generate(model, prompt); dt=time.time()-t0

                if not is_impossible and answer_texts:
                    scores = evaluator.evaluate_multiple_answers(resp, answer_texts)
                else:
                    no_ans = ["no answer","cannot answer","not provided","unknown","unanswerable"]
                    detected = any(ind in (resp.lower() if resp else "") for ind in no_ans)
                    scores = {'em': 1.0 if detected else 0.0, 'f1': 1.0 if detected else 0.0}

                results.append({
                    'dataset':'squad_v2','question':question,'response':resp,
                    'ground_truth_answers':answer_texts,'is_impossible':is_impossible,
                    'exact_match':scores['em'],'f1_score':scores['f1'],
                    'inference_time':dt,'tokens_generated':tok,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp)
                })
                if (i+1)%10==0: logger.info(f"SQuAD v2 processed {i+1}/{len(ds) if not streaming else sample_size}")
            except Exception as e:
                logger.error(f"SQuAD v2 item {i} error: {e}", exc_info=True)
        logger.info(f"SQuAD v2 completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running SQuAD v2: {e}", exc_info=True)
        return []

def run_fever_benchmark(model, sample_size: int = 200, streaming: bool = False):
    """
    FIXED: FEVER (Fact Extraction and VERification) benchmark
    """
    logger.info(f"Running FEVER sample_size={sample_size} (streaming={streaming})")
    try:
        # Use validation split which has proper labels
        if streaming:
            ds_iter = load_dataset_retry("mwong/fever-evidence-related", split="paper_dev", streaming=True, download_config=DC)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("mwong/fever-evidence-related", split="paper_dev", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Failed to load FEVER: {e}", exc_info=True)
        return []

    results = []
    for i, item in enumerate(ds):
        try:
            claim = item.get("claim", "") or ""
            label = item.get("label", "") or ""   # 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'
            evidence = item.get("evidence", None)

            # Extract evidence text if available
            context_text = ""
            if evidence and isinstance(evidence, list):
                evidence_texts = []
                for ev_group in evidence:
                    if isinstance(ev_group, list):
                        for ev_item in ev_group:
                            if isinstance(ev_item, list) and len(ev_item) >= 3:
                                # Evidence format: [annotation_id, evidence_id, wiki_url, sent_id]
                                # Get the text from the evidence
                                ev_text = str(ev_item[2]) if len(ev_item) > 2 else ""
                                if ev_text and ev_text != "":
                                    evidence_texts.append(ev_text)
                context_text = "\n".join(evidence_texts[:3])  # Limit context

            # Build prompt for fact verification
            fact_prompt = f"Given the evidence, classify this claim as SUPPORTS, REFUTES, or NOT ENOUGH INFO: {claim}"
            prompt = model.format_prompt(fact_prompt, context_text if context_text.strip() else None)

            t0 = time.time()
            resp, tok = safe_generate(model, prompt)
            dt = time.time() - t0

            # Better FEVER evaluation - check for label keywords in response
            resp_upper = resp.upper()
            predicted_label = ""
            if "SUPPORT" in resp_upper and "NOT" not in resp_upper:
                predicted_label = "SUPPORTS"
            elif "REFUTE" in resp_upper:
                predicted_label = "REFUTES"
            elif "NOT ENOUGH" in resp_upper or "INSUFFICIENT" in resp_upper:
                predicted_label = "NOT ENOUGH INFO"
            
            em_score = 1.0 if predicted_label == label else 0.0
            scores = {'em': em_score, 'f1': em_score}  # For classification, EM=F1

            results.append({
                'dataset': 'fever',
                'claim': claim,
                'response': resp,
                'label': label,
                'predicted_label': predicted_label,
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'inference_time': dt,
                'tokens_generated': tok,
                'utility_score': model.extract_utility_score(resp),
                'is_relevant': model.extract_relevance(resp),
                'support_level': model.extract_support(resp),
                'uses_retrieval': model.uses_retrieval(resp),
                'has_context': bool(context_text.strip())
            })

            if (i + 1) % 10 == 0:
                logger.info(f"FEVER processed {i + 1}/{len(ds) if not streaming else sample_size}")

        except Exception as e:
            logger.error(f"FEVER item {i} error: {e}", exc_info=True)

    logger.info(f"FEVER completed with {len(results)} samples")
    return results

    
def run_ms_marco_benchmark(model, sample_size: int = 200, streaming=False):
    """
    Proxy with MS MARCO v2.1 validation; robust passage handling.
    """
    logger.info(f"Running RAGBench proxy (MS MARCO) sample_size={sample_size} (streaming={streaming})")
    try:
        try:
            if streaming:
                ds_iter = load_dataset_retry("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
                ds = list(itertools.islice(ds_iter, sample_size))
            else:
                ds = load_dataset_retry("microsoft/ms_marco","v2.1", split="validation", download_config=DC)
                if sample_size < len(ds): ds = ds.select(range(sample_size))
        except Exception as e:
            logger.warning(f"MS MARCO not available: {e}")
            return []

        results=[]
        for i,item in enumerate(ds):
            try:
                query = item.get("query","")

                # passages may be dict of lists
                passages = item.get("passages", {})
                if isinstance(passages, dict):
                    texts = passages.get("passage_text", [])
                    if isinstance(texts, list):
                        context_text = "\n".join(t for t in texts[:5] if t)
                    else:
                        context_text = ""
                else:
                    context_text = ""

                # answers
                answers = [a for a in (item.get("answers") or []) if a]
                wf = [a for a in (item.get("wellFormedAnswers") or []) if a]
                answer_texts = answers + wf

                prompt = model.format_prompt(query, context_text if context_text.strip() else None)
                t0=time.time(); resp, tok = safe_generate(model, prompt); dt=time.time()-t0
                scores = evaluator.evaluate_multiple_answers(resp, answer_texts) if answer_texts else {'em':0.0,'f1':0.0}

                results.append({
                    'dataset':'msmarco','query':query,'response':resp,
                    'ground_truth_answers':answer_texts,'exact_match':scores['em'],'f1_score':scores['f1'],
                    'inference_time':dt,'tokens_generated':tok,
                    'utility_score':model.extract_utility_score(resp),'is_relevant':model.extract_relevance(resp),
                    'support_level':model.extract_support(resp),'uses_retrieval':model.uses_retrieval(resp),
                    'num_passages': len(passages.get("passage_text", [])) if isinstance(passages, dict) else 0
                })
                if (i+1)%10==0: logger.info(f"MSMarco processed {i+1}/{len(ds) if not streaming else sample_size}")
            except Exception as e:
                logger.error(f"MSMarco item {i} error: {e}", exc_info=True)
        logger.info(f"MSMarco proxy completed with {len(results)} samples")
        return results
    except Exception as e:
        logger.error(f"Error running MSMarco proxy: {e}", exc_info=True)
        return []

def run_ragtruth_benchmark(model, sample_size: int = 200, streaming: bool = False):
    """
    RAGTruth benchmark from wandb/RAGTruth-processed
    """
    logger.info(f"Running RAGTruth sample_size={sample_size} (streaming={streaming})")
    try:
        if streaming:
            ds_iter = load_dataset_retry("wandb/RAGTruth-processed", split="train", streaming=True, download_config=DC)
            ds = list(itertools.islice(ds_iter, sample_size))
        else:
            ds = load_dataset_retry("wandb/RAGTruth-processed", split="train", download_config=DC)
            if sample_size < len(ds):
                ds = ds.select(range(sample_size))
    except Exception as e:
        logger.error(f"Failed to load RAGTruth: {e}", exc_info=True)
        return []

    results = []
    for i, item in enumerate(ds):
        try:
            # RAGTruth fields
            question = item.get("question", "") or item.get("query", "") or ""
            context = item.get("context", "") or item.get("passage", "") or ""
            answer = item.get("answer", "") or item.get("ground_truth", "") or ""
            
            # Handle different answer formats
            if isinstance(answer, list):
                answer_texts = [str(a) for a in answer if a]
            else:
                answer_texts = [str(answer)] if answer else []

            prompt = model.format_prompt(question, context if context.strip() else None)
            t0 = time.time()
            resp, tok = safe_generate(model, prompt)
            dt = time.time() - t0

            scores = evaluator.evaluate_multiple_answers(resp, answer_texts) if answer_texts else {'em': 0.0, 'f1': 0.0}

            results.append({
                'dataset': 'ragtruth',
                'question': question,
                'response': resp,
                'ground_truth_answers': answer_texts,
                'exact_match': scores['em'],
                'f1_score': scores['f1'],
                'inference_time': dt,
                'tokens_generated': tok,
                'utility_score': model.extract_utility_score(resp),
                'is_relevant': model.extract_relevance(resp),
                'support_level': model.extract_support(resp),
                'uses_retrieval': model.uses_retrieval(resp),
                'has_context': bool(context.strip())
            })

            if (i + 1) % 10 == 0:
                logger.info(f"RAGTruth processed {i + 1}/{len(ds) if not streaming else sample_size}")

        except Exception as e:
            logger.error(f"RAGTruth item {i} error: {e}", exc_info=True)

    logger.info(f"RAGTruth completed with {len(results)} samples")
    return results

# ----------------------- Aggregation & I/O -----------------------

def compute_aggregate_metrics(results):
    if not results: return {}
    metrics = ['exact_match','f1_score','utility_score']
    aggregated={}
    for m in metrics:
        vals=[r.get(m,0.0) for r in results if m in r]
        if vals:
            aggregated[m] = {
                'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'count': len(vals), 'min': float(np.min(vals)), 'max': float(np.max(vals))
            }
    for b in ['is_relevant','uses_retrieval']:
        vals=[float(r.get(b,False)) for r in results if b in r]
        if vals:
            aggregated[b] = {'mean': float(np.mean(vals)), 'count': len(vals)}
    support = Counter([r.get('support_level','unknown') for r in results])
    aggregated['support_distribution'] = dict(support)
    return aggregated

def save_results_to_json(results, filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving {filename}: {e}", exc_info=True)

# ----------------------- Main -----------------------

def main():
    print("="*70)
    print("SELF-RAG EVALUATION (Schema-safe + Retry-hardened)")
    print("="*70)

    logger.info("Initializing Self-RAG model...")
    try:
        model = SelfRAGModel(
            model_path="selfrag/selfrag_llama2_7b",
            download_dir="/gscratch/h2lab/akari/model_cache",
            dtype="half"
        )
        logger.info("✅ Model init OK")
    except Exception as e:
        logger.error(f"❌ Model init failed: {e}", exc_info=True)
        return

    # Smaller default to prove the loop, then scale up after it works
    sample_size = int(os.environ.get("SR_SAMPLE_SIZE", "200"))
    streaming = os.environ.get("SR_STREAMING", "0") == "1"

    results={}
    # All benchmarks including fixed FEVER and RAGTruth
    benchmarks = [
        ("Natural Questions", run_natural_questions_benchmark),
        ("TriviaQA", run_trivia_qa_benchmark),
        ("HotpotQA", run_hotpot_qa_benchmark),
        ("SQuAD v2", run_squad_v2_benchmark),
        ("FEVER", run_fever_benchmark),
        ("MSMarco", run_ms_marco_benchmark),
        ("RAGTruth", run_ragtruth_benchmark),
    ]

    logger.info(f"Running {len(benchmarks)} benchmarks; sample_size={sample_size}; streaming={streaming}")
    for name, func in benchmarks:
        print(f"\n{'='*60}\n🚀 RUNNING: {name}\n{'='*60}")
        try:
            t0=time.time()
            bench_results = func(model, sample_size=sample_size, streaming=streaming)
            t1=time.time()
            key = name.lower().replace(" ","_")
            if bench_results:
                aggregated = compute_aggregate_metrics(bench_results)
                results[key] = {
                    'individual_results': bench_results,
                    'aggregated_metrics': aggregated,
                    'total_samples': len(bench_results),
                    'execution_time': t1 - t0
                }
                logger.info(f"✅ {name}: {len(bench_results)} samples in {t1-t0:.2f}s")
            else:
                results[key] = {
                    'individual_results': [],
                    'aggregated_metrics': {},
                    'total_samples': 0,
                    'execution_time': t1 - t0,
                    'status': 'failed'
                }
                logger.warning(f"⚠️ {name} produced no results")
            save_results_to_json(results, f"selfrag_results_partial_{int(time.time())}.json")
        except Exception as e:
            logger.error(f"❌ Error running {name}: {e}", exc_info=True)
            key = name.lower().replace(" ","_")
            results[key] = {
                'individual_results': [],
                'aggregated_metrics': {},
                'total_samples': 0,
                'execution_time': 0,
                'status': 'error',
                'error_message': str(e)
            }

    final = f"selfrag_evaluation_final_{int(time.time())}.json"
    save_results_to_json(results, final)

    # Console summary
    print("\n" + "="*80)
    print("🏆 SELF-RAG EVALUATION COMPLETE - FINAL SUMMARY")
    print("="*80)
    succ = sum(1 for v in results.values() if v.get('total_samples',0)>0)
    total = sum(v.get('total_samples',0) for v in results.values())
    for k,v in results.items():
        name = k.upper().replace("_"," ")
        if v.get('total_samples',0)>0:
            ag=v['aggregated_metrics']
            print(f"\n📈 {name}: n={v['total_samples']}  time={v.get('execution_time',0):.2f}s")
            if 'exact_match' in ag:
                em=ag['exact_match']; print(f"   EM: {em['mean']:.3f} ± {em['std']:.3f} (n={em['count']})")
            if 'f1_score' in ag:
                f1=ag['f1_score']; print(f"   F1: {f1['mean']:.3f} ± {f1['std']:.3f} (n={f1['count']})")
            if 'utility_score' in ag:
                u=ag['utility_score']; print(f"   Utility: {u['mean']:.3f} ± {u['std']:.3f}")
        else:
            print(f"\n❌ {name}: {v.get('status','no-data')}")

    print("\n" + "="*80)
    print(f"📊 OVERALL: {succ}/7 benchmarks produced results; total samples: {total}")
    print(f"🗂  Results saved to: {final}")
    print("="*80)
    return results

def _dist_cleanup():
    try:
        if dist.is_available() and dist.is_initialized():
            # Optional: try a short barrier so peers don't race the destroy
            try:
                dist.barrier(timeout=datetime.timedelta(seconds=5))
            except Exception:
                pass
            dist.destroy_process_group()
    except Exception:
        pass

atexit.register(_dist_cleanup)

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")

    print("🔥 SELF-RAG EVALUATION SYSTEM (hardened)")
    print("="*70)
    print("🔍 Pre-flight checks...")

    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory/1e9
            print(f"✅ GPU: {name} ({mem:.1f} GB), {n} visible")
        else:
            print("⚠️ No GPU detected")
    except Exception:
        print("⚠️ PyTorch not available for GPU check")

    for pkg in ['vllm','datasets','transformers','torch']:
        try:
            __import__(pkg); print(f"✅ {pkg} available")
        except Exception:
            print(f"❌ {pkg} missing")

    print("\n🚀 Starting evaluation...")
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal: {e}", exc_info=True)
        print("❌ Evaluation failed. See logs.")
    finally:
        _dist_cleanup()
