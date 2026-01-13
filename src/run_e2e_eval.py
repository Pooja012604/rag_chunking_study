# src/run_e2e_eval.py
import json
import os
import random
import re
import string
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import faiss
import ollama
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------
@dataclass
class ChunkConfig:
    name: str
    chunk_words: int
    overlap_words: int


CHUNK_CONFIGS = [
    ChunkConfig("small", 200, 50),
    ChunkConfig("medium", 400, 80),
    ChunkConfig("large", 800, 120),
]

DEFAULT_MODEL = "phi3:mini"
DEFAULT_K_LIST = [1, 2, 4]


# -----------------------------
# SQuAD utils
# -----------------------------
def load_squad_v1(path: str) -> List[Dict]:
    """
    Returns a flat list of QA items:
    {id, context, question, answers(list[str])}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                qid = qa.get("id", "")
                question = qa["question"]
                answers = [a["text"] for a in qa.get("answers", [])]
                # SQuAD v1.1 always has answers, but keep safe fallback
                if not answers:
                    answers = [""]
                items.append(
                    {
                        "id": qid,
                        "context": context,
                        "question": question,
                        "answers": answers,
                    }
                )
    return items


def word_chunk(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


# -----------------------------
# Retrieval (Sentence-Transformers + FAISS cosine)
# -----------------------------
def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    return emb


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    # cosine similarity via inner product on L2-normalized vectors
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index


def retrieve_topk(
    model: SentenceTransformer,
    index: faiss.Index,
    chunks: List[str],
    question: str,
    k: int,
) -> List[Tuple[float, str]]:
    q_emb = embed_texts(model, [question])
    scores, idxs = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        results.append((float(score), chunks[int(idx)]))
    return results


# -----------------------------
# SQuAD answer scoring (EM/F1)
# Standard-style normalization used in SQuAD evaluation scripts
# -----------------------------
def normalize_answer(s: str) -> str:
    def lower(text): return text.lower()
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gt_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def squad_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    return max(metric_fn(prediction, gt) for gt in ground_truths)


# -----------------------------
# LLM call (Ollama)
# -----------------------------
def llm_answer_ollama(model_name: str, question: str, context: str) -> str:
    prompt = f"""You are answering a question using ONLY the provided context.
If the answer is not in the context, reply exactly: NOT_FOUND.

Context:
{context}

Question:
{question}

Answer (only the answer, no explanation):
"""
    # Deterministic as possible
    resp = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": 0,
        },
    )
    text = (resp.get("response") or "").strip()

    # Keep only first line (models sometimes add extra)
    text = text.splitlines()[0].strip()

    # Normalize common prefixes
    text = re.sub(r"^(answer\s*:\s*)", "", text, flags=re.IGNORECASE).strip()
    return text


# -----------------------------
# Main experiment
# -----------------------------
def main():
    # Paths
    data_path = os.path.join("data", "dev-v1.1.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}. Put SQuAD dev file in data/ as dev-v1.1.json")

    # Settings (edit these if you want)
    seed = 42
    n = 100   # start small for LLM-based eval; increase later (e.g., 300, 1000, full)
    k_list = DEFAULT_K_LIST
    ollama_model = DEFAULT_MODEL

    print(f"Using Ollama model: {ollama_model}")
    print(f"Sample size (n): {n}")
    print(f"k values: {k_list}")
    print(f"Random seed: {seed}")

    # Load + sample
    all_items = load_squad_v1(data_path)
    print(f"Total QA pairs found: {len(all_items)}")

    random.seed(seed)
    sample = random.sample(all_items, k=min(n, len(all_items)))

    # Shared embedding model
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []

    for cfg in CHUNK_CONFIGS:
        print(f"\n=== Strategy: {cfg.name} ({cfg.chunk_words}/{cfg.overlap_words}) ===")

        # Build chunks for all sampled contexts
        all_chunks = []
        for item in sample:
            all_chunks.extend(word_chunk(item["context"], cfg.chunk_words, cfg.overlap_words))

        # De-duplicate to reduce repeated embedding
        all_chunks = list(dict.fromkeys(all_chunks))
        print(f"Chunks to index: {len(all_chunks)}")

        # Embed + index
        chunk_emb = embed_texts(emb_model, all_chunks, batch_size=64)
        index = build_faiss_index(chunk_emb)

        # Evaluate each k
        for k in k_list:
            print(f"\n  -- Evaluating k={k} --")
            ems, f1s, not_found = 0.0, 0.0, 0

            for i, item in enumerate(sample, start=1):
                top = retrieve_topk(emb_model, index, all_chunks, item["question"], k=k)
                context = "\n\n".join([t[1] for t in top])

                pred = llm_answer_ollama(ollama_model, item["question"], context)

                if pred.strip() == "NOT_FOUND":
                    not_found += 1

                em = squad_max_over_ground_truths(exact_match_score, pred, item["answers"])
                f1 = squad_max_over_ground_truths(f1_score, pred, item["answers"])

                ems += em
                f1s += f1

                rows.append(
                    {
                        "strategy": cfg.name,
                        "chunk_words": cfg.chunk_words,
                        "overlap_words": cfg.overlap_words,
                        "k": k,
                        "id": item["id"],
                        "question": item["question"],
                        "gold_answers": " || ".join(item["answers"]),
                        "pred_answer": pred,
                        "em": em,
                        "f1": f1,
                    }
                )

                if i % 10 == 0:
                    print(f"    progress: {i}/{len(sample)}")

            em_pct = (ems / len(sample)) * 100.0
            f1_pct = (f1s / len(sample)) * 100.0
            nf_pct = (not_found / len(sample)) * 100.0

            print(f"  EM@{k}: {em_pct:.2f}% | F1@{k}: {f1_pct:.2f}% | NOT_FOUND: {nf_pct:.2f}%")

    df = pd.DataFrame(rows)
    out_path = "e2e_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved detailed results to: {os.path.abspath(out_path)}")

    # Summary
    summary = (
        df.groupby(["strategy", "chunk_words", "overlap_words", "k"])[["em", "f1"]]
        .mean()
        .reset_index()
    )
    summary["em_percent"] = summary["em"] * 100
    summary["f1_percent"] = summary["f1"] * 100
    summary_out = "e2e_summary.csv"
    summary.to_csv(summary_out, index=False, encoding="utf-8")
    print(f"Saved summary to: {os.path.abspath(summary_out)}")

    print("\nSummary (mean EM/F1):")
    for _, r in summary.iterrows():
        print(
            f"{r['strategy']:>6} | k={int(r['k'])} | "
            f"EM={r['em_percent']:.2f}% | F1={r['f1_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
