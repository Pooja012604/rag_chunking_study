import json
import random
import csv
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SQUAD_FILE = DATA_DIR / "dev-v1.1.json"

OUT_DIR = Path(__file__).resolve().parents[1]
RESULTS_CSV = OUT_DIR / "retrieval_results.csv"

N_QA = 300     # fast because no LLM calls
TOP_K = 2

def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s

def load_squad(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]["text"] if qa["answers"] else ""
                if answer:
                    rows.append({"context": context, "question": question, "gold_answer": answer})
    return rows

def chunk_words(text: str, chunk_words: int, overlap_words: int):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks

def build_index(model, texts):
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index

def search(model, index, chunks, query, top_k=4):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append((float(score), chunks[int(idx)]))
    return results

def gold_in_retrieved(gold: str, retrieved_text: str) -> int:
    g = norm(gold)
    r = norm(retrieved_text)
    if not g:
        return 0
    return 1 if g in r else 0

def main():
    if not SQUAD_FILE.exists():
        raise FileNotFoundError(f"Could not find {SQUAD_FILE}. Put dev-v1.1.json in /data")

    rows = load_squad(SQUAD_FILE)
    random.seed(42)
    sample = random.sample(rows, k=min(N_QA, len(rows)))
    print("Sample size:", len(sample))

    strategies = [
        ("small", 200, 50),
        ("medium", 400, 80),
        ("large", 800, 120),
    ]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    out_rows = []

    for strat_name, cw, ow in strategies:
        # Build chunks from the sampled contexts
        all_chunks = []
        for r in sample:
            all_chunks.extend(chunk_words(r["context"], cw, ow))
        all_chunks = list(dict.fromkeys(all_chunks))  # de-dup

        print(f"\n[{strat_name}] chunks: {len(all_chunks)}")
        index = build_index(model, all_chunks)

        hits = 0
        for i, r in enumerate(sample, 1):
            q = r["question"]
            gold = r["gold_answer"]

            retrieved = search(model, index, all_chunks, q, top_k=TOP_K)
            retrieved_text = "\n\n".join([t for _, t in retrieved])

            hit = gold_in_retrieved(gold, retrieved_text)
            hits += hit

            out_rows.append({
                "strategy": strat_name,
                "chunk_words": cw,
                "overlap_words": ow,
                "top_k": TOP_K,
                "question": q,
                "gold_answer": gold,
                "hit_gold_in_topk": hit,
            })

            if i % 50 == 0:
                print(f"  {strat_name}: completed {i}/{len(sample)}")

        print(f"[{strat_name}] Grounded@{TOP_K} = {hits/len(sample)*100:.1f}%")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print("\nSaved:", RESULTS_CSV)

if __name__ == "__main__":
    main()
