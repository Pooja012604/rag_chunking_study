import csv
import re
from collections import defaultdict
from pathlib import Path

RESULTS_CSV = Path(__file__).resolve().parents[1] / "results.csv"

def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation
    return s

def contains_answer(pred: str, gold: str) -> int:
    """Soft exact-match: 1 if gold appears inside pred (after normalization)."""
    p = norm(pred)
    g = norm(gold)
    if not g:
        return 0
    return 1 if g in p else 0

def grounded(gold: str, retrieved_context: str) -> int:
    """
    Better groundedness proxy for chunking experiments:
    1 if the GOLD answer appears in retrieved context (after normalization).
    This measures whether retrieval brought supporting evidence.
    """
    g = norm(gold)
    c = norm(retrieved_context)
    if not g:
        return 0
    return 1 if g in c else 0

def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"results.csv not found at: {RESULTS_CSV}")

    by_strategy = defaultdict(list)

    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strat = row["strategy"]
            pred = row["pred_answer"]
            gold = row["gold_answer"]
            ctx = row["retrieved_context"]
            latency = float(row["latency_sec"])

            em = contains_answer(pred, gold)
            gr = grounded(gold, ctx)

            by_strategy[strat].append((em, gr, latency))

    print("\nResults (on your sample):")
    print("strategy | n | exact_match% | grounded% | avg_latency_sec")
    print("-" * 60)

    for strat in sorted(by_strategy.keys()):
        items = by_strategy[strat]
        n = len(items)
        em_avg = sum(x[0] for x in items) / n * 100
        gr_avg = sum(x[1] for x in items) / n * 100
        lat_avg = sum(x[2] for x in items) / n

        print(f"{strat:8} | {n:1d} | {em_avg:10.1f} | {gr_avg:8.1f} | {lat_avg:14.2f}")

if __name__ == "__main__":
    main()
