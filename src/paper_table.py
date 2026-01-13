import csv
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parents[1] / "retrieval_results.csv"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Not found: {CSV_PATH}")

    # Aggregate hits per strategy
    total = defaultdict(int)
    hits = defaultdict(int)
    cw = {}
    ow = {}
    k = None

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strat = row["strategy"]
            total[strat] += 1
            hits[strat] += int(row["hit_gold_in_topk"])
            cw[strat] = int(row["chunk_words"])
            ow[strat] = int(row["overlap_words"])
            k = int(row["top_k"])

    print("\nFINAL TABLE (copy into paper):")
    print("strategy | chunk_words | overlap_words | n | Grounded@k (%)")
    print("-" * 62)

    for strat in sorted(total.keys()):
        n = total[strat]
        score = hits[strat] / n * 100
        print(f"{strat:8} | {cw[strat]:10d} | {ow[strat]:12d} | {n:3d} | {score:13.1f}")

    print(f"\n(k = {k})  file = {CSV_PATH}")

if __name__ == "__main__":
    main()
