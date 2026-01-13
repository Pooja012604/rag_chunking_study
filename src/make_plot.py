import matplotlib.pyplot as plt
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[1] / "grounded_plot.png"

def main():
    # Your measured results (n=300, SQuAD dev)
    ks = [1, 2, 4]

    grounded = {
        "small (200/50)":  [84.3, 91.0, 95.0],
        "medium (400/80)": [84.3, 91.0, 95.0],
        "large (800/120)": [84.7, 91.0, 95.0],
    }

    for label, ys in grounded.items():
        plt.plot(ks, ys, marker="o", label=label)

    plt.xticks(ks)
    plt.ylim(0, 100)
    plt.xlabel("k (top-k retrieved chunks)")
    plt.ylabel("Grounded@k (%)  [gold answer in retrieved context]")
    plt.title("Effect of Chunk Size on Retrieval Groundedness")
    plt.grid(True)
    plt.legend()

    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
