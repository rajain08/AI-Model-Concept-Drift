
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/baseline_full")

with (RESULTS_DIR / "metrics.json").open() as f:
    results = json.load(f)["results"]

for dataset, m in results.items():
    matrix = np.array([
        [m["tn"], m["fp"]],
        [m["fn"], m["tp"]]
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1], labels=["Ham", "Spam"])
    ax.set_yticks([0, 1], labels=["Ham", "Spam"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"{dataset.upper()} Confusion Matrix")

    for row in range(2):
        for col in range(2):
            ax.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center",
                fontsize=15
            )

    fig.tight_layout()

    filename = RESULTS_DIR / f"{dataset}_confusion_matrix.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)

    print(f"Saved: {filename}")

print("All confusion matrices generated successfully.")
