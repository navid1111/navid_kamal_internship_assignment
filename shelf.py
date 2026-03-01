"""
Share of Shelf Analytics
Treats the entire test dataset as a single representative shelf.
Calculates and visualizes Percentage Share of Shelf per product class (SKU).
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter
from pathlib import Path


def load_results(json_path: str) -> list:
    """Load YOLO prediction results from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def count_detections(results: list) -> Counter:
    """Count total detections per class across all images."""
    class_counts = Counter()
    for entry in results:
        # Support both formats: list of dicts or dict with 'predictions' key
        predictions = entry.get("predictions", entry) if isinstance(entry, dict) else entry
        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, dict):
                    label = pred.get("class_name") or pred.get("name") or pred.get("label") or str(pred.get("class", "unknown"))
                    class_counts[label] += 1
    return class_counts


def calculate_share_of_shelf(class_counts: Counter) -> dict:
    """Calculate percentage share of shelf for each class."""
    total = sum(class_counts.values())
    if total == 0:
        return {}
    return {cls: (count / total) * 100 for cls, count in class_counts.items()}


def plot_share_of_shelf(share: dict, counts: Counter, output_path: str = "share_of_shelf.png"):
    """Generate a clean horizontal bar chart of Share of Shelf."""
    # Sort by percentage descending
    sorted_items = sorted(share.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    percentages = [item[1] for item in sorted_items]
    raw_counts = [counts[lbl] for lbl in labels]

    n = len(labels)
    fig_height = max(8, n * 0.35)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Color gradient from deep blue to light blue
    colors = plt.cm.Blues(np.linspace(0.85, 0.35, n))

    bars = ax.barh(range(n), percentages, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    # Add percentage + count labels on each bar
    for i, (bar, pct, cnt) in enumerate(zip(bars, percentages, raw_counts)):
        x_pos = bar.get_width()
        label_text = f"  {pct:.1f}%  (n={cnt})"
        ax.text(
            x_pos + 0.1, i, label_text,
            va="center", ha="left", fontsize=8,
            color="#333333", fontweight="bold"
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel("Share of Shelf (%)", fontsize=12, labelpad=10)
    ax.set_title(
        "Share of Shelf Analytics\nPercentage of Each SKU Across Test Dataset",
        fontsize=14, fontweight="bold", pad=20
    )

    ax.set_xlim(0, max(percentages) * 1.3)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    total = sum(counts.values())
    fig.text(
        0.99, 0.01,
        f"Total detections: {total:,}  |  Classes: {len(labels)}",
        ha="right", va="bottom", fontsize=9, color="gray"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ Chart saved to: {output_path}")
    plt.close()


def print_summary_table(share: dict, counts: Counter):
    """Print a sorted summary table to the console."""
    sorted_items = sorted(share.items(), key=lambda x: x[1], reverse=True)
    total = sum(counts.values())

    print("\n" + "=" * 55)
    print(f"{'Rank':<5} {'Class':<25} {'Count':>7} {'Share':>8}")
    print("=" * 55)
    for rank, (cls, pct) in enumerate(sorted_items, 1):
        print(f"{rank:<5} {cls:<25} {counts[cls]:>7,} {pct:>7.2f}%")
    print("=" * 55)
    print(f"{'TOTAL':<5} {'':<25} {total:>7,} {'100.00%':>8}")
    print("=" * 55 + "\n")


def run_from_model(
    model_path,
    test_images="dataset/dataset_stratified/test/images",
    output_json="final_try_results.json",
    output_chart="share_of_shelf.png",
    conf=0.5,
    iou=0.5,
):
    """Run predictions with a trained model and compute share-of-shelf analytics."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    results = model.predict(
        source=test_images,
        conf=conf,
        iou=iou,
        save=False,
        verbose=False,
    )

    # Convert YOLO results to the JSON format expected by shelf helpers
    predictions = []
    for r in results:
        preds = []
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i])
            preds.append({"class_name": r.names[cls_id], "class": cls_id})
        predictions.append({"predictions": preds})

    with open(output_json, "w") as f:
        import json
        json.dump(predictions, f, indent=2)
    print(f"📂 Predictions saved to: {output_json}")

    class_counts = count_detections(predictions)
    if not class_counts:
        print("❌ No detections found.")
        return

    share = calculate_share_of_shelf(class_counts)
    print_summary_table(share, class_counts)
    plot_share_of_shelf(share, class_counts, output_path=output_chart)


def main():
    # Locate results JSON — try common paths
    candidate_paths = [
        "final_try_results.json",
        "runs/detect/runs/train/final_try_baseline_fixed_5/final_try_results.json",
        "runs/final_try_results.json",
    ]

    json_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            json_path = path
            break

    if json_path is None:
        # Fallback: search recursively
        matches = list(Path(".").rglob("final_try_results.json"))
        if matches:
            json_path = str(matches[0])

    if json_path is None:
        print("❌ Could not find final_try_results.json")
        print("   Please place it in the working directory or update candidate_paths.")
        return

    print(f"📂 Loading results from: {json_path}")
    results = load_results(json_path)

    class_counts = count_detections(results)
    if not class_counts:
        print("❌ No detections found in results file. Check the JSON structure.")
        return

    share = calculate_share_of_shelf(class_counts)
    print_summary_table(share, class_counts)
    plot_share_of_shelf(share, class_counts, output_path="share_of_shelf.png")


if __name__ == "__main__":
    main()