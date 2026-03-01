# baseline_eval.py
from ultralytics import YOLO


def run_evaluation(
    model_path="runs/detect/runs/train/final_try_baseline_fixed_5/weights/best.pt",
    data_yaml="dataset/dataset/data.yaml",
    conf=0.5,
    iou=0.5,
):
    """Validate a YOLO model and return a metrics dict."""
    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        conf=conf,
        iou=iou,
        verbose=False,
        workers=0,
    )

    metrics = {
        "precision": float(results.results_dict["metrics/precision(B)"]),
        "recall":    float(results.results_dict["metrics/recall(B)"]),
        "mAP50":     float(results.results_dict["metrics/mAP50(B)"]),
        "mAP50_95":  float(results.results_dict["metrics/mAP50-95(B)"]),
    }

    print("\n── Evaluation Results ──")
    print(f"Precision : {metrics['precision']:.3f}")
    print(f"Recall    : {metrics['recall']:.3f}")
    print(f"mAP50     : {metrics['mAP50']:.3f}")
    print(f"mAP50-95  : {metrics['mAP50_95']:.3f}")

    return metrics


if __name__ == '__main__':
    run_evaluation()