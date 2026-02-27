# baseline_eval.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/runs/train/final_try_baseline_fixed_5/weights/best.pt')  # pretrained base — we haven't trained yet

    results = model.val(
        data='dataset/dataset/data.yaml',
        conf=0.5,
        iou=0.5,
        verbose=False
    )

    print("\n── Baseline on Fixed Dataset ──")
    print(f"Precision : {results.results_dict['metrics/precision(B)']:.3f}")
    print(f"Recall    : {results.results_dict['metrics/recall(B)']:.3f}")
    print(f"mAP50     : {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"mAP50-95  : {results.results_dict['metrics/mAP50-95(B)']:.3f}")