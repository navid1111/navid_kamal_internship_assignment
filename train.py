import os
import wandb
from ultralytics import YOLO


def run_training(
    data_yaml="dataset/dataset_stratified/data.yaml",
    model_base="yolo11m.pt",
    project="runs/train",
    name="final_try_baseline_fixed_5",
    epochs=100,
    imgsz=640,
    batch=8,
    use_wandb=True,
):
    """Train YOLO and return the path to the best checkpoint."""

    # Try to initialize W&B — training continues even if it fails
    run = None
    if use_wandb:
        try:
            run = wandb.init(
                entity="navidkamal-islamic-university-of-technology",
                project="retail_object-detection",
                name=name,
                config={
                    "model": model_base,
                    "epochs": epochs,
                    "imgsz": imgsz,
                    "batch": batch,
                    "optimizer": "SGD",
                    "lr0": 0.01,
                    "cls": 1.5,
                },
            )
            print("✅ W&B initialized successfully.")
        except Exception as e:
            print(f"⚠️  W&B login failed: {e}")
            print("⚠️  Continuing training without W&B logging...\n")

    model = YOLO(model_base)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        conf=0.5,
        iou=0.5,
        cls=1.5,
        augment=False,
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        patience=15,
        device=0,  # Use first GPU (CUDA:0), set to 'cpu' to disable
        workers=0,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
    )

    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    print(f"\n── Training complete ──")
    print(f"Best model saved to: {best}")

    # Log to W&B only if the run was successfully initialized
    if run is not None:
        try:
            run.summary["best_precision"] = results.results_dict.get("metrics/precision(B)", 0)
            run.summary["best_recall"] = results.results_dict.get("metrics/recall(B)", 0)
            run.summary["best_mAP50"] = results.results_dict.get("metrics/mAP50(B)", 0)
            run.summary["best_mAP50_95"] = results.results_dict.get("metrics/mAP50-95(B)", 0)
            run.summary["best_fitness"] = results.best_fitness

            run.log({
                "final_precision": results.results_dict.get("metrics/precision(B)", 0),
                "final_recall": results.results_dict.get("metrics/recall(B)", 0),
                "final_mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "final_mAP50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "total_epochs_trained": len(results.epochs) if hasattr(results, 'epochs') else epochs,
            })
            run.finish()
            print("✅ W&B run finished and metrics logged.")
        except Exception as e:
            print(f"⚠️  W&B logging failed after training: {e}")
    else:
        print("ℹ️  W&B was not active — metrics not logged remotely.")
        print(f"ℹ️  Results are saved locally at: {results.save_dir}")

    return best


if __name__ == '__main__':
    run_training()