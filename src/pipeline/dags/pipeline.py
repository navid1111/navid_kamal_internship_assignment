"""
Retail Object-Detection Pipeline
─────────────────────────────────
check_data_quality
        ↓
   [BranchOperator]
    ↙           ↘
rebuild_splits   skip_rebuild
    ↘           ↙
   [join / gate]
        ↓
   train_model
        ↓
  evaluate_model
        ↓
 run_share_of_shelf
        ↓
  push_to_registry
"""

from airflow.sdk import dag, task
import os
import sys

# Ensure the project root is on sys.path so modules are importable
_PROJECT_ROOT = os.environ.get("PYTHONPATH", "/opt/airflow/project")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import from src package
from src.data.diagnose import analyze_dataset
from src.data.rebuild_splits import main as rebuild_main
from src.model.train import run_training
from src.model.eval import run_evaluation
from src.utils.shelf import run_from_model
from src.utils.gpu import check_gpu_availability

# ── Configuration (paths are relative to the project root) ───────
_P = _PROJECT_ROOT  # short alias
DATASET_YAML        = os.path.join(_P, "dataset/dataset_stratified/data.yaml")
STRATIFIED_YAML     = os.path.join(_P, "dataset/dataset_stratified/data.yaml")
EXPECTED_CLASSES     = 70
IMBALANCE_THRESHOLD  = 10.0

MODEL_BASE    = os.path.join(_P, "yolo11m.pt")
TRAIN_PROJECT = os.path.join(_P, "runs/train")
TRAIN_NAME    = "pipeline_run"


@dag(dag_id="branch_dag")
def branch_dag():

    # ── 1. Check data quality ────────────────────────────────
    @task.python
    def check_data_quality():
        """Analyse class balance / coverage and decide if splits need rebuilding."""
        results = analyze_dataset(DATASET_YAML)

        needs_rebuild = not os.path.exists(STRATIFIED_YAML)
        for _split, stats in results.items():
            if stats.get("num_classes", 0) < EXPECTED_CLASSES:
                needs_rebuild = True

        return {"needs_rebuild": needs_rebuild, "diagnostics": results}

    # ── 2. Branch decision ───────────────────────────────────
    @task.branch
    def decide_rebuild(quality_result):
        """Route to rebuild_splits or skip_rebuild based on quality check."""
        if quality_result["needs_rebuild"]:
            return "rebuild_splits"
        return "skip_rebuild"

    # ── 3a. Rebuild splits (stratified) ──────────────────────
    @task.python
    def rebuild_splits():
        """Re-create train/val/test with stratified sampling."""
        rebuild_main(base_dir=_PROJECT_ROOT)

    # ── 3b. Skip rebuild ─────────────────────────────────────
    @task.python
    def skip_rebuild():
        """No-op — data quality is acceptable."""
        print("Data quality acceptable — skipping rebuild.")

    # ── 4. Join gate ─────────────────────────────────────────
    @task.python(trigger_rule="none_failed_min_one_success")
    def join():
        """Rejoin the two branches before training."""
        print("Branch complete — continuing pipeline.")

    # ── 4.5. Check GPU ───────────────────────────────────────
    @task.python
    def check_gpu():
        """Check GPU availability before training."""
        print(f"✓ GPU available: {check_gpu_availability()}")
        return check_gpu_availability()

    # ── 5. Train model ───────────────────────────────────────
    @task.python
    def train_model():
        """Train YOLO on the (possibly rebuilt) stratified dataset."""
        best = run_training(
            data_yaml=STRATIFIED_YAML,
            model_base=MODEL_BASE,
            project=TRAIN_PROJECT,
            name=TRAIN_NAME,
            use_wandb=True,
        )
        return best

    # ── 6. Evaluate model ────────────────────────────────────
    @task.python
    def evaluate_model(model_path: str):
        """Validate the trained model and return key metrics."""
        metrics = run_evaluation(
            model_path=model_path,
            data_yaml=STRATIFIED_YAML,
        )
        return {"model_path": model_path, "metrics": metrics}

    # ── 7. Share-of-Shelf analytics ──────────────────────────
    @task.python
    def run_share_of_shelf(eval_result: dict):
        """Run predictions on test images and compute shelf-share percentages."""
        run_from_model(
            model_path=eval_result["model_path"],
            test_images=os.path.join(_P, "dataset/dataset_stratified/test/images"),
            output_json=os.path.join(_P, "final_try_results.json"),
            output_chart=os.path.join(_P, "share_of_shelf.png"),
        )
        return eval_result["model_path"]

    # ── 8. Push to model registry ────────────────────────────
    @task.python
    def push_to_registry(model_path: str):
        """Push the best checkpoint to the model registry."""
        print(f"Pushing model to registry: {model_path}")
        # TODO: implement actual push (e.g. MLflow, W&B, S3)
        print("Model registered successfully (placeholder).")

    # ── DAG wiring ───────────────────────────────────────────
    quality     = check_data_quality()
    branch      = decide_rebuild(quality)
    do_rebuild  = rebuild_splits()
    do_skip     = skip_rebuild()
    gate        = join()
    gpu_info    = check_gpu()
    model_path  = train_model()
    eval_result = evaluate_model(model_path)
    shelf_out   = run_share_of_shelf(eval_result)
    push_to_registry(shelf_out)

    branch >> [do_rebuild, do_skip] >> gate >> gpu_info >> model_path


# Instantiate the DAG
branch_dag()
