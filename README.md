# Retail OS IM  Assignment

## Overview
This project is focused on object detection for -related classes using YOLOv8 and Weights & Biases (W&B) for experiment tracking. The dataset is highly imbalanced and contains 76 classes, with many classes having very few samples. The project demonstrates the impact of proper dataset splitting and experiment tracking on model performance.

---

## Project Structure
```
assignment/
├── dataset/                # Original dataset (images & labels, ignored by git)
├── runs/                   # YOLO training runs (ignored by git)
├── wandb/                  # W&B logs (ignored by git)
├── train.py                # Main training script
├── diagnose.py             # Dataset analysis script
├── check_splits.py         # Checks class coverage in splits
├── rebuild_splits.py       # Script to rebuild stratified splits
├── requirements.txt        # Python dependencies
└── ...
```

---

## Dataset Issues & Solution
- **Original splits:** Many classes missing from val/test, severe imbalance (see below)
- **After stratified split:** Nearly all classes present in all splits, improved recall

### Example: Class Coverage Before/After
| Split   | Classes Present (Before) | Classes Present (After) |
|---------|-------------------------|-------------------------|
| Train   | 73                      | 76                      |
| Val     | 32                      | 74                      |
| Test    | 29                      | 74                      |

- Classes with <10 train samples remain hard to learn.

---

## Training Results

### Before Stratified Split
- Many classes missing from val/test
- Recall and mAP were much lower

### After Stratified Split
- **Precision:** ↑ 0.93
- **Recall:** ↑ 0.91
- **mAP50:** ↑ 0.94
- **mAP50-95:** ↑ 0.68

![Loss and Metrics Curves](runs/detect/runs/train/final_try_baseline_fixed_5/results.png)

---

## Confusion Matrix

![Confusion Matrix](runs/detect/runs/train/final_try_baseline_fixed_5/confusion_matrix_normalized.png)

---

## How to Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up W&B:**
   ```bash
   wandb login
   ```
3. **Train the model:**
   ```bash
   python train.py
   ```
4. **Analyze splits:**
   ```bash
   python check_splits.py
   python diagnose.py
   ```
5. **Rebuild splits (if needed):**
   ```bash
   python rebuild_splits.py
   ```

---

## Key Scripts
- `train.py`: Trains YOLOv8, logs to W&B
- `diagnose.py`: Prints per-class stats, imbalance
- `check_splits.py`: Checks class coverage in splits
- `rebuild_splits.py`: Rebuilds splits with stratified sampling

---

## Notes
- The dataset and all outputs are ignored by git for privacy and size reasons.
- For best results, always check class coverage and rebalance splits if needed.
- W&B project: [retail_object-detection](https://wandb.ai/navidkamal-islamic-university-of-technology/retail_object-detection)

---

## Example Results Table
| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 100   | 0.93      | 0.91   | 0.94  | 0.68     |

---

## Author
Navid Kamal
Islamic University of Technology
