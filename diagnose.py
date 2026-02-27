import os
from collections import Counter
import yaml


def analyze_dataset(data_yaml_path):
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    # Get the directory containing the YAML file for relative path resolution
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))

    for split in ['train', 'val', 'test']:
        if split not in data:
            continue

        image_path = data[split]

        # Make paths absolute relative to the YAML file location
        image_path = os.path.normpath(os.path.join(yaml_dir, image_path))
        # Convert images path to labels path
        label_path = image_path.replace("images", "labels")

        class_counts = Counter()

        print(f"\n{split.upper()} SET:")

        for label_file in os.listdir(label_path):
            label_file_path = os.path.join(label_path, label_file)

            if not label_file.endswith(".txt"):
                continue

            with open(label_file_path) as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1

        print(f"  Total objects: {sum(class_counts.values())}")
        print(f"  Classes present: {len(class_counts)}")

        if class_counts:
            max_cls = max(class_counts.values())
            min_cls = min(class_counts.values())
            print(f"  Imbalance ratio: {max_cls/min_cls:.1f}x")
            print(f"  Per-class counts: {dict(class_counts)}")

analyze_dataset('dataset/dataset_stratified/data.yaml')