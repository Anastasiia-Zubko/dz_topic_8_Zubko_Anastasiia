# Здійсніть імпорт необхідних бібліотек.
# Вибір робочого середовища: на власній машині 
# Підготовка даних: завантажені локально
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from ultralytics import YOLO

# Конфігурація YOLO:
class CFG:
    EPOCHS = 50  
    BATCH_SIZE = 16
    BASE_MODEL = "yolov9s" 
    BASE_MODEL_WEIGHTS = f"{BASE_MODEL}.pt"
    EXP_NAME = "indoor_objects_detection"
    OPTIMIZER = "auto"
    LR = 1e-3
    WEIGHT_DECAY = 5e-4
    PATIENCE = 10
    OUTPUT_DIR = "./runs/detect"

# data.yaml
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "data.yaml")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Dataset configuration file not found: {config_path}")
else:
    print(f"Using dataset configuration: {config_path}")

# Знайомство з даними:
image_paths = glob.glob(os.path.join(script_dir, "valid", "images", "*.jpg"))
label_paths = glob.glob(os.path.join(script_dir, "valid", "labels", "*.txt"))

if len(image_paths) > 0:
    img = cv2.imread(image_paths[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Sample Image from Dataset")
    plt.show()
else:
    print("No images found in validation dataset.")

class_counts = {}
for label_path in label_paths:
    with open(label_path, 'r') as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

plt.bar(class_counts.keys(), class_counts.values())
plt.xlabel("Class ID")
plt.ylabel("Instance Count")
plt.title("Class Distribution in Validation Set")
plt.show()

# Навчання моделі:

model = YOLO(CFG.BASE_MODEL_WEIGHTS)

model.train(
    data=config_path,
    epochs=CFG.EPOCHS,
    batch=CFG.BATCH_SIZE,
    optimizer=CFG.OPTIMIZER,
    lr0=CFG.LR,
    weight_decay=CFG.WEIGHT_DECAY,
    project=CFG.OUTPUT_DIR,
    name=CFG.EXP_NAME
)

results = model.val()
print("Validation Metrics:", results.results_dict)

# Аналіз результатів:
if hasattr(results, 'metrics') and isinstance(results.metrics, dict):
    plt.figure(figsize=(10, 5))
    has_plot = False
    if 'metrics/mAP50(B)' in results.results_dict:
        plt.plot(results.results_dict.get('metrics/mAP50(B)', []), label="mAP@50")
        has_plot = True
    if 'metrics/recall(B)' in results.results_dict:
        plt.plot(results.results_dict.get('metrics/recall(B)', []), label="Recall")
        has_plot = True
    if has_plot:
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.title("Model Performance")
        plt.legend()
        plt.show(block=False)
    else:
        print("Warning: No valid metric data found for plotting.")
else:
    print("Warning: Unable to extract performance metrics.")

if hasattr(results, 'results_dict') and 'metrics/precision(B)' in results.results_dict:
    print("\nPer-Class Performance:")
    for class_id, name in model.names.items():
        precision = results.results_dict.get(f'class/{name}/precision', 'N/A')
        recall = results.results_dict.get(f'class/{name}/recall', 'N/A')
        f1_score = results.results_dict.get(f'class/{name}/f1', 'N/A')
        print(f"Class {class_id} ({name}): Precision={precision}, Recall={recall}, F1={f1_score}")
else:
    print("Warning: Unable to extract per-class performance metrics.")

if hasattr(results, 'confusion_matrix'):
    conf_matrix = results.confusion_matrix.matrix
    print("Confusion Matrix:\n", conf_matrix)
    if isinstance(conf_matrix, np.ndarray):
        plt.imshow(conf_matrix, cmap='Blues')
        plt.colorbar()
        plt.title("Confusion Matrix")
        plt.show()
    else:
        print("Warning: Confusion matrix is not in the expected format.")
else:
    print("Warning: Confusion matrix not found in results.")
