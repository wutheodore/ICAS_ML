import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\Marcellynno\ICAS_ML"

GROUPED_VAL_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "split_grouped",
    "val"
)

RESNET_PATH = os.path.join(
    BASE_DIR,
    "experiment_19_resnet50_best.pth"
)

DENSENET_PATH = os.path.join(
    BASE_DIR,
    "experiment_21_densenet121_best.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Exact V22 settings
RESNET_WEIGHT = 0.50
DENSENET_WEIGHT = 0.50
THRESHOLD = 0.46

BATCH_SIZE = 16

print("=" * 70)
print("EXPERIMENT 28 - GROUPED VALIDATION GENERALIZATION TEST")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATASET
# ============================================================

class ThermalDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.samples = []

        # IMPORTANT:
        # class 0 = ICAS
        # class 1 = Non-ICAS

        class_names = ["icas", "non_icas"]

        for label, class_name in enumerate(class_names):

            class_dir = os.path.join(root_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"WARNING: folder not found: {class_dir}")
                continue

            for filename in os.listdir(class_dir):

                filepath = os.path.join(class_dir, filename)

                if os.path.isfile(filepath):

                    self.samples.append(
                        (filepath, label, filename)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        filepath, label, filename = self.samples[index]

        image = Image.open(filepath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, filename


# ============================================================
# TRANSFORM
# EXACTLY MATCH V22
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD GROUPED VALIDATION DATA
# ============================================================

dataset = ThermalDataset(
    GROUPED_VAL_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print("Grouped validation dataset:")
print(f"Total images: {len(dataset)}")

icas_count = sum(
    1 for _, label, _ in dataset.samples if label == 0
)

non_icas_count = sum(
    1 for _, label, _ in dataset.samples if label == 1
)

print(f"ICAS: {icas_count}")
print(f"Non-ICAS: {non_icas_count}")


# ============================================================
# BUILD RESNET50
# EXACT V19 ARCHITECTURE
# ============================================================

print()
print("Loading ResNet50...")

resnet = models.resnet50(weights=None)

# Unfreeze architecture used in V19
resnet.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

resnet_checkpoint = torch.load(
    RESNET_PATH,
    map_location=DEVICE
)

resnet.load_state_dict(
    resnet_checkpoint["model_state_dict"]
)

resnet = resnet.to(DEVICE)
resnet.eval()


# ============================================================
# BUILD DENSENET121
# EXACT V21 ARCHITECTURE
# ============================================================

print("Loading DenseNet121...")

densenet = models.densenet121(weights=None)

densenet.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)
)

densenet_checkpoint = torch.load(
    DENSENET_PATH,
    map_location=DEVICE
)

densenet.load_state_dict(
    densenet_checkpoint["model_state_dict"]
)

densenet = densenet.to(DEVICE)
densenet.eval()


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_resnet_probs = []
all_densenet_probs = []
all_ensemble_probs = []
all_filenames = []

print()
print("Running grouped validation evaluation...")

with torch.no_grad():

    for images, labels, filenames in loader:

        images = images.to(DEVICE)

        # ----------------------------------------------------
        # ResNet50
        # ----------------------------------------------------

        resnet_logits = resnet(images)

        resnet_probs = torch.softmax(
            resnet_logits,
            dim=1
        )[:, 0]

        # ----------------------------------------------------
        # DenseNet121
        # ----------------------------------------------------

        densenet_logits = densenet(images)

        densenet_probs = torch.softmax(
            densenet_logits,
            dim=1
        )[:, 0]

        # ----------------------------------------------------
        # V22 ENSEMBLE
        # ----------------------------------------------------

        ensemble_probs = (
            RESNET_WEIGHT * resnet_probs +
            DENSENET_WEIGHT * densenet_probs
        )

        all_labels.extend(
            labels.numpy()
        )

        all_resnet_probs.extend(
            resnet_probs.cpu().numpy()
        )

        all_densenet_probs.extend(
            densenet_probs.cpu().numpy()
        )

        all_ensemble_probs.extend(
            ensemble_probs.cpu().numpy()
        )

        all_filenames.extend(filenames)


# ============================================================
# CONVERT ARRAYS
# ============================================================

y_true = np.array(all_labels)

resnet_probs = np.array(all_resnet_probs)

densenet_probs = np.array(all_densenet_probs)

ensemble_probs = np.array(all_ensemble_probs)


# ============================================================
# IMPORTANT:
# CLASS 0 = ICAS
# CLASS 1 = NON-ICAS
#
# Therefore:
# probability >= threshold => ICAS
# probability < threshold  => Non-ICAS
# ============================================================

y_pred = np.where(
    ensemble_probs >= THRESHOLD,
    0,
    1
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_true,
    y_pred
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

try:

    auroc = roc_auc_score(
        (y_true == 0).astype(int),
        ensemble_probs
    )

except Exception:

    auroc = float("nan")


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("EXPERIMENT 28 RESULTS")
print("=" * 70)

print(f"Accuracy:          {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"AUROC:             {auroc:.4f}")

print()
print("Confusion Matrix")
print("Rows = Actual")
print("Columns = Predicted")
print()
print("                 Pred ICAS   Pred Non-ICAS")
print(
    f"Actual ICAS       {cm[0,0]:>6}       {cm[0,1]:>6}"
)
print(
    f"Actual Non-ICAS   {cm[1,0]:>6}       {cm[1,1]:>6}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("Classification Report:")
print()

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "ICAS",
            "Non-ICAS"
        ],
        digits=4
    )
)


# ============================================================
# MODEL COMPARISON
# ============================================================

print()
print("=" * 70)
print("COMPARISON WITH ORIGINAL V22 IMAGE-LEVEL VALIDATION")
print("=" * 70)

print()
print("V22 image-level validation:")
print("Accuracy:          73.24%")
print("Balanced Accuracy: 60.16%")
print("AUROC:             0.6030")

print()
print("V22 grouped validation:")
print(f"Accuracy:          {accuracy * 100:.2f}%")
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
print(f"AUROC:             {auroc:.4f}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

output_file = os.path.join(
    BASE_DIR,
    "experiment_28_grouped_v22_predictions.csv"
)

import csv

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "true_label",
        "resnet_icas_probability",
        "densenet_icas_probability",
        "ensemble_icas_probability",
        "predicted_label"
    ])

    for i in range(len(y_true)):

        true_name = (
            "ICAS"
            if y_true[i] == 0
            else "Non-ICAS"
        )

        pred_name = (
            "ICAS"
            if y_pred[i] == 0
            else "Non-ICAS"
        )

        writer.writerow([
            all_filenames[i],
            true_name,
            f"{resnet_probs[i]:.6f}",
            f"{densenet_probs[i]:.6f}",
            f"{ensemble_probs[i]:.6f}",
            pred_name
        ])


print()
print(f"Predictions saved to:")
print(output_file)

print()
print("=" * 70)
print("EXPERIMENT 28 COMPLETE")
print("=" * 70)

print()
print("IMPORTANT:")
print("- Grouped validation was NOT used for training.")
print("- Threshold was NOT optimized on grouped validation.")
print("- Test set was NOT used.")
print("- This is a generalization check.")