from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# SETTINGS
# ============================================================

SPLIT_DIR = Path("dataset/split_image_level")

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20
PATIENCE = 5

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path(
    "experiment_27_region_resnet50_best.pth"
)


print("=" * 70)
print("EXPERIMENT 27 - REGION-BASED RESNET50")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATASET
# ============================================================

class ThermalDataset(Dataset):

    def __init__(self, split):

        self.samples = []

        split_path = SPLIT_DIR / split

        for class_name, label in [
            ("icas", 0),
            ("non_icas", 1)
        ]:

            folder = split_path / class_name

            files = (
                list(folder.glob("*.jpg"))
                + list(folder.glob("*.jpeg"))
                + list(folder.glob("*.png"))
            )

            for file in files:

                self.samples.append(
                    (file, label)
                )

        print(
            f"{split}: {len(self.samples)} images"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(
            path
        ).convert("RGB")

        image = transform(image)

        return image, label


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD DATA
# ============================================================

train_dataset = ThermalDataset(
    "train"
)

val_dataset = ThermalDataset(
    "val"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# REGION CREATION
# ============================================================

def create_regions(images):
    """
    Input:
        images = [B, 3, 224, 224]

    Returns:
        list of 4 tensors:
        whole face
        upper face
        middle face
        lower face
    """

    B, C, H, W = images.shape

    # --------------------------------------------------------
    # Whole face
    # --------------------------------------------------------

    whole = images


    # --------------------------------------------------------
    # Upper face
    # Approximately forehead + eyes
    # --------------------------------------------------------

    upper = images[
        :,
        :,
        0:int(H * 0.50),
        :
    ]


    # --------------------------------------------------------
    # Middle face
    # Approximately eyes + nose
    # --------------------------------------------------------

    middle = images[
        :,
        :,
        int(H * 0.25):int(H * 0.75),
        :
    ]


    # --------------------------------------------------------
    # Lower face
    # Approximately nose + mouth + chin
    # --------------------------------------------------------

    lower = images[
        :,
        :,
        int(H * 0.50):H,
        :
    ]


    regions = [
        whole,
        upper,
        middle,
        lower
    ]


    resized_regions = []

    for region in regions:

        region = torch.nn.functional.interpolate(
            region,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        resized_regions.append(
            region
        )

    return resized_regions


# ============================================================
# RESNET50 BACKBONE
# ============================================================

print("\nLoading pretrained ResNet50...")


weights = models.ResNet50_Weights.DEFAULT

backbone = models.resnet50(
    weights=weights
)


# Remove classifier
feature_size = backbone.fc.in_features

backbone.fc = nn.Identity()


# Freeze backbone
for param in backbone.parameters():
    param.requires_grad = False


backbone = backbone.to(DEVICE)

backbone.eval()


print(
    f"Feature size per region: {feature_size}"
)

print(
    "Backbone frozen."
)


# ============================================================
# REGION FUSION CLASSIFIER
# ============================================================

NUM_REGIONS = 4

combined_features = (
    feature_size * NUM_REGIONS
)


classifier = nn.Sequential(

    nn.Linear(
        combined_features,
        1024
    ),

    nn.ReLU(),

    nn.Dropout(0.50),

    nn.Linear(
        1024,
        256
    ),

    nn.ReLU(),

    nn.Dropout(0.30),

    nn.Linear(
        256,
        2
    )
)


classifier = classifier.to(
    DEVICE
)


print(
    f"Combined feature size: "
    f"{combined_features}"
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()


optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = DEVICE.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_region_features(images):

    regions = create_regions(
        images
    )

    features = []

    with torch.no_grad():

        for region in regions:

            region = region.to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):

                feature = backbone(
                    region
                )

            features.append(
                feature
            )

    combined = torch.cat(
        features,
        dim=1
    )

    return combined


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch():

    classifier.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        features = extract_region_features(
            images
        )


        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=use_amp
        ):

            outputs = classifier(
                features
            )

            loss = criterion(
                outputs,
                labels
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


    return (
        running_loss / total,
        correct / total
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate():

    classifier.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    running_loss = 0.0
    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            features = extract_region_features(
                images
            )


            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=use_amp
            ):

                outputs = classifier(
                    features
                )

                loss = criterion(
                    outputs,
                    labels
                )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predictions = outputs.argmax(
                dim=1
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            total += labels.size(0)


            all_labels.extend(
                labels.cpu().numpy()
            )


            all_predictions.extend(
                predictions.cpu().numpy()
            )


            # Class 0 = ICAS
            all_probabilities.extend(
                probabilities[:, 0]
                .cpu()
                .numpy()
            )


    val_loss = (
        running_loss / total
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    # For balanced accuracy sklearn expects
    # the actual class labels directly.
    balanced = balanced_accuracy_score(
        all_labels,
        all_predictions
    )


    y_true_icas = (
        np.array(all_labels) == 0
    ).astype(int)


    try:

        auroc = roc_auc_score(
            y_true_icas,
            all_probabilities
        )

    except:

        auroc = float("nan")


    return (
        val_loss,
        accuracy,
        balanced,
        auroc,
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_accuracy = -1.0
best_balanced = -1.0

best_epoch = 0

patience_counter = 0


print("\n")
print("=" * 70)
print("START TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_acc = (
        train_one_epoch()
    )


    (
        val_loss,
        val_acc,
        val_balanced,
        val_auroc,
        labels,
        predictions,
        probabilities
    ) = evaluate()


    scheduler.step()


    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_acc * 100:.2f}%"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_acc * 100:.2f}%"
    )

    print(
        f"Val BalAcc : "
        f"{val_balanced * 100:.2f}%"
    )

    print(
        f"Val AUROC  : "
        f"{val_auroc:.4f}"
    )


    # --------------------------------------------------------
    # Save based on validation accuracy
    # --------------------------------------------------------

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        best_balanced = val_balanced

        best_epoch = epoch

        patience_counter = 0


        torch.save(
            {
                "epoch": epoch,

                "classifier_state_dict":
                    classifier.state_dict(),

                "val_accuracy":
                    val_acc,

                "val_balanced_accuracy":
                    val_balanced,

                "val_auroc":
                    val_auroc
            },
            MODEL_PATH
        )


        print(
            "  *** BEST MODEL SAVED ***"
        )


    else:

        patience_counter += 1

        print(
            f"  No improvement "
            f"({patience_counter}/{PATIENCE})"
        )


    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# LOAD BEST
# ============================================================

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


classifier.load_state_dict(
    checkpoint[
        "classifier_state_dict"
    ]
)


(
    val_loss,
    val_acc,
    val_balanced,
    val_auroc,
    labels,
    predictions,
    probabilities
) = evaluate()


print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)


print(
    f"Validation Accuracy: "
    f"{val_acc * 100:.2f}%"
)


print(
    f"Validation Balanced Accuracy: "
    f"{val_balanced * 100:.2f}%"
)


print(
    f"Validation AUROC: "
    f"{val_auroc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        labels,
        predictions
    )
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        labels,
        predictions,
        target_names=[
            "icas",
            "non_icas"
        ],
        digits=4
    )
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\n")
print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)


threshold_results = []


for threshold in np.arange(
    0.10,
    0.91,
    0.01
):

    threshold_predictions = np.where(
        probabilities >= threshold,
        0,
        1
    )


    accuracy = accuracy_score(
        labels,
        threshold_predictions
    )


    balanced = balanced_accuracy_score(
        labels,
        threshold_predictions
    )


    threshold_results.append(
        (
            threshold,
            accuracy,
            balanced
        )
    )


best_threshold_accuracy = max(
    threshold_results,
    key=lambda x: x[1]
)


best_threshold_balanced = max(
    threshold_results,
    key=lambda x: x[2]
)


print(
    "\nBest threshold for accuracy:"
)

print(
    f"Threshold: "
    f"{best_threshold_accuracy[0]:.2f}"
)

print(
    f"Accuracy: "
    f"{best_threshold_accuracy[1] * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{best_threshold_accuracy[2] * 100:.2f}%"
)


print(
    "\nBest threshold for balanced accuracy:"
)

print(
    f"Threshold: "
    f"{best_threshold_balanced[0]:.2f}"
)

print(
    f"Accuracy: "
    f"{best_threshold_balanced[1] * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{best_threshold_balanced[2] * 100:.2f}%"
)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 27 COMPLETE")
print("=" * 70)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation accuracy: "
    f"{val_acc * 100:.2f}%"
)

print(
    f"Best validation balanced accuracy: "
    f"{val_balanced * 100:.2f}%"
)

print(
    f"Best validation AUROC: "
    f"{val_auroc:.4f}"
)

print(
    f"Model saved: {MODEL_PATH}"
)

print(
    "\nTEST SET WAS NOT USED."
)

print("=" * 70)