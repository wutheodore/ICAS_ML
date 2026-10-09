import torch

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from data_loader_v2 import val_loader
from resnet_model_v3 import ResNetClassifierV3


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD EXACT V3 MODEL
# ============================================================

model = ResNetClassifierV3(
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "best_resnet_v3_model.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\n")
print("=" * 60)
print("V3 VALIDATION DIAGNOSTIC")
print("=" * 60)


print("\nConfusion Matrix:")
print("                 Predicted")
print("                 ICAS  Non-ICAS")

print(
    f"Actual ICAS     {cm[0][0]:4d}  {cm[0][1]:8d}"
)

print(
    f"Actual Non-ICAS {cm[1][0]:4d}  {cm[1][1]:8d}"
)


# ============================================================
# METRICS
# ============================================================

accuracy = sum(
    true == pred
    for true, pred
    in zip(
        all_labels,
        all_predictions
    )
) / len(all_labels)


balanced_acc = balanced_accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score(
    all_labels,
    all_predictions,
    pos_label=0,
    zero_division=0
)


recall = recall_score(
    all_labels,
    all_predictions,
    pos_label=0,
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_predictions,
    pos_label=0,
    zero_division=0
)


print("\nOverall Metrics:")

print(
    f"Accuracy:          "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_acc * 100:.2f}%"
)

print(
    f"ICAS Precision:    "
    f"{precision * 100:.2f}%"
)

print(
    f"ICAS Recall:       "
    f"{recall * 100:.2f}%"
)

print(
    f"ICAS F1 Score:     "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "ICAS",
            "Non-ICAS"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

icas_predictions = sum(
    p == 0
    for p in all_predictions
)

non_icas_predictions = sum(
    p == 1
    for p in all_predictions
)


print("Prediction Distribution:")

print(
    f"Predicted ICAS:     "
    f"{icas_predictions}"
)

print(
    f"Predicted Non-ICAS: "
    f"{non_icas_predictions}"
)


print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETED")
print("=" * 60)