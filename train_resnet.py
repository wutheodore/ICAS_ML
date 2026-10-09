import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import train_loader, val_loader
from resnet_model import ResNetClassifier


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ==========================================
# MODEL
# ==========================================

model = ResNetClassifier(num_classes=2)
model = model.to(device)


# ==========================================
# CLASS WEIGHTS
# ==========================================

# ImageFolder mapping:
# 0 = ICAS
# 1 = Non-ICAS

icas_count = 212
non_icas_count = 452

total = icas_count + non_icas_count

weight_icas = total / (2 * icas_count)
weight_non_icas = total / (2 * non_icas_count)

class_weights = torch.tensor(
    [weight_icas, weight_non_icas],
    dtype=torch.float32
).to(device)

print("Class weights:", class_weights)


# ==========================================
# LOSS
# ==========================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


# ==========================================
# SCHEDULER
# ==========================================

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)


# ==========================================
# TRAINING SETTINGS
# ==========================================

NUM_EPOCHS = 25

best_val_accuracy = 0.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(NUM_EPOCHS):

    # ======================================
    # TRAIN
    # ======================================

    model.train()

    running_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        total_samples += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        100.0 * correct / total_samples
    )


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += loss.item()

            predictions = outputs.argmax(
                dim=1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = (
        val_running_loss / len(val_loader)
    )

    val_accuracy = (
        100.0 * val_correct / val_total
    )


    # ======================================
    # SAVE HISTORY
    # ======================================

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_resnet_model.pth"
        )


    # ======================================
    # UPDATE LR
    # ======================================

    scheduler.step()


    # ======================================
    # PRINT
    # ======================================

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']}"
    )

    print("-" * 60)


# ==========================================
# FINISHED
# ==========================================

print()
print("=" * 60)
print("RESNET18 TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best model saved as: "
    "best_resnet_model.pth"
)


# ==========================================
# LOSS CURVE
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "ResNet18 Training and Validation Loss"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "resnet_loss_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# ACCURACY CURVE
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "ResNet18 Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "resnet_accuracy_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()