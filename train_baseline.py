import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import train_loader, val_loader
from baseline_cnn import CustomCNN


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

model = CustomCNN(num_classes=2)
model = model.to(device)


# ==========================================
# CLASS WEIGHTS
# ==========================================

# Our ImageFolder mapping is:
# ICAS = 0
# Non-ICAS = 1

# Training distribution:
# ICAS = 212
# Non-ICAS = 452

class_weights = torch.tensor(
    [452 / 664, 212 / 664],
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
# LEARNING RATE SCHEDULER
# ==========================================

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=15,
    gamma=0.1
)


# ==========================================
# TRAINING SETTINGS
# ==========================================

NUM_EPOCHS = 30

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

    # --------------------------------------
    # TRAINING
    # --------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

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

        _, predictions = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # --------------------------------------
    # VALIDATION
    # --------------------------------------

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

            _, predictions = torch.max(
                outputs,
                1
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


    # --------------------------------------
    # SAVE HISTORY
    # --------------------------------------

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


    # --------------------------------------
    # LEARNING RATE
    # --------------------------------------

    scheduler.step()


    # --------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_baseline_model.pth"
        )


    # --------------------------------------
    # PRINT RESULTS
    # --------------------------------------

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
# TRAINING FINISHED
# ==========================================

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best model saved as: "
    "best_baseline_model.pth"
)


# ==========================================
# PLOT TRAINING CURVES
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
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "baseline_loss_curve.png",
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
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "baseline_accuracy_curve.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()