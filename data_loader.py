import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import Counter

# ==========================================
# SETTINGS
# ==========================================

DATA_DIR = "dataset/split"

BATCH_SIZE = 16
IMAGE_SIZE = 224

# ==========================================
# DATA TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    # ImageNet normalization
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD DATASETS
# ==========================================

train_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/val",
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    f"{DATA_DIR}/test",
    transform=val_test_transform
)

# ==========================================
# SHOW CLASS MAPPING
# ==========================================

print("Class mapping:")
print(train_dataset.class_to_idx)

print()

# ==========================================
# CREATE DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# PRINT DATASET INFORMATION
# ==========================================

print("Dataset information:")
print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")

print()

print("Number of batches:")
print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches:       {len(test_loader)}")

# ==========================================
# CHECK CLASS DISTRIBUTION
# ==========================================

train_labels = [label for _, label in train_dataset.samples]

class_counts = Counter(train_labels)

print()
print("Training class distribution:")

for class_name, class_index in train_dataset.class_to_idx.items():
    print(
        f"{class_name}: "
        f"{class_counts[class_index]} images"
    )

# ==========================================
# TEST ONE BATCH
# ==========================================

images, labels = next(iter(train_loader))

print()
print("Sample batch:")
print(f"Image tensor shape: {images.shape}")
print(f"Label tensor shape: {labels.shape}")
print(f"Labels: {labels}")

print()
print("DataLoader setup completed successfully!")