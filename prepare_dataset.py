import os
import random
import shutil
from pathlib import Path

# =========================
# SETTINGS
# =========================

SOURCE_DIR = Path("dataset/datasets/thermal_classification_cropped")
OUTPUT_DIR = Path("dataset/split")

CLASSES = ["non_icas", "icas"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

# =========================
# CHECK RATIOS
# =========================

assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0

random.seed(SEED)

# =========================
# CHECK SOURCE DATASET
# =========================

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {SOURCE_DIR}"
    )

print("Source dataset found:")
print(SOURCE_DIR)
print()

# =========================
# CREATE OUTPUT FOLDERS
# =========================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        folder = OUTPUT_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

# =========================
# SPLIT DATA
# =========================

for class_name in CLASSES:

    class_dir = SOURCE_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class folder not found: {class_dir}"
        )

    images = [
        file for file in class_dir.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"{class_name}:")
    print(f"  Total:      {len(images)}")
    print(f"  Train:      {len(train_images)}")
    print(f"  Validation: {len(val_images)}")
    print(f"  Test:       {len(test_images)}")
    print()

    # Copy files
    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / class_name / image.name
        )

    for image in val_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "val" / class_name / image.name
        )

    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / class_name / image.name
        )

print("===================================")
print("Dataset split completed successfully!")
print("===================================")
print()
print(f"Output directory: {OUTPUT_DIR}")