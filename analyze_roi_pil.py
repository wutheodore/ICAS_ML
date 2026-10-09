from pathlib import Path
from PIL import Image
import numpy as np
import csv

DATASET = Path("dataset/datasets/thermal_classification_cropped")

classes = ["icas", "non_icas"]

all_results = []
failed = []

for class_name in classes:
    folder = DATASET / class_name

    files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))

    print()
    print("=" * 70)
    print(f"{class_name.upper()}: {len(files)} images found")
    print("=" * 70)

    success = 0

    for path in files:
        try:
            # IMPORTANT:
            # PIL reads the file directly and avoids cv2.imread()
            with Image.open(path) as img:
                img = img.convert("RGB")
                rgb = np.array(img)

            # Convert RGB to grayscale
            gray = np.mean(rgb, axis=2)

            # Pixels brighter than 10 are considered non-background
            mask = gray > 10

            if not np.any(mask):
                failed.append((class_name, path.name, "all pixels are black"))
                continue

            ys, xs = np.where(mask)

            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())

            roi_width = x_max - x_min + 1
            roi_height = y_max - y_min + 1

            image_height, image_width = gray.shape

            coverage = (
                mask.sum()
                / (image_width * image_height)
            )

            all_results.append({
                "class": class_name,
                "filename": path.name,
                "image_width": image_width,
                "image_height": image_height,
                "roi_width": roi_width,
                "roi_height": roi_height,
                "coverage": coverage
            })

            success += 1

        except Exception as e:
            failed.append((class_name, path.name, str(e)))

    print(f"Successfully read: {success}/{len(files)}")


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL FILE READING CHECK")
print("=" * 70)

icas_total = sum(1 for r in all_results if r["class"] == "icas")
non_icas_total = sum(1 for r in all_results if r["class"] == "non_icas")

print(f"ICAS successfully read:     {icas_total}/303")
print(f"Non-ICAS successfully read: {non_icas_total}/647")
print(f"TOTAL successfully read:    {len(all_results)}/950")
print(f"TOTAL failed:               {len(failed)}")


# ============================================================
# ROI STATISTICS
# ============================================================

for class_name in classes:

    rows = [
        r for r in all_results
        if r["class"] == class_name
    ]

    if not rows:
        continue

    widths = np.array([r["roi_width"] for r in rows])
    heights = np.array([r["roi_height"] for r in rows])
    coverages = np.array([r["coverage"] for r in rows])

    print()
    print("-" * 70)
    print(class_name.upper())
    print("-" * 70)

    print("ROI width:")
    print(f"  Mean:   {widths.mean():.1f}")
    print(f"  Median: {np.median(widths):.1f}")
    print(f"  Min:    {widths.min()}")
    print(f"  Max:    {widths.max()}")

    print()
    print("ROI height:")
    print(f"  Mean:   {heights.mean():.1f}")
    print(f"  Median: {np.median(heights):.1f}")
    print(f"  Min:    {heights.min()}")
    print(f"  Max:    {heights.max()}")

    print()
    print("Non-black pixel coverage:")
    print(f"  Mean:   {coverages.mean():.4f}")
    print(f"  Median: {np.median(coverages):.4f}")
    print(f"  Min:    {coverages.min():.4f}")
    print(f"  Max:    {coverages.max():.4f}")


# ============================================================
# SAVE CSV
# ============================================================

output_file = "roi_analysis_pil.csv"

with open(output_file, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "filename",
            "image_width",
            "image_height",
            "roi_width",
            "roi_height",
            "coverage"
        ]
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# FAILED FILES
# ============================================================

if failed:

    with open(
        "roi_failed_files.txt",
        "w",
        encoding="utf-8"
    ) as f:

        for class_name, filename, error in failed:
            f.write(
                f"{class_name} | {filename} | {error}\n"
            )

    print()
    print(f"Failed filenames saved to: roi_failed_files.txt")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Results saved to: {output_file}")