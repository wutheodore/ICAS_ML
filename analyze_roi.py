import os
import cv2
import numpy as np
from PIL import Image

DATASET = r"dataset\datasets\thermal_classification_cropped"

classes = ["icas", "non_icas"]


def analyze_image(path):
    img = cv2.imread(path)

    if img is None:
        return None

    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pixels that are not nearly black
    mask = gray > 10

    if not np.any(mask):
        return None

    ys, xs = np.where(mask)

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    width = x2 - x1 + 1
    height = y2 - y1 + 1

    image_h, image_w = gray.shape

    coverage = mask.mean()

    return {
        "width": image_w,
        "height": image_h,
        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2,
        "roi_width": width,
        "roi_height": height,
        "coverage": coverage,
        "mean_brightness": gray.mean(),
    }


print("=" * 70)
print("THERMAL IMAGE ROI ANALYSIS")
print("=" * 70)

all_results = []

for class_name in classes:

    folder = os.path.join(DATASET, class_name)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print()
    print(f"{class_name.upper()}: {len(files)} images")

    results = []

    for filename in files:

        path = os.path.join(folder, filename)

        result = analyze_image(path)

        if result is not None:
            results.append(result)

    if not results:
        continue

    roi_widths = [r["roi_width"] for r in results]
    roi_heights = [r["roi_height"] for r in results]
    coverages = [r["coverage"] for r in results]

    print()
    print("ROI width:")
    print(f"  Mean:   {np.mean(roi_widths):.1f}")
    print(f"  Median: {np.median(roi_widths):.1f}")
    print(f"  Min:    {np.min(roi_widths)}")
    print(f"  Max:    {np.max(roi_widths)}")

    print()
    print("ROI height:")
    print(f"  Mean:   {np.mean(roi_heights):.1f}")
    print(f"  Median: {np.median(roi_heights):.1f}")
    print(f"  Min:    {np.min(roi_heights)}")
    print(f"  Max:    {np.max(roi_heights)}")

    print()
    print("Non-black pixel coverage:")
    print(f"  Mean:   {np.mean(coverages):.4f}")
    print(f"  Median: {np.median(coverages):.4f}")
    print(f"  Min:    {np.min(coverages):.4f}")
    print(f"  Max:    {np.max(coverages):.4f}")


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)