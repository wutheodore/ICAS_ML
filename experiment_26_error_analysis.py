from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import csv
import shutil
import math


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = Path("experiment_25_v22_predictions.csv")

OUTPUT_DIR = Path("experiment_26_error_analysis")

IMAGE_SIZE = 224
COLUMNS = 5

# Maximum images per category in contact sheet
MAX_PER_CATEGORY = 50


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("EXPERIMENT 26 - V22 VISUAL ERROR ANALYSIS")
print("=" * 70)

if not CSV_FILE.exists():

    print(
        f"\nERROR: {CSV_FILE} was not found."
    )

    print(
        "Run experiment_25_diagnose_v22.py first."
    )

    raise SystemExit


OUTPUT_DIR.mkdir(
    exist_ok=True
)


categories = [
    "correct_icas",
    "missed_icas",
    "correct_non_icas",
    "false_positive_non_icas"
]


for category in categories:

    folder = OUTPUT_DIR / category

    folder.mkdir(
        exist_ok=True
    )


# ============================================================
# READ CSV
# ============================================================

rows = []

with open(
    CSV_FILE,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        rows.append(row)


print(
    f"\nLoaded {len(rows)} validation predictions."
)


# ============================================================
# CLASSIFY ERRORS
# ============================================================

category_images = {
    category: []
    for category in categories
}


for row in rows:

    actual = int(row["actual"])
    predicted = int(row["predicted"])

    filename = row["filename"]

    if actual == 0 and predicted == 0:

        category = "correct_icas"

    elif actual == 0 and predicted == 1:

        category = "missed_icas"

    elif actual == 1 and predicted == 1:

        category = "correct_non_icas"

    elif actual == 1 and predicted == 0:

        category = "false_positive_non_icas"

    else:

        continue


    category_images[category].append(
        row
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("CATEGORY SUMMARY")
print("=" * 70)

for category in categories:

    count = len(
        category_images[category]
    )

    print(
        f"{category:25s}: {count}"
    )


# ============================================================
# COPY IMAGES
# ============================================================

print("\n")
print("=" * 70)
print("COPYING IMAGES")
print("=" * 70)


for category in categories:

    items = category_images[category]

    output_folder = (
        OUTPUT_DIR / category
    )

    for number, row in enumerate(
        items,
        start=1
    ):

        source = Path(
            row["filename"]
        )

        if not source.exists():

            print(
                f"WARNING: Missing {source}"
            )

            continue


        destination = (
            output_folder
            / f"{number:03d}_{source.name}"
        )

        shutil.copy2(
            source,
            destination
        )


    print(
        f"{category}: {len(items)} copied"
    )


# ============================================================
# CONTACT SHEET FUNCTION
# ============================================================

def create_contact_sheet(
    items,
    category,
    output_file
):

    if len(items) == 0:

        return


    items = items[
        :MAX_PER_CATEGORY
    ]


    rows_needed = math.ceil(
        len(items) / COLUMNS
    )


    label_height = 60

    sheet_width = (
        COLUMNS * IMAGE_SIZE
    )

    sheet_height = (
        rows_needed
        * (IMAGE_SIZE + label_height)
    )


    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )


    draw = ImageDraw.Draw(
        sheet
    )


    for index, row in enumerate(
        items
    ):

        image_path = Path(
            row["filename"]
        )


        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = image.resize(
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                )
            )

        except Exception as e:

            print(
                f"Could not open "
                f"{image_path}: {e}"
            )

            continue


        column = index % COLUMNS

        sheet_row = index // COLUMNS


        x = (
            column
            * IMAGE_SIZE
        )

        y = (
            sheet_row
            * (IMAGE_SIZE + label_height)
        )


        sheet.paste(
            image,
            (x, y)
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        actual = (
            "ICAS"
            if int(row["actual"]) == 0
            else "Non"
        )

        predicted = (
            "ICAS"
            if int(row["predicted"]) == 0
            else "Non"
        )


        probability = float(
            row["ensemble_p_icas"]
        )


        text = (
            f"{index + 1}: "
            f"A={actual} P={predicted}\n"
            f"P(ICAS)={probability:.2f}"
        )


        draw.text(
            (
                x + 3,
                y + IMAGE_SIZE + 3
            ),
            text,
            fill="black"
        )


    sheet.save(
        output_file
    )


# ============================================================
# CREATE CONTACT SHEETS
# ============================================================

print("\n")
print("=" * 70)
print("CREATING CONTACT SHEETS")
print("=" * 70)


for category in categories:

    output_file = (
        OUTPUT_DIR
        / f"{category}_contact_sheet.jpg"
    )


    create_contact_sheet(
        category_images[category],
        category,
        output_file
    )


    print(
        f"Created: {output_file}"
    )


# ============================================================
# CREATE DETAILED CSV
# ============================================================

details_file = (
    OUTPUT_DIR
    / "error_analysis_details.csv"
)


with open(
    details_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "category",
        "filename",
        "actual",
        "predicted",
        "resnet_p_icas",
        "densenet_p_icas",
        "ensemble_p_icas"
    ]


    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )


    writer.writeheader()


    for category in categories:

        for row in category_images[
            category
        ]:

            writer.writerow({
                "category": category,
                "filename": row["filename"],
                "actual": row["actual"],
                "predicted": row["predicted"],
                "resnet_p_icas":
                    row["resnet_p_icas"],
                "densenet_p_icas":
                    row["densenet_p_icas"],
                "ensemble_p_icas":
                    row["ensemble_p_icas"]
            })


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 70)
print("EXPERIMENT 26 COMPLETE")
print("=" * 70)

print(
    f"\nOutput folder:"
)

print(
    OUTPUT_DIR
)

print(
    "\nContact sheets:"
)

for category in categories:

    print(
        f"  {category}_contact_sheet.jpg"
    )

print(
    "\nDetailed results:"
)

print(
    details_file
)

print(
    "\nTEST SET WAS NOT USED."
)